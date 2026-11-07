"""扫描件识别与 OCR 降级：先判断「有没有文字层」，再决定能不能识别。

为什么需要这个模块？
比赛要求用一张扫描版论文（1706.03762v1_img.pdf）验证「OCR 识别与结构化还原」。
扫描件里每个字都是像素，`page.get_text()` 只能拿到空串，原来的解析链会
一路走到空结果然后 422 —— 用户看到的是「解析失败」，却不知道失败的原因是
「这份 PDF 没有文字层」，也不知道本机压根没装 OCR 引擎。

本模块把这件事拆成两步，绝不假装成功：
1. `analyze()`：逐页统计文字层字符数，明确指出哪些页是扫描件；
   即使文件损坏也只返回报告，不把异常抛给 API 层。
2. `ocr_pages()`：OCR 引擎真的可用时才动手；单页失败就跳过继续，
   扫描件常有坏页，不能因为一页让整份文档失败；引擎不可用时返回空字典，
   由调用方把「无法识别」原样告诉用户。

判定口径：`get_text("text")` 去掉所有空白后不足 30 个字符的页算扫描页。
30 是经验阈值：真实文字页的正文远大于它，而扫描页通常只剩页码或几个
看不见的控制字符。

为什么是 RapidOCR（PP-OCRv4 的 ONNX 版）而不是 Tesseract：
Tesseract 要另装可执行文件与语言包，评委机器上装不上就等于这条用例直接 0 分；
RapidOCR 是纯 wheel，`uv sync` 完就能跑，中英文同一个模型，不用挑语言参数。

它有三个必须自己收拾的毛病，都在下面单独说明：
1. **丢空格** —— "Attention Is All You Need" 会读成 "AttentionIsAllYouNeed"。
2. **读序会串栏** —— 双栏论文的识别结果按检测框顺序返回，左右栏会交错。
3. **慢** —— 一页论文正文要 6~9 秒（CPU 上就是这么慢），几百页的扫描教材得先截断。
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from statistics import median

import pymupdf

# 一页文字层去掉空白后少于这么多字符，就认为它没有可用的文字层。
MIN_PAGE_TEXT_CHARS = 30
# OCR 的分辨率。72 dpi 是量出来的峰值，而且反直觉 —— 分辨率越高越差：
#   48dpi F1 0.476 · 60dpi 0.835 · **72dpi 0.866** · 96dpi 0.827 · 120dpi 0.786 · 200dpi 0.539
# 分辨率高时模型反而把词粘起来（粘连率 72dpi 0.2% → 120dpi 4.6% → 150dpi 9.6%），
# 因为裁剪行统一缩放到 48 px 高，原图越大则被压得越狠、字间空白先被抹平。
# 识别耗时与 dpi 基本无关。数据与复现命令见 docs/EXPERIMENTS.md E-018。
DEFAULT_DPI = 72
# 实测每页约 9 秒（PP-OCRv4 在 CPU 上就是这么慢），40 页要 6 分钟。
# 默认 20 页：扫描版论文（15 页）能整份认完，教材如实标注"只认了前 20 页"。
MAX_OCR_PAGES = 20
# 同时识别的页数。15 页扫描件实测：串行 176s → 4 路 117s（1.5 倍），
# 再往上没有收益（8 路 109s，而 6 路反而是 170s，跑两遍差一倍）。
# 一页之内 onnxruntime 就已经把核占满了，多页并发只是在抢同一批核。
PARALLEL_PAGES = 4
# 行距超过行高的这个倍数就断段。正文行距通常是 1.2–1.5 倍字高，
# 段间距明显更大，0.85 这个阈值（以行高计，而非字高）能分开两种情形。
PARA_GAP_FACTOR = 0.85
# 标题式短行的判据：宽度不到本页中位行宽的这个比例，且词数不超过上限。
# 见 `_standalone_line` —— 扫描件上这是 `References` 唯一能被认出来的机会。
HEADING_LINE_WIDTH_RATIO = 0.5
HEADING_LINE_MAX_WORDS = 5
# XY-cut 的递归深度上限。正常版面两三层就切完了；不封顶的话，
# 一张被切成上千行的散乱图片会把递归拖成栈溢出。
MAX_CUT_DEPTH = 8
# 横切的缝至少要有一个行高的这个比例，否则那只是相邻两行的正常行距。
MIN_CUT_RATIO = 0.35
# 一行文字的宽高比下限。横排正文行普遍在 10 以上，最窄的单个数字也有 3 倍；
# 宽高比不到 2 的只可能是竖排/旋转文字（投稿戳、竖排标签），整行丢掉。
MIN_LINE_ASPECT = 2.0

@dataclass(slots=True)
class OcrWord:
    """OCR 读出来的一个词：文本 + 在页面坐标系（pt）上的框。"""

    text: str
    bbox: tuple[float, float, float, float]


@dataclass(slots=True)
class OcrLine:
    """OCR 读出来的一行。`size` 用行的像素高度当字号代理。"""

    text: str
    bbox: tuple[float, float, float, float]
    words: list[OcrWord]
    size: float


@dataclass(slots=True)
class ScanReport:
    """一份 PDF 的文字层体检报告。"""

    pages: int                 # 总页数
    readable_pages: int        # 有可用文字层的页数
    scanned_pages: list[int]   # 判定为扫描件的页码（1 起）
    text_chars: int            # 文字层总字符数（去空白后）
    ocr_available: bool        # 本机能不能真的 OCR
    note: str                  # 给人看的一句中文说明


def _visible_chars(text: str) -> int:
    """文字层去掉空白后的字符数。

    只看非空白字符，因为扫描件里偶尔也会有一两个空格或换行符；
    按原始长度统计会把「空页」误判成「有文字」。
    """
    return len("".join(text.split()))


# ------------------------------------------------------------------ 引擎


def _engine_disabled() -> bool:
    """`RAG_OCR=off` 可以在测试或不想要 OCR 的环境里彻底关掉。

    关掉时不只是不调用，连模型都不加载 —— RapidOCR 首次构造要 3.2 秒，
    纯粹解析文字层 PDF 的进程不该付这笔钱。
    """
    return os.environ.get("RAG_OCR", "").strip().lower() in {"0", "off", "false", "no"}


@functools.lru_cache(maxsize=1)
def _engine():
    """进程内单例。首次构造要 3.2 秒（加载两个 ONNX 模型），之后每页才算推理。

    任何一步失败都返回 None 而不是抛异常：OCR 是**可选的增强**，
    没有它系统仍然要能跑，只是如实标注"这份扫描件读不了"。
    """
    if _engine_disabled():
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:  # noqa: BLE001 —— 没装、装坏了、模型缺失都走这里
        return None
    try:
        return RapidOCR()
    except Exception:  # noqa: BLE001
        return None


def _ocr_available() -> bool:
    """本机能否真的执行 OCR：引擎能构造出来就算可用。"""
    return _engine() is not None


@functools.lru_cache(maxsize=1)
def ocr_enabled() -> bool:
    """`_ocr_available` 的缓存版。切块时会按页问，几百页不能每页都试一遍加载。"""
    return _ocr_available()


def _short(error: Exception) -> str:
    """把异常压成一行，避免把多行堆栈塞进给人看的 note。"""
    return " ".join(str(error).split())[:160]


# ------------------------------------------------------------------ 识别


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _words_from_text(
    text: str, box: tuple[int, int, int, int], scale: float
) -> list[OcrWord]:
    """按模型自己吐的空格切词，词框按字数据行宽均分。

    为什么不再自己从像素里找空格：试过，不如它。PP-OCR 吃掉的空格可以用
    墨迹投影（逐列统计深色像素、找贯通空隙）补回来，实测也确实能补一部分，
    但**整体更差** —— 当时（默认还是 120 dpi）词级 F1 从 0.789 掉到 0.529：
    抗锯齿会把一个字母劈成两三段，段与段之间 5~6 px 的缝和真正的词间隔
    一样宽，于是空格被插进词中间（"performanc e"）。自适应门槛、加大对齐
    带宽都救不回来。模型自己的空格才是不折腾的那个答案。
    （测法与数字见 docs/EXPERIMENTS.md E-018。）

    代价是词框不准：这里按字数均分整行宽度，只够画高亮，不够做精确定位。
    """
    x0, y0, x1, y1 = (v * scale for v in box)
    pieces = text.split()
    if not pieces:
        return []
    step = (x1 - x0) / max(sum(len(p) + 1 for p in pieces), 1)
    out, cursor = [], x0
    for piece in pieces:
        out.append(OcrWord(piece, (cursor, y0, cursor + step * len(piece), y1)))
        cursor += step * (len(piece) + 1)
    return out


def _reading_order(lines: list[OcrLine]) -> list[OcrLine]:
    """把检测框排成人的阅读顺序。

    为什么需要：RapidOCR 返回的顺序是检测先后，双栏论文会读成
    「左栏第一行 → 右栏第一行 → 左栏第二行 …」，正文彻底错乱，
    下游的切块、检索、引用全都建在错乱文本上。

    做法是递归 XY-cut：每一步在「横切」和「竖切」里挑缝更宽的那个。
    - 横切：找一条没有任何行跨过的水平空白带（行与行之间的大间距）。
      **先看横切**很关键：论文的章节标题、摘要、图注都是通栏的，
      它们会把竖缝拦腰截断，于是整页都切不开、只能按 y 排序，两栏照样交错。
      先横切把通栏的标题带剥离出去，剩下的双栏区域才露出干净的竖缝。
    - 竖切：找一条没有任何行跨过的竖直空白带（栏间距）。

    递归到切不开为止；叶子按「上到下、左到右」排。
    """
    return _xy_cut(list(lines), depth=0)


def _row_sort(lines: list[OcrLine]) -> list[OcrLine]:
    """按「排」排：先把纵坐标相近的行并成一排，排内再从左到右。

    不能直接按 `(y, x)`：论文首页四个作者并排，扫描后基线常差 0.5~2 pt，
    直接比 y 会把他们按差值大小打乱 —— 实测读成「Jakob、Noam、Ashish、Niki」，
    正确顺序是 Ashish、Noam、Niki、Jakob。
    """
    if len(lines) <= 1:
        return list(lines)
    ordered = sorted(lines, key=lambda line: (line.bbox[1] + line.bbox[3]) / 2)
    median_h = sorted(line.bbox[3] - line.bbox[1] for line in lines)[len(lines) // 2] or 1.0
    tolerance = max(median_h * 0.5, 1.0)

    rows: list[list[OcrLine]] = [[ordered[0]]]
    for line in ordered[1:]:
        center = (line.bbox[1] + line.bbox[3]) / 2
        prev = (rows[-1][-1].bbox[1] + rows[-1][-1].bbox[3]) / 2
        if center - prev > tolerance:
            rows.append([line])
        else:
            rows[-1].append(line)

    out: list[OcrLine] = []
    for row in rows:
        out.extend(sorted(row, key=lambda line: line.bbox[0]))
    return out


def _xy_cut(lines: list[OcrLine], *, depth: int) -> list[OcrLine]:
    if len(lines) <= 1 or depth >= MAX_CUT_DEPTH:
        return _row_sort(lines)

    horizontal = _widest_horizontal_gap(lines)
    vertical = _widest_vertical_gap(lines)
    if horizontal is None and vertical is None:
        return _row_sort(lines)

    # 两种缝都能切时，按「缝宽 / 行高」比谁更像版面分隔：横切靠行距，
    # 竖切靠栏间距，单位不同，用行高归一化后才好比较。
    median_h = sorted(line.bbox[3] - line.bbox[1] for line in lines)[len(lines) // 2] or 1.0
    h_score = (horizontal[0] / median_h) if horizontal else 0.0
    v_score = (vertical[0] / median_h) if vertical else 0.0

    if horizontal is not None and h_score >= v_score:
        cut = horizontal[1]
        top = [line for line in lines if line.bbox[3] <= cut]
        bottom = [line for line in lines if line.bbox[1] >= cut]
        return _xy_cut(top, depth=depth + 1) + _xy_cut(bottom, depth=depth + 1)

    cut = vertical[1]
    left = [line for line in lines if line.bbox[2] <= cut]
    right = [line for line in lines if line.bbox[0] >= cut]
    return _xy_cut(left, depth=depth + 1) + _xy_cut(right, depth=depth + 1)


def _clean_cut(
    intervals: list[tuple[float, float]], *, minimum: float, lo: float, hi: float
) -> float | None:
    """在 `[lo, hi]` 里找最宽的一条「没有任何区间跨过」的缝，返回缝的中点。

    区间是闭的（某行的 x0..x1）。缝必须**严格**落在两个相邻区间之间：
    有行跨过哪怕 1 pt，切下去就会把那行劈成两半，读序反而更乱。
    """
    ordered = sorted(intervals)
    best: float | None = None
    best_width = 0.0
    reach = ordered[0][1]
    for start, end in ordered[1:]:
        gap = start - reach
        if gap >= minimum:
            center = (reach + start) / 2
            if lo <= center <= hi and gap > best_width:
                best, best_width = center, gap
        reach = max(reach, end)
    return best


def _widest_horizontal_gap(lines: list[OcrLine]) -> tuple[float, float] | None:
    """最宽的水平空白带：返回 (宽度, 切点 y)。"""
    y0 = min(line.bbox[1] for line in lines)
    y1 = max(line.bbox[3] for line in lines)
    median_h = sorted(line.bbox[3] - line.bbox[1] for line in lines)[len(lines) // 2] or 1.0
    cut = _clean_cut(
        [(line.bbox[1], line.bbox[3]) for line in lines],
        minimum=max(median_h * MIN_CUT_RATIO, 2.0),
        lo=y0,
        hi=y1,
    )
    if cut is None:
        return None
    below = [line.bbox[1] for line in lines if line.bbox[1] >= cut]
    above = [line.bbox[3] for line in lines if line.bbox[3] <= cut]
    if not below or not above:
        return None
    return (min(below) - max(above), cut)


def _widest_vertical_gap(lines: list[OcrLine]) -> tuple[float, float] | None:
    """最宽的竖直空白带：返回 (宽度, 切点 x)。"""
    x0 = min(line.bbox[0] for line in lines)
    x1 = max(line.bbox[2] for line in lines)
    median_h = sorted(line.bbox[3] - line.bbox[1] for line in lines)[len(lines) // 2] or 1.0
    # 栏间距至少 1.5 个行高才算数：字间距量级的缝到处都是，切了只会把行切碎
    cut = _clean_cut(
        [(line.bbox[0], line.bbox[2]) for line in lines],
        minimum=median_h * 1.5,
        lo=x0,
        hi=x1,
    )
    if cut is None:
        return None
    right = [line.bbox[0] for line in lines if line.bbox[0] >= cut]
    left = [line.bbox[2] for line in lines if line.bbox[2] <= cut]
    if not right or not left:
        return None
    return (min(right) - max(left), cut)


def _to_lines(result: list, size: tuple[int, int], scale: float) -> list[OcrLine]:
    """把 RapidOCR 的原始结果转成 `OcrLine` 列表（未排序）。

    `size` 是整页像素的 (高, 宽)，用来把检测框夹回页内；`scale` 是 pt/像素。
    """
    height, width = size
    lines: list[OcrLine] = []
    for item in result or []:
        try:
            box, text = item[0], str(item[1] or "")
        except (IndexError, TypeError, ValueError):
            continue
        if not text.strip():
            continue

        px = (
            max(0, int(min(p[0] for p in box))),
            max(0, int(min(p[1] for p in box))),
            min(width, int(max(p[0] for p in box))),
            min(height, int(max(p[1] for p in box))),
        )
        if px[2] <= px[0] or px[3] <= px[1]:
            continue
        # 横排的一行文字，框一定宽远大于高（最窄的 `1` 也有 3 倍以上）。
        # 反过来「高≥宽」说明是竖排或旋转文字 —— 论文边上那个竖排的
        # arXiv 投稿戳就会被切成十几块碎渣（"1"、"2"、"un"、"CL]"），
        # 它们挤在左边距里，会把正文整体挤到后面去，读序全乱。
        if (px[2] - px[0]) < (px[3] - px[1]) * MIN_LINE_ASPECT:
            continue
        if len(text.strip()) <= 1:
            continue

        words = _words_from_text(text, px, scale)
        if not words:
            continue
        lines.append(
            OcrLine(
                text=" ".join(word.text for word in words),
                bbox=_union([word.bbox for word in words]),
                words=words,
                size=max((px[3] - px[1]) * scale, 1.0),
            )
        )
    return lines


def _render(page: "pymupdf.Page", dpi: int):
    """页面 → (BGR 像素数组, pt/像素 的换算系数)。

    必须显式要 RGB：灰度页取出来的通道数是 1，直接按 3 通道 reshape 会崩。
    RapidOCR 内部按 OpenCV 的约定吃 BGR，所以最后反一下通道。
    """
    import numpy as np

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
    flat = np.frombuffer(pix.samples, dtype=np.uint8)
    image = flat.reshape(pix.height, pix.width, 3)[:, :, ::-1]
    return np.ascontiguousarray(image), 72.0 / dpi


_LINE_CACHE: dict[tuple[str, int, int], list[OcrLine]] = {}
# 磁盘缓存版本号。改了识别参数（dpi、过滤规则、读序算法）就加一，
# 否则旧结果会被当成新算法的产物读回来 —— 看起来"改了没效果"。
CACHE_VERSION = 1


def _page_key(page: "pymupdf.Page", dpi: int) -> tuple[str, int, int]:
    return (getattr(page.parent, "name", "") or "", page.number, dpi)


def _cache_dir():
    """OCR 结果落到资料库旁边，和 PDF 走同一个卷，容器里也一样挂得到。

    为什么不放在 `scan.py` 边上或系统临时目录：临时目录会被清理，
    而这个缓存正是为了"下次别再算两分钟"才存在的。
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    store = Path(os.environ.get("PAPER_STORE_DIR") or root / "data" / "papers")
    return store.parent / "ocr_cache"


def _disk_key(page: "pymupdf.Page", dpi: int) -> str | None:
    """按「文件内容 + 页码 + 参数」生成缓存键。

    用 大小 + 修改时间 而不是哈希整个文件：几百 MB 的 PDF 读一遍就为了算哈希，
    比 OCR 这页还慢。文件名不同就是不同的键，覆盖上传同名文件也会因大小/时间变化而失效。
    """
    import hashlib
    from pathlib import Path

    try:
        path = Path(getattr(page.parent, "name", "") or "").resolve()
        stat = path.stat()
    except (OSError, ValueError):
        return None  # 从内存流打开的文档没有稳定路径，干脆不缓存
    raw = f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{page.number}|{dpi}|{CACHE_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_cached(key: str | None) -> list[OcrLine] | None:
    if key is None:
        return None
    import json

    try:
        payload = json.loads((_cache_dir() / f"{key}.json").read_text(encoding="utf-8"))
        return [
            OcrLine(
                text=item["t"],
                bbox=tuple(item["b"]),
                size=float(item["s"]),
                words=[OcrWord(w[0], tuple(w[1])) for w in item["w"]],
            )
            for item in payload
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return None  # 缓存坏了就当没有，重新识别即可


def _store_cached(key: str | None, lines: list[OcrLine]) -> None:
    """把一页的识别结果写下来。写不进去不影响这次请求。"""
    if key is None:
        return
    import json

    payload = [
        {
            "t": line.text,
            "b": list(line.bbox),
            "s": line.size,
            "w": [[word.text, list(word.bbox)] for word in line.words],
        }
        for line in lines
    ]
    try:
        folder = _cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{key}.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, folder / f"{key}.json")
    except OSError:
        pass


def ocr_lines(page: "pymupdf.Page", *, dpi: int = DEFAULT_DPI) -> list[OcrLine]:
    """识别一页，返回按阅读顺序排好的行。失败返回空列表，绝不抛异常。

    两级缓存，缺一不可：
    - 内存（进程内）：切块与词索引是两次独立调用，不缓存的话同一页要识别两遍；
    - 磁盘（跨进程）：资料库每次启动都会把所有 PDF 重新切一遍
      （`registry._load_existing`），没有磁盘缓存的话，库里有扫描件就意味着
      **每次重启都要重跑两三分钟 OCR**，演示时足以让人以为服务挂了。
    """
    engine = _engine()
    if engine is None:
        return []

    key = _page_key(page, dpi)
    cached = _LINE_CACHE.get(key)
    if cached is not None:
        return cached

    disk = _disk_key(page, dpi)
    cached = _load_cached(disk)
    if cached is not None:
        _LINE_CACHE[key] = cached
        return cached

    try:
        image, scale = _render(page, dpi)
    except Exception:  # noqa: BLE001
        return []

    try:
        # 不要 `return_word_box=True`：它给的逐字框是把行框均分出来的假坐标
        # （每字 13~14 px，完全一致），量不出词间空隙，还得多跑一遍解码。
        result, _ = engine(image)
    except Exception:  # noqa: BLE001 —— 扫描页千奇百怪，单页失败不该拖垮整份文档
        result = None
    lines = _reading_order(_to_lines(result, image.shape[:2], scale))

    if lines:  # 整页失败不写缓存，免得一次偶发失败被永久记住
        _store_cached(disk, lines)
    if len(_LINE_CACHE) > 512:  # 长文档不必无限占内存，扔了重算即可
        _LINE_CACHE.clear()
    _LINE_CACHE[key] = lines
    return lines


# ------------------------------------------------------------------ 对外接口


def warm(
    path: str,
    *,
    dpi: int = DEFAULT_DPI,
    workers: int = PARALLEL_PAGES,
    limit: int = MAX_OCR_PAGES,
) -> int:
    """先把这份 PDF 里缺文字层的页并发识别一遍，返回真正识别的页数。

    为什么要单独一步：切块器是**逐页同步**调用 OCR 钩子的，串行识别 15 页要两三分钟
    —— 上传接口就那么挂着。而这些页彼此毫无依赖，扔进线程池并发跑就行。
    识别结果写进缓存，随后切块器逐页取用时就是内存命中。

    实测（15 页扫描件，冷缓存，同一台机器）：串行 176s → 4 线程 117s，约 1.5 倍；
    再往上加线程没有收益（6 线程 170s、8 线程 109s，跑三遍每次还能差一倍）。
    **不要把它当成"线程数 × 加速比"去调**：一页之内 onnxruntime 已经把 16 个物理核
    用满了，多页并发只是在抢同一批核，所以只有这点余量。真正的大头是一页要识别
    几十行文字，见 docs/EXPERIMENTS.md E-018。

    每个线程**各自 open 一份文档**：PyMuPDF 的 `Document` 明确不能跨线程共用，
    共用一个对象去并发取页，轻则读到别的页，重则直接崩进程。
    """
    if not ocr_enabled():
        return 0

    try:
        doc = pymupdf.open(path)
    except Exception:
        return 0

    try:
        targets = [
            index
            for index in range(doc.page_count)
            if _visible_chars(doc[index].get_text("text")) < MIN_PAGE_TEXT_CHARS
        ][: max(0, limit)]
    except Exception:
        return 0
    finally:
        doc.close()

    if not targets:
        return 0

    def _one(index: int) -> None:
        try:
            own = pymupdf.open(path)
        except Exception:
            return
        try:
            ocr_lines(own[index], dpi=dpi)
        except Exception:  # noqa: BLE001 —— 单页坏了不该拖垮整份文档
            pass
        finally:
            own.close()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(targets)))) as pool:
        list(pool.map(_one, targets))
    return len(targets)


def _page_list(pages: list[int]) -> str:
    """把页码压成区间，如 `1、3-5、9`，避免长文档的说明过长。"""
    if not pages:
        return ""
    parts: list[str] = []
    start = prev = pages[0]
    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = page
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return "、".join(parts)


def _make_note(
    pages: int,
    readable_pages: int,
    scanned_pages: list[int],
    text_chars: int,
    ocr_available: bool,
) -> str:
    """生成一句中文说明，讲清「是什么情况」和「能不能 OCR」。"""
    if pages == 0:
        return "PDF 没有任何页面，无法判断文字层。"
    if not scanned_pages:
        return f"共 {pages} 页，全部有文字层（{text_chars} 个字符），无需 OCR。"
    if len(scanned_pages) == pages:
        head = f"共 {pages} 页，全部缺少文字层，判定为纯扫描件"
    else:
        head = (
            f"共 {pages} 页，其中 {len(scanned_pages)} 页缺少文字层"
            f"（第 {_page_list(scanned_pages)} 页），按扫描件处理"
        )
    if ocr_available:
        tail = "本机 OCR 引擎（RapidOCR）可用，解析时会自动识别。"
    else:
        tail = "本机没有可用的 OCR 引擎（RapidOCR 未安装或加载失败），无法识别这些页。"
    return f"{head}；{tail}"


def analyze(path: str) -> ScanReport:
    """逐页判断文字层，返回扫描件体检报告。

    这里刻意**不抛异常**：API 层拿到的应该是一份可解释的报告，
    而不是 500。打不开、加密、坏页都退化成「没有可用文字层」的描述。
    """
    ocr_ok = ocr_enabled()
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        # 文件不存在、不是 PDF、头部损坏都会走到这里。
        return ScanReport(
            pages=0,
            readable_pages=0,
            scanned_pages=[],
            text_chars=0,
            ocr_available=ocr_ok,
            note=f"无法打开文件，按非 PDF 或已损坏处理：{_short(exc)}",
        )

    try:
        pages = doc.page_count
        scanned: list[int] = []
        text_chars = 0
        for page_no in range(1, pages + 1):
            try:
                text = doc[page_no - 1].get_text("text")
            except Exception:
                # 单页抽不出文字（坏页、加密）比整份失败好：按扫描页记，
                # 后续 OCR 也失败时上层还能看到「这一页没有文字层」。
                text = ""
            chars = _visible_chars(text)
            text_chars += chars
            if chars < MIN_PAGE_TEXT_CHARS:
                scanned.append(page_no)
        readable_pages = pages - len(scanned)
    finally:
        doc.close()

    return ScanReport(
        pages=pages,
        readable_pages=readable_pages,
        scanned_pages=scanned,
        text_chars=text_chars,
        ocr_available=ocr_ok,
        note=_make_note(pages, readable_pages, scanned, text_chars, ocr_ok),
    )


def ocr_pages(
    path: str,
    pages: list[int] | None = None,
    *,
    dpi: int = DEFAULT_DPI,
    limit: int | None = MAX_OCR_PAGES,
) -> dict[int, str]:
    """对指定页做 OCR，返回 {页码: 识别文本}。

    `pages=None` 表示整份文档。OCR 引擎不可用、文件打不开、单页识别失败，
    都只让结果字典少几项，**绝不抛异常** —— 调用方可以据此明确告诉用户
    「能识别哪几页、哪几页没识别出来」，而不是假装整份成功。

    `limit` 是硬上限：不设的话几小时才跑得完的扫描书会让请求一直挂着。
    超出的页直接不出现结果字典里，调用方按"没识别出来"处理即可。
    """
    if not ocr_enabled():
        return {}

    try:
        doc = pymupdf.open(path)
    except Exception:
        return {}

    try:
        total = doc.page_count
        targets = list(range(1, total + 1)) if pages is None else list(pages)
        if limit is not None:
            targets = targets[: max(0, limit)]
        result: dict[int, str] = {}
        for page_no in targets:
            if not (1 <= page_no <= total):
                continue  # 越界页码直接跳过，不为一页废掉整批
            lines = ocr_lines(doc[page_no - 1], dpi=dpi)
            if lines:
                result[page_no] = "\n".join(line.text for line in lines)
        return result
    finally:
        doc.close()


def _width_p90(widths) -> float:
    """本页行宽的 90 分位，当作"正文满行有多宽"的参考值。

    为什么不用中位数：带插图的页面上，图注、坐标轴标签、示意图里的零碎
    文字又多又短 —— 实测第 3 页的中位行宽只有 50pt，而正文满行是 397pt。
    中位数一被拉低，"不到满行一半才算短行"的阈值就成了 20pt，整页没有一行
    够格独立成块，`3.2 Attention` 这类标题全被并进正文里。
    """
    values = sorted(widths)
    if not values:
        return 1.0
    return values[max(0, min(len(values) - 1, int(len(values) * 0.9) - 1))] or 1.0


def _standalone_line(line: "OcrLine", body_width: float) -> bool:
    """这一行像不像独立成行的标题（`References`、`5.2 Hardware and Schedule`）。

    为什么要单独判一次：切块器认标题有三条路 —— 已知关键词、章节编号、字号。
    扫描件上前两条能用，字号那条基本废掉（一行的"字号"是像素墨迹高度），
    `bold` 更是恒 False —— 像素里没有字体名。所以 `References` 这种
    "正文字号 + 加粗"的标题，能被认出来的唯一途径就是先独立成块、
    再命中关键词表。

    判据两条都要：宽度不到本页满行宽的 50%，且不超过 5 个词。
    `body_width` 由 `_width_p90` 给，别拿中位数代替（原因见那里）。
    真实事故：不拆的话 `References` 会被并进上一段正文的尾巴
    （`…available soon. References`），整份 15 页扫描件抽不出 1 条引文，
    而它的文字版能抽出 36 条。
    """
    width = line.bbox[2] - line.bbox[0]
    return (
        width < body_width * HEADING_LINE_WIDTH_RATIO
        and len(line.words) <= HEADING_LINE_MAX_WORDS
    )


def ocr_blocks_for_page(page: "pymupdf.Page", *, dpi: int = DEFAULT_DPI) -> dict | None:
    """把一页交给 OCR，拼成 `get_text("dict")` 那种结构（带 bbox 与字号）。

    为什么不直接返回字符串：切块器要靠 `size` / `bold` 认标题，靠 `bbox` 画高亮。
    返回纯文本等于把扫描件降级成"一整页一大块"，目录树和溯源框全没了。

    和原生 `get_text("dict")` 的差异，都是扫描件里不可能恢复的东西：
    - `size` 用一行的像素高度当代理。检测框是「上到最高字顶、下到最低字底」，
      高度取决于这一行里有没有括号、升部、降部，**不是字号**：实测同一份论文
      正文 13~19pt、真标题 13~20pt，两个区间几乎完全重叠。所以切块器
      明确不拿它判标题（`Para.size_proxy`），标题靠关键词与编号识别。
    - `bold` 一律 False。像素里没有字体名，猜就是编。
    - 段落靠行距切分，不靠 PDF 的 block 结构（扫描件本来就没有）。

    返回 None 表示"这一页 OCR 不了"（引擎缺失 / 单页失败），
    调用方应当退回"没有正文"的语义，而不是塞进空字符串。
    """
    if not ocr_enabled():
        return None
    lines = ocr_lines(page, dpi=dpi)
    if not lines:
        return None

    blocks: list[dict] = []
    group: list[OcrLine] = []
    body = median(line.size for line in lines) or 1.0

    def flush() -> None:
        if not group:
            return
        blocks.append(
            {
                "type": 0,
                "bbox": list(_union([line.bbox for line in group])),
                "lines": [
                    {
                        "bbox": list(line.bbox),
                        "wmode": 0,
                        "dir": (1.0, 0.0),
                        "spans": [
                            {
                                "size": line.size,
                                "flags": 0,
                                # 词后面补空格：切块器是 "".join(span 文本) 拼行的，
                                # 不补的话整行又会粘成一串（丢空格那个坑的余波）
                                "font": "ocr",
                                "bbox": list(word.bbox),
                                "text": word.text + " ",
                            }
                            for word in line.words
                        ],
                    }
                    for line in group
                ],
            }
        )
        group.clear()

    ref_width = _width_p90(line.bbox[2] - line.bbox[0] for line in lines)
    for line in lines:
        if group:
            prev = group[-1]
            gap = line.bbox[1] - prev.bbox[3]
            size_jump = abs(line.size - prev.size) > max(prev.size, 1.0) * 0.15
            if (
                gap > prev.size * PARA_GAP_FACTOR
                or line.bbox[1] < prev.bbox[3]
                or size_jump
                # 标题式短行两头都要断开：只断前面的话，`References` 会和
                # 紧随其后的第一条文献并成一块，照样不是纯标题。
                or _standalone_line(line, ref_width)
                or _standalone_line(prev, ref_width)
            ):
                flush()
        group.append(line)
    flush()

    return {"width": page.rect.width, "height": page.rect.height, "blocks": blocks}


def ocr_words(page: "pymupdf.Page", *, dpi: int = DEFAULT_DPI) -> list[tuple]:
    """一页的词流，格式与 `page.get_text("words")` 对齐：`(x0,y0,x1,y1,词,块,行,序号)`。

    词索引（`TextIndex`）靠这个给扫描件建索引。没有它，扫描件能问答、
    却一条引用都落不到原文 —— 抗幻觉在扫描件上等于全线失效。
    """
    if not ocr_enabled():
        return []
    out: list[tuple] = []
    for line_no, line in enumerate(ocr_lines(page, dpi=dpi)):
        for word_no, word in enumerate(line.words):
            x0, y0, x1, y1 = word.bbox
            out.append((x0, y0, x1, y1, word.text, 0, line_no, word_no))
    return out
