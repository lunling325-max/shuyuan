"""把 PDF 里的表格和插图还原成结构化数据。

为什么非要有这一层：切块器只关心"读顺序的一段文字"，表格在它眼里就是
一堆被拆散的行 —— `BLEU` 在表头、`41.8` 在第五行，两者之间没有任何关系。
检索到"Transformer 的 BLEU 是多少"时，模型拿到的是这些碎裂的行，
说对了是运气，说错了也没人看得出来。表格得按**行×列**重新组装一次。

三件事只做到"够用"，不做"完美"：

1. **表头与表体**：`page.find_tables()` 给的是网格与坐标，单元格文字要自己从
   `get_text("words")` 里按坐标捞。直接用它自带的 `extract()` 会把 `24.6`
   拆成 `246` + `.`（它按文本行切，而小数点常常是单独一个 text run）。
2. **图注**：表格标题在上方、图片标题在下方（多数期刊排版如此），两头都找，
   取最近的一条，并且把它后面紧挨着的续行一起收进来 —— 图注一般是两三行。
3. **不做 OCR 表格**：扫描页没有文字层，`find_tables()` 一律返回空。与其猜，
   不如如实说"这份扫描件提不出表格"（`note` 里会写明）。

开销：`find_tables()` 实测约 0.29 秒/页，75 页的 GPT-3 要 22 秒。所以这里
只解析前 `MAX_ANALYZE_PAGES` 页，并且结果落盘 —— 同一份 PDF 只算一次。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.parsing.chunker import _LEADER_RUN

# 逐页调用 `find_tables()` 约 0.29 秒。40 页 ≈ 12 秒，是"点一下等得住"的上限；
# 541 页的教材要两分半，不能让请求挂着。
MAX_ANALYZE_PAGES = 40
# 图注/表注与对象之间的最大距离（pt）。再远就不是这张表的标题了。
CAPTION_WINDOW = 64.0
# 插图面积下限：小于整页 2% 的都是期刊 logo、图标、扫描噪点。
MIN_FIGURE_AREA_RATIO = 0.02
# 整页大小的"图"是扫描件自己那整页位图，不是插图。
MAX_FIGURE_AREA_RATIO = 0.80
# 单元格里同一行的判定容差（以行高为单位）。表格里数字框常常不在同一基线上。
ROW_TOLERANCE = 0.55
# 两个字之间小于这个间距就不插空格（`24.6`、`1.0×`这一类）。
# 阈值必须小于真正的词间空格：9pt 字体的空格实测约 2pt，而"小数点自带一个
# text run"那种碎片间距接近 0。取 0.12 倍行高（≈1.1pt）两头都分得开。
GLUE_RATIO = 0.12

# 编号必须连子号一起取：GPT-3 里的表题是 `Table 2.1`、`Table 2.2`、`Table 3.1`，
# 只取 `\d+` 会让三张不同的表全叫 "Table 2"，界面上看着像同一张表重复了三次。
CAPTION_RE = re.compile(r"^(?:table|figure|fig\.?|图|表)\s*(\d+(?:\.\d+)*[a-z]?)", re.IGNORECASE)
_TABLE_WORDS = ("table", "表")
_FIGURE_WORDS = ("figure", "fig", "图")

CACHE_VERSION = 4


@dataclass(frozen=True, slots=True)
class DocTable:
    """一张表格。`rows` 已经按行列重排过，空行已剔除。"""

    page: int
    bbox: tuple[float, float, float, float]
    label: str
    caption: str
    rows: tuple[tuple[str, ...], ...]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    def markdown(self) -> str:
        """Markdown 表格：给模型看、给前端复制走都一样。"""
        if not self.rows:
            return ""
        width = self.n_cols
        lines = []
        for i, row in enumerate(self.rows):
            cells = list(row) + [""] * (width - len(row))
            # Markdown 表格里不能有裸换行（会把行切断），用 <br> 保住格内分行
            flat = [c.replace("|", "\\|").replace("\n", "<br>") for c in cells]
            lines.append("| " + " | ".join(flat) + " |")
            if i == 0:
                lines.append("|" + "---|" * width)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class DocFigure:
    page: int
    bbox: tuple[float, float, float, float]
    label: str
    caption: str
    width: int  # 像素尺寸（原始嵌入图像），不是磅值
    height: int


@dataclass(frozen=True, slots=True)
class VisualReport:
    tables: tuple[DocTable, ...]
    figures: tuple[DocFigure, ...]
    pages: int
    analyzed_pages: int
    scanned_pages: tuple[int, ...]
    note: str
    cached: bool = False


# ------------------------------------------------------------------ 单元格文字

def _words_in(words: list, rect) -> list:
    """按**中心点**判断词属于哪个格子。

    不能用"整词都在格子里"：紧贴边框的词会被整行丢掉，数字列正好贴着框线。
    """
    x0, y0, x1, y1 = rect
    out = []
    for w in words:
        cx = (w[0] + w[2]) / 2
        cy = (w[1] + w[3]) / 2
        if x0 - 1.5 <= cx <= x1 + 1.5 and y0 - 1.5 <= cy <= y1 + 1.5:
            out.append(w)
    return out


def _glue(row: list, line_height: float) -> str:
    """一行里的词按 x 排好后拼起来：贴得近就不加空格（`24.6`、`1.0×`）。"""
    if not row:
        return ""
    row = sorted(row, key=lambda w: w[0])
    buf = row[0][4]
    for prev, cur in zip(row, row[1:]):
        gap = cur[0] - prev[2]
        buf += ("" if gap < line_height * GLUE_RATIO else " ") + cur[4]
    return " ".join(buf.split())


def _bands(inside: list, line_height: float) -> list[list]:
    """把一堆词按 y 聚成视觉行，保持自上而下的顺序。"""
    bands: list[list] = []
    for w in sorted(inside, key=lambda w: (w[1] + w[3]) / 2):
        centre = (w[1] + w[3]) / 2
        if bands and abs(centre - bands[-1][0]) <= line_height * ROW_TOLERANCE:
            bands[-1][1].append(w)
        else:
            bands.append([centre, [w]])
    return [band[1] for band in bands]


def _median_height(words: list) -> float:
    heights = sorted(w[3] - w[1] for w in words)
    return heights[len(heights) // 2] or 1.0


def _cell_text(words: list, rect) -> str:
    """一个格子里的词拼成文本；格内有多行时用 `\n` 分开，**不拼成一串**。"""
    inside = _words_in(words, rect)
    if not inside:
        return ""
    line_height = _median_height(inside)
    return "\n".join(_glue(band, line_height) for band in _bands(inside, line_height))


def _visual_rows(words: list, cells: list) -> list[tuple[str, ...]]:
    """把一个表格行按**视觉行**拆成若干真行。

    booktabs 排法（只有三条横线，没有逐行横线）里 `find_tables()` 只切出
    "表头 + 一整块表体"：GPT-3 的 Table 2.1 是 2 行 × 8 列，10 个模型的数字
    全挤在一格。只靠格内换行还不够 —— Transformer 的 Table 2 第一个模型没有
    EN-DE 值（原文是 `-`），少了那一行，后面所有数字都会错位。
    所以按行内 y 聚线、逐列分配：缺值的格子**留空**，行列才对得上。

    `cells` 里有 None（合并单元格）时返回空列表，让调用方回落到逐格拼接。
    """
    if not cells or any(c is None for c in cells):
        return []
    x0 = min(c[0] for c in cells)
    y0 = min(c[1] for c in cells)
    x1 = max(c[2] for c in cells)
    y1 = max(c[3] for c in cells)
    inside = _words_in(words, (x0, y0, x1, y1))
    if not inside:
        return []
    line_height = _median_height(inside)

    out: list[tuple[str, ...]] = []
    for band in _bands(inside, line_height):
        # 只按 x 分配：y 已由这一条视觉线固定住了
        out.append(
            tuple(
                _glue([w for w in band if cx0 - 1.5 <= (w[0] + w[2]) / 2 <= cx1 + 1.5], line_height)
                for cx0, _, cx1, _ in cells
            )
        )
    return [row for row in out if any(row)]


# ------------------------------------------------------------------ 标题匹配

def _caption_for(blocks: list, bbox, kind: str) -> tuple[str, str]:
    """找最近的 `Table N` / `Figure N` 标题，返回 (标签, 全文)。

    表格标题在上面、插图标题在下面，所以两头都找，取距离最近的；
    找到之后把它后面紧挨着的续行一起接上（图注通常两三行）。
    """
    x0, y0, x1, y1 = bbox
    best = None
    for i, b in enumerate(blocks):
        text = " ".join(str(b[4]).split())
        m = CAPTION_RE.match(text)
        if not m or not text.lower().startswith(_TABLE_WORDS if kind == "table" else _FIGURE_WORDS):
            continue
        above = b[3] <= y0 + 2
        below = b[1] >= y1 - 2
        if above:
            distance, step = y0 - b[3], -1
        elif below:
            distance, step = b[1] - y1, 1
        else:
            continue
        if distance > CAPTION_WINDOW:
            continue
        if best is None or distance < best[0]:
            best = (distance, i, step)
    if best is None:
        return "", ""

    _, index, step = best
    block = blocks[index]
    label_word = "Table" if kind == "table" else "Figure"
    label = f"{label_word} {CAPTION_RE.match(' '.join(str(block[4]).split())).group(1)}"
    parts = [" ".join(str(block[4]).split())]
    # 续行：同方向的紧邻块，且自己不是另一条标题
    cursor = block
    for _ in range(2):
        nxt = _neighbour(blocks, cursor, step)
        if nxt is None:
            continue
        text = " ".join(str(nxt[4]).split())
        if CAPTION_RE.match(text):
            break
        parts.append(text)
        cursor = nxt
    return label, " ".join(parts)


def _neighbour(blocks: list, block, step: int):
    """`step = -1` 取上方紧邻的块，`1` 取下方。间距超过 1.5 行高就不算续行。"""
    height = block[3] - block[1] or 10.0
    candidates = []
    for b in blocks:
        if b is block:
            continue
        if step < 0:
            gap = block[1] - b[3]
        else:
            gap = b[1] - block[3]
        if 0 <= gap <= height * 1.5:
            candidates.append((gap, b))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


# ------------------------------------------------------------------ 提取

def _toc_pages(doc) -> set[int]:
    """整页都是目录的页码。

    GPT-3 的目录页会被 `find_tables()` 认成一张 32 行 × 3 列的"表"
    （章节名 + 页码的对齐方式确实像表格）。目录不是数据，别混进图表清单。
    """
    pages = set()
    for page in doc:
        if len(_LEADER_RUN.findall(page.get_text())) >= 3:
            pages.add(page.number + 1)
    return pages


def _tables_on_page(page, blocks: list) -> list[DocTable]:
    words = page.get_text("words")
    out: list[DocTable] = []
    for table in page.find_tables().tables:
        if table.row_count < 2 or table.col_count < 2:
            continue  # 单行/单列的多半是把图注、公式框误认成表格
        rows = []
        for row in table.rows:
            cells = list(row.cells)
            visual = _visual_rows(words, cells)
            if visual:
                rows.extend(visual)
                continue
            # 有合并单元格的格子拼不出视觉行，退回逐格拼接（格内换行照样保留）
            joined = tuple(_cell_text(words, rect) for rect in cells if rect is not None)
            if any(joined):
                rows.append(joined)
        if len(rows) < 2:
            continue
        label, caption = _caption_for(blocks, table.bbox, "table")
        # 首页（标题页）上抓不到表题的网格一律不要：论文首页的"作者 + 单位"
        # 本身就是两列对齐的版面，`find_tables()` 会把它认成一张 7×2 的表，
        # 单元格里全是人名。首页上真正的数据表几乎都带表题，宁缺勿滥。
        if page.number == 0 and not label:
            continue
        out.append(
            DocTable(
                page=page.number + 1,
                bbox=tuple(round(v, 1) for v in table.bbox),
                label=label,
                caption=caption,
                rows=tuple(rows),
            )
        )
    return out


def _figures_on_page(page) -> list[DocFigure]:
    area_limit = page.rect.width * page.rect.height
    blocks = page.get_text("blocks")
    out: list[DocFigure] = []
    seen: set[tuple[int, int, int, int]] = set()
    for image in page.get_images(full=True):
        xref, width, height = image[0], image[2], image[3]
        for rect in page.get_image_rects(xref):
            area = rect.width * rect.height
            # 面积过小＝图标；过大＝扫描件整页位图。两头都不要。
            if area < area_limit * MIN_FIGURE_AREA_RATIO or area > area_limit * MAX_FIGURE_AREA_RATIO:
                continue
            key = (round(rect.x0), round(rect.y0), round(rect.x1), round(rect.y1))
            if key in seen:
                continue
            seen.add(key)
            label, caption = _caption_for(blocks, tuple(rect), "figure")
            out.append(
                DocFigure(
                    page=page.number + 1,
                    bbox=(round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1)),
                    label=label,
                    caption=caption,
                    width=int(width),
                    height=int(height),
                )
            )
    out.sort(key=lambda f: (f.bbox[1], f.bbox[0]))
    return out


def analyze(path: str | Path, *, max_pages: int = MAX_ANALYZE_PAGES) -> VisualReport:
    """解析整份 PDF 的表格与插图（带磁盘缓存）。

    打不开的文件返回空报告而不是抛异常：图表解析失败不该让整个页面 500，
    "这份 PDF 提不出图表"本身就是一条可以显示给用户的结果。
    """
    key = _disk_key(path, max_pages)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    try:
        doc = pymupdf.open(str(path))
    except Exception:  # noqa: BLE001 —— 损坏的 PDF 也是一种"没有图表"
        return VisualReport((), (), 0, 0, (), "这份 PDF 打不开，无法解析表格与插图。")

    try:
        total = doc.page_count
        limit = min(total, max(0, max_pages))
        toc = _toc_pages(doc)
        tables: list[DocTable] = []
        figures: list[DocFigure] = []
        scanned: list[int] = []
        for page_no in range(1, limit + 1):
            page = doc[page_no - 1]
            blocks = page.get_text("blocks")
            if sum(len(str(b[4]).strip()) for b in blocks) < 10:
                scanned.append(page_no)
                continue  # 没有文字层：find_tables 在这里什么也认不出来
            if page_no in toc:
                continue
            tables.extend(_tables_on_page(page, blocks))
            figures.extend(_figures_on_page(page))
        report = VisualReport(
            tables=tuple(tables),
            figures=tuple(figures),
            pages=total,
            analyzed_pages=limit,
            scanned_pages=tuple(scanned),
            note=_make_note(total, limit, tables, figures, scanned),
        )
    finally:
        doc.close()

    _store_cached(key, report)
    return report


def render_figure(path: str | Path, page_no: int, bbox, *, dpi: int = 110) -> bytes | None:
    """把一块区域渲染成 PNG（给前端做缩略图）。

    为什么不直接抽嵌入的原图：原图可能是 CMYK、带透明遮罩、或者 4000px 宽，
    浏览器要么显示不出来，要么白白下载几 MB。按版面尺寸重渲染一次最省事。
    页码由调用方给（`DocFigure.page`），不去逐页反查 —— 541 页的教材
    反查一遍就是几十秒。
    """
    try:
        doc = pymupdf.open(str(path))
    except Exception:  # noqa: BLE001
        return None
    try:
        if not (1 <= page_no <= doc.page_count):
            return None
        page = doc[page_no - 1]
        rect = pymupdf.Rect(*(float(v) for v in bbox))
        if rect.is_empty or rect.is_infinite:
            return None
        return page.get_pixmap(clip=rect, dpi=dpi).tobytes("png")
    finally:
        doc.close()


def _make_note(total: int, limit: int, tables: list, figures: list, scanned: list[int]) -> str:
    if not tables and not figures:
        if scanned:
            return (
                f"前 {limit} 页里有 {len(scanned)} 页是扫描页（没有文字层），"
                "表格与插图需要文字层才能定位；这份 PDF 提不出图表。"
            )
        return "没有找到表格或插图。"
    parts = [f"表格 {len(tables)} 张、插图 {len(figures)} 幅"]
    if limit < total:
        parts.append(f"只解析了前 {limit} 页（共 {total} 页），后面的图表没有提取")
    if scanned:
        parts.append(f"跳过 {len(scanned)} 页扫描页（没有文字层）")
    return "，".join(parts) + "。"


# ------------------------------------------------------------------ 磁盘缓存

def _cache_dir() -> Path:
    """和 OCR 缓存并列：同一个卷、同一个清理口径。"""
    root = Path(__file__).resolve().parents[4]
    store = Path(os.environ.get("PAPER_STORE_DIR") or root / "data" / "papers")
    return store.parent / "visual_cache"


def _disk_key(path: str | Path, max_pages: int) -> str | None:
    import hashlib

    try:
        stat = Path(path).resolve().stat()
    except OSError:
        return None
    raw = f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{max_pages}|{CACHE_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_cached(key: str | None) -> VisualReport | None:
    if key is None:
        return None
    import json

    try:
        payload = json.loads((_cache_dir() / f"{key}.json").read_text(encoding="utf-8"))
        return VisualReport(
            tables=tuple(
                DocTable(
                    page=item["page"],
                    bbox=tuple(item["bbox"]),
                    label=item["label"],
                    caption=item["caption"],
                    rows=tuple(tuple(row) for row in item["rows"]),
                )
                for item in payload["tables"]
            ),
            figures=tuple(
                DocFigure(
                    page=item["page"],
                    bbox=tuple(item["bbox"]),
                    label=item["label"],
                    caption=item["caption"],
                    width=item["width"],
                    height=item["height"],
                )
                for item in payload["figures"]
            ),
            pages=payload["pages"],
            analyzed_pages=payload["analyzed_pages"],
            scanned_pages=tuple(payload["scanned_pages"]),
            note=payload["note"],
            cached=True,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None  # 缓存坏了就当没有，重新解析


def _store_cached(key: str | None, report: VisualReport) -> None:
    if key is None:
        return
    import json

    payload = {
        "tables": [
            {
                "page": t.page,
                "bbox": list(t.bbox),
                "label": t.label,
                "caption": t.caption,
                "rows": [list(row) for row in t.rows],
            }
            for t in report.tables
        ],
        "figures": [
            {
                "page": f.page,
                "bbox": list(f.bbox),
                "label": f.label,
                "caption": f.caption,
                "width": f.width,
                "height": f.height,
            }
            for f in report.figures
        ],
        "pages": report.pages,
        "analyzed_pages": report.analyzed_pages,
        "scanned_pages": list(report.scanned_pages),
        "note": report.note,
    }
    try:
        folder = _cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{key}.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, folder / f"{key}.json")
    except OSError:
        pass  # 写不进去不影响这次请求
