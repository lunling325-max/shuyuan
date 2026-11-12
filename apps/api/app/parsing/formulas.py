"""公式块：把"有编号的独立公式"从版面里认出来，给出页码、坐标、编号与裁图。

为什么叫「公式块」而不是「LaTeX 抽取」

PDF 里公式是**排好版的字形**，不是 LaTeX 源码。实测（`1706.03762v7` 的式 1）：

| 原文 | 文字层里读出来 |
|---|---|
| softmax(QKᵀ / √d_k)V | `softmax(QKT` + `√` + `dk` + `)V`（上标 T 被摊平成普通字重，根号没有 ToUnicode 映射） |
| y = ϕ₀ + ϕ₁x | `f[x, ϕ] y = = ϕ0 + ϕ1x.`（分式横线不是字，分子分母的读序被拼在一起） |

也就是说：**文本层能告诉你"这里有一个公式、是第几个"，但还原不出公式本身。**
要真的 LaTeX，得靠数学 OCR（pix2tex 一类）或 arXiv 的源码包，两样都没做，
也不假装做了 —— 拼一个看起来像 LaTeX 的字符串出来，比不给更坏。

所以这一层给四样东西：在**第几页**、**编号**（式 (1)）、**坐标**（前端按它裁图，
图是准的）、以及一段**明说有损**的版面文本（给检索用，不给数学用）。

判据只有一个：**右边缘那个孤零零的 `(1)`**。为什么只认编号：

- 编号是作者给的，不是我们猜的 —— 认到 `(2.1)` 就是作者写的第 2.1 式；
- 无编号的居中公式（GPT-3、BERT 全是这种）要靠"数学字体占比 + 短行 + 上下留白"
  去猜，实测会认到 `6.0 × 10−4` 这种表里的数字和 `nparams` 这种表头。
  宁可一个不认，也不要把表格单元格当成公式（E-041）。

代价写在 `note` 里：这几篇论文本来就一条编号公式都没有时，界面要说清是**没有**，
而不是"解析失败"。
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# 和图表同一套上限：40 页是"点一下等得住"的边界（公式检测比 find_tables 便宜得多，
# 但两栏大书的后面几百页也没人在演示里翻）。
MAX_ANALYZE_PAGES = 40
# 缓存格式版本：字段一改就 +1，否则老缓存会被当成新格式读进来
CACHE_VERSION = 1

# 公式编号：整行只有 `(1)` / `(2.1)` / `(12)`。不认 `[1]`（那是参考文献）、
# 不认 `1.`（那是章节号）、不认 `(A.1)`（附录里常见，但目前没有对应的版面判据，
# 认了还得再编一套，先不认）。
NUMBER = re.compile(r"^\(\s*(\d{1,3}(?:\.\d{1,3})*)\s*\)$")

# 数学字体：Computer Modern 家族（CMMI 数学斜体、CMSY 符号、CMEX 大运算符、
# CMR 罗马体数字）、MSBM（AMS 黑板粗）、STIX / Latin Modern / XITS（现代模板）。
# 命中任意一种说明这一段落在数学排版里。
MATH_FONT = re.compile(
    r"CMMI|CMSY|CMEX|CMR\d|MSBM|Symbol|Euclid|STIX|Latin ?Modern|XITS|TeX", re.IGNORECASE
)

# 图注/表注开头的行。编号窗口里出现它，说明这一行不是公式体
# （表头里也会有一个孤零零的 `(1)` 表示脚注标记）。
CAPTION_HEAD = re.compile(r"^(?:table|figure|fig\.?|图|表)\s*\d", re.IGNORECASE)

# 公式体里数学字体字符的占比下限。实测有编号公式都在 0.56 以上，
# 而正文段落的数学字体占比不到 0.2（`d_model = 512` 这种行内式撑不起一段）。
MIN_MATH_RATIO = 0.30
# 编号到栏目右边缘的容差（pt）。排到右边缘 14pt 以外的，不是公式编号。
EDGE_TOLERANCE = 14.0
# 编号本身的宽度上限（pt）：`(12)` 在 10pt 字号下约 11pt 宽，取 48 已经很宽松。
MAX_NUMBER_WIDTH = 48.0
# 上下取行的窗口（以编号行高为单位）。公式比正文高，固定值会漏掉根号下面的分母。
BAND_SLACK = 1.6
# 「同一栏」的判定：右边缘对到 6pt 以内的正文行。再宽松一点就会把上一栏的行算进来。
COLUMN_EDGE_TOLERANCE = 6.0
# 把公式体上下接到紧贴着的数学行上（求和上下限、分式的分子分母）的三个门槛。
# 间隙按磅算：正文行距约 11pt，行高约 10pt —— 8pt 只够接"排版上属于同一条公式"的那些行。
GROW_MATH_RATIO = 0.5
GROW_OVERLAP = 0.5
GROW_GAP = 8.0
# 并排接的横向间隙（pt）：公式里段与段之间的空格约 3–6pt，取 12 已经很宽松。
GROW_SIDE_GAP = 12.0


@dataclass(frozen=True, slots=True)
class DocFormula:
    """一个有编号的独立公式。`bbox` 是公式体的包围盒（不含编号）。"""

    page: int
    bbox: tuple[float, float, float, float]
    number: str          # "1" / "2.1"
    label: str           # "式 (1)"
    text: str            # 明说有损的版面文本
    math_ratio: float    # 数学字体字符占比，用来解释"为什么认成公式"

    @property
    def key(self) -> str:
        return f"{self.page}:{self.number}"


@dataclass(frozen=True, slots=True)
class FormulaReport:
    formulas: tuple[DocFormula, ...]
    pages: int
    analyzed_pages: int
    scanned_pages: tuple[int, ...]
    note: str
    cached: bool = False


@dataclass(frozen=True, slots=True)
class _Line:
    """检测用的一行。比 `layout.Line` 多两样：数学字体字符数、bbox 四角。

    不复用 `layout.page_lines` 的原因：它按页开一次文档，40 页就是 40 次 open；
    这里要整篇走一遍，而且在同一页里还要反复按坐标筛行。
    """

    bbox: tuple[float, float, float, float]
    text: str
    math_chars: int
    chars: int

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def middle(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


def _lines_of(page) -> list[_Line]:
    out: list[_Line] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:  # 1 是图片块
            continue
        for raw in block.get("lines", []):
            spans = raw.get("spans", [])
            text = "".join(span.get("text") or "" for span in spans)
            if not text.strip():
                continue
            # 空白不计入：分母是去掉空格后的字符数，分子也得是，否则会算出 1.15 这种占比
            math_chars = sum(
                len("".join((span.get("text") or "").split()))
                for span in spans
                if MATH_FONT.search(span.get("font") or "")
            )
            box = raw["bbox"]
            out.append(
                _Line(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    text=" ".join(text.split()),
                    math_chars=math_chars,
                    chars=len("".join(text.split())),
                )
            )
    return out


def _edges(lines: list[_Line], *, side: int) -> list[float]:
    """栏目右边缘的候选（`side=2`）。

    不能只取"全页最右那一行"：两栏排版的论文，左栏的公式编号对着左栏的右边缘，
    用右栏的边缘去比就永远差着一栏宽。取**出现三次以上的边缘**（正文行大多对齐），
    编号挨着其中任意一个就算数。一页里没有稳定的边缘时退回"最外侧那一行"，
    至少不会一个都认不出。
    """
    counted = Counter(round(line.bbox[side]) for line in lines if len(line.text) >= 40)
    edges = [x for x, count in counted.items() if count >= 3]
    if edges:
        return edges
    if not counted:
        return []
    return [max(counted) if side == 2 else min(counted)]


def _column_left(number_line: _Line, lines: list[_Line]) -> float | None:
    """编号所在那一栏的左边缘：**右边缘和编号对齐的那些正文行**，取最左的起点。

    为什么不用"全页出现三次以上的左边缘"：图注块常常自成一段窄栏
    （UDL 第 32 页的图注从 303.5 起），它会把栏的左边缘算到 303，
    于是下一段真正的公式（从 216 起）被判成"别的栏"、整条丢掉。
    右边缘对齐的正文行才说明"同一栏的正文是从哪儿开始的"。
    """
    column = [
        line
        for line in lines
        if line is not number_line
        and len(line.text) >= 40
        and abs(line.bbox[2] - number_line.bbox[2]) <= COLUMN_EDGE_TOLERANCE
    ]
    return min((line.bbox[0] for line in column), default=None)


def _overlap_ratio(line: _Line, box: tuple[float, float, float, float]) -> float:
    """这一行在水平方向上被 `box` 盖住了几成。"""
    width = line.bbox[2] - line.bbox[0]
    if width <= 0:
        return 0.0
    covered = min(line.bbox[2], box[2]) - max(line.bbox[0], box[0])
    return max(0.0, covered) / width


def _grow(
    band: list[_Line], lines: list[_Line], number_line: _Line
) -> list[_Line]:
    """把公式体往上/往下接到**紧贴着**的数学行上。

    求和符号的上下限、分式的分子分母、根式的被开方数，常常被排成独立的行，
    而且离主行只有一两磅 —— 只看"离编号行多近"会正好漏掉它们
    （UDL 的式 2.5 一开始把求和上限 `I` 和左边的 `L[ϕ] =` 都漏在框外）。

    三个条件同时成立才接：**数学字体占比过半**，且要么
    ①**水平方向被当前框盖住大半**、垂直间隙在 `GROW_GAP` 以内，
    要么 ②**水平方向和当前框只隔一个空格**、垂直方向已经重叠
    （等号左边那截 `L[ϕ]` 就属于 ② —— 它横在公式框左边，一辈子也盖不住）。
    正文段落夹在公式上下时，第一条就把它挡住了 —— 正文的数学字体占比不到两成。
    """
    current = list(band)
    box = (
        min(line.bbox[0] for line in current),
        min(line.bbox[1] for line in current),
        max(line.bbox[2] for line in current),
        max(line.bbox[3] for line in current),
    )
    while True:
        grown = False
        for line in lines:
            if line is number_line or line in current:
                continue
            if not line.chars or line.math_chars / line.chars < GROW_MATH_RATIO:
                continue
            # 与当前框的垂直间隙（上下都算）
            gap = max(box[1] - line.bbox[3], line.bbox[1] - box[3], 0.0)
            side_gap = max(box[0] - line.bbox[2], line.bbox[0] - box[2], 0.0)
            stacked = _overlap_ratio(line, box) >= GROW_OVERLAP and gap <= GROW_GAP
            side_by_side = side_gap <= GROW_SIDE_GAP and gap <= 0.0
            if not (stacked or side_by_side):
                continue
            current.append(line)
            box = (
                min(box[0], line.bbox[0]),
                min(box[1], line.bbox[1]),
                max(box[2], line.bbox[2]),
                max(box[3], line.bbox[3]),
            )
            grown = True
        if not grown:
            return current


def _formula_at(
    number_line: _Line,
    lines: list[_Line],
    right_edges: list[float],
) -> DocFormula | None:
    """以一行 `(N)` 为锚点，往上下一小段里把公式体收进来。"""
    if not any(abs(number_line.bbox[2] - edge) <= EDGE_TOLERANCE for edge in right_edges):
        return None
    if number_line.bbox[2] - number_line.bbox[0] > MAX_NUMBER_WIDTH:
        return None

    # 只收**同一栏**里的行。两栏页面里，编号左边紧挨着的常常是另一栏的正文
    # （UDL 的式 2.3 一开始把左栏那一行 "Argmin function" 也吞了进来）。
    column_left = _column_left(number_line, lines)

    # 公式体在编号**左边**；上下窗口按编号行高伸缩 —— 带根号/分式的公式比正文高，
    # 固定的 14pt 会把根号下面那层分母漏在框外。
    slack = max(14.0, number_line.height * BAND_SLACK)
    band = [
        line
        for line in lines
        if line is not number_line
        and line.bbox[2] <= number_line.bbox[0] - 2
        and (column_left is None or line.bbox[0] >= column_left - 4)
        and abs(line.middle - number_line.middle) <= slack
    ]
    if not band:
        return None

    total = sum(line.chars for line in band)
    math = sum(line.math_chars for line in band)
    if not total or math / total < MIN_MATH_RATIO:
        return None
    if any(CAPTION_HEAD.match(line.text) for line in band):
        return None

    band = _grow(band, lines, number_line)
    total = sum(line.chars for line in band)
    math = sum(line.math_chars for line in band)

    band.sort(key=lambda line: (round(line.bbox[1], 1), line.bbox[0]))
    bbox = (
        min(line.bbox[0] for line in band),
        min(line.bbox[1] for line in band),
        max(line.bbox[2] for line in band),
        max(line.bbox[3] for line in band),
    )
    match = NUMBER.match(number_line.text)
    number = match.group(1) if match else ""
    return DocFormula(
        page=0,  # 调用方按当前页号填
        bbox=bbox,
        number=number,
        label=f"式 ({number})",
        text=" ".join(line.text for line in band),
        math_ratio=round(math / total, 2),
    )


def analyze(path: str | Path, *, max_pages: int = MAX_ANALYZE_PAGES) -> FormulaReport:
    """扫一遍前 `max_pages` 页，认出所有带编号的公式。结果落盘，同一份 PDF 只算一次。"""
    key = _disk_key(path, max_pages)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    doc = pymupdf.open(str(path))
    formulas: list[DocFormula] = []
    scanned: list[int] = []
    total = doc.page_count
    limit = min(max_pages, total)
    try:
        for index in range(limit):
            lines = _lines_of(doc[index])
            if not lines:
                # 没有文字层：扫描页。公式靠字形认，扫描件上没有字形可认。
                scanned.append(index + 1)
                continue
            right_edges = _edges(lines, side=2)
            seen: set[str] = set()
            for line in lines:
                if not NUMBER.match(line.text):
                    continue
                found = _formula_at(line, lines, right_edges)
                if found is None or found.number in seen:
                    continue
                seen.add(found.number)
                formulas.append(
                    DocFormula(
                        page=index + 1,
                        bbox=found.bbox,
                        number=found.number,
                        label=found.label,
                        text=found.text,
                        math_ratio=found.math_ratio,
                    )
                )
        report = FormulaReport(
            formulas=tuple(formulas),
            pages=total,
            analyzed_pages=limit,
            scanned_pages=tuple(scanned),
            note=_make_note(total, limit, formulas, scanned),
        )
    finally:
        doc.close()

    _store_cached(key, report)
    return report


def _make_note(total: int, limit: int, formulas: list[DocFormula], scanned: list[int]) -> str:
    if not formulas:
        if scanned:
            return (
                f"前 {limit} 页里有 {len(scanned)} 页是扫描页（没有文字层），"
                "公式需要字形信息才能定位，这份 PDF 提不出公式。"
            )
        return (
            "没有找到带编号的公式。有的论文（例如 GPT-3、BERT）把公式排成不编号的居中行，"
            "这一层只认作者写了编号的那些 —— 编号是我们唯一不敢猜的锚点。"
        )
    parts = [f"带编号的公式 {len(formulas)} 个"]
    if limit < total:
        parts.append(f"只解析了前 {limit} 页（共 {total} 页），后面的公式没有提取")
    if scanned:
        parts.append(f"跳过 {len(scanned)} 页扫描页（没有文字层）")
    return "，".join(parts) + "。"


# ------------------------------------------------------------------ 磁盘缓存

def _cache_dir() -> Path:
    root = Path(__file__).resolve().parents[4]
    store = Path(os.environ.get("PAPER_STORE_DIR") or root / "data" / "papers")
    return store.parent / "formula_cache"


def _disk_key(path: str | Path, max_pages: int) -> str | None:
    import hashlib

    try:
        stat = Path(path).resolve().stat()
    except OSError:
        return None
    raw = f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{max_pages}|{CACHE_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_cached(key: str | None) -> FormulaReport | None:
    if key is None:
        return None
    import json

    try:
        payload = json.loads((_cache_dir() / f"{key}.json").read_text(encoding="utf-8"))
        return FormulaReport(
            formulas=tuple(
                DocFormula(
                    page=item["page"],
                    bbox=tuple(item["bbox"]),
                    number=item["number"],
                    label=item["label"],
                    text=item["text"],
                    math_ratio=item["math_ratio"],
                )
                for item in payload["formulas"]
            ),
            pages=payload["pages"],
            analyzed_pages=payload["analyzed_pages"],
            scanned_pages=tuple(payload["scanned_pages"]),
            note=payload["note"],
            cached=True,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None  # 缓存坏了就当没有，重新扫一遍


def _store_cached(key: str | None, report: FormulaReport) -> None:
    if key is None:
        return
    import json

    payload = {
        "formulas": [
            {
                "page": f.page,
                "bbox": list(f.bbox),
                "number": f.number,
                "label": f.label,
                "text": f.text,
                "math_ratio": f.math_ratio,
            }
            for f in report.formulas
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
