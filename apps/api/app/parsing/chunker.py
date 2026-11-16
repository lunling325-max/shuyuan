"""结构感知切块：PDF → 带章节路径与坐标的语义块。

为什么不用定长切？
1. **定位**：检索出来的块必须能回答"这段在论文哪一节"，章节路径是溯源的一部分。
2. **语义**：定长切会把标题和正文切开，模型拿到的块开头是半句没有主语的话。
3. **抗幻觉**：页眉页脚、页码、"arXiv:xxxx" 这类每页重复的噪声混进块里，
   会诱导模型把它们当成论文内容（E-002 就是这么翻车的）。

切块规则：
- 以段落（PDF 的 block）为单位，**绝不切开段落**；段落超长才按句号切。
- 标题永远开启新块；标题同时进入 section 路径。
- 在 ≥30% 页面重复出现的页边内容视为页眉页脚，整块丢弃。
- 行尾连字符接上一行（`signi-\\ncantly` → `significantly`），
  连字（ﬁ → fi）用与 TextIndex 同一套归一化。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

import pymupdf

from app.parsing.text_index import normalize, split_tokens

# 预算：目标是 350 token 一块，硬上限 700。
# 太小的块检索召回碎片化，太大的块把不相关信息一起喂给模型。
TARGET_TOKENS = 350
MAX_TOKENS = 700
MIN_TOKENS = 60

_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s*(\S.*)$")
_UPPER_HEADING = re.compile(r"^[A-Z][A-Z0-9 &\-:,]{3,40}$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
# arXiv 左侧那行旋转的投稿戳："arXiv:1706.03762v1  [cs.CL]  12 Jun 2017"。
# 它 20pt，比论文标题（17.2pt）还大，字号规则必然把它当一级标题；
# 而它是竖排在左边距里的，既不在页眉带也不在页脚带，重复出现也剔不掉。
# 两头都出事：当标题时整页正文挂到假章节下、目录多一条；
# 不当标题时它又混进正文，版本对比里出现"删除 arXiv:... "这种鬼话。
# 所以直接按内容扔掉 —— 这个模式足够独特，不会误伤引用它的正文。
ARXIV_STAMP = re.compile(r"arxiv:\d{4}\.\d{4,5}v\d+\s*\[[a-z\-]+\.[a-z]{2}\]", re.I)
# 目录页的判据：点引导线成串出现（`3 Results 10 . . . . . . 11`）。
# GPT-3 第 2 页整页是自己的目录，`8 Conclusion 40` 这种行正好符合"编号 + 短标题"
# 的标题判据，于是目录条目变成了章节名，还排在 `1 Introduction` 前面；
# 那块目录文本本身也会被当成证据召回。目录是导航不是内容，整页不进索引。
# 实测五份样本：只有 GPT-3 第 2 页命中（16 处），其余每页都是 0，阈值 3 很安全。
_LEADER_RUN = re.compile(r"(?:\.\s){4,}|\.{5,}")
# 一页文字层去空白后少于这么多字符，才值得让 OCR 再读一遍。
# 与 scan.py 的 MIN_PAGE_TEXT_CHARS 同一个数量级，但这里更保守：
# 宁可少认一页，也不要让 OCR 去覆盖一页本来就有文字的页（OCR 会丢公式和字号）。
_OCR_MIN_CHARS = 10
_LETTER = re.compile(r"[A-Za-z]")
# 比正文小到这个比例的段落直接不当标题：图表里的字（表头单元格、坐标轴刻度、
# 图注编号）实测只有正文的三到五成高。
#
# 取 0.7 是量出来的，不是拍的。FSA 论文（2508.18224）正文 10.0pt：
#
# | 类别 | 字号 | 占正文 |
# |---|---|---|
# | 真子章节（2.1 / 3.1 / 4.4 …） | 8.41 ~ 8.86 | 0.84 ~ 0.89 |
# | 图表碎字（`64 Seqlen (K)`、`2.78 H100 NVL`、`Q3 Q6`） | 3.2 ~ 5.5 | 0.32 ~ 0.55 |
#
# 先按 0.9 卡过一次，结果把**全部子章节**一起卡掉了 —— LaTeX 双栏模板的
# \subsection 本来就比正文小一号。真正的空档在 0.55 与 0.84 之间，门槛落在 0.7。
MIN_HEADING_SIZE_RATIO = 0.7
# 页码粘在标题两头：`2 1 Introduction`（页眉）、`1.1 Supervised learning 3`（页脚编号）
_PAGE_PREFIX = re.compile(r"^\d{1,3}\s+(?=\d+(?:\.\d+)*\.?\s+\S)")
_TRAILING_PAGE = re.compile(r"\s+\d{1,3}$")
_KEYWORD_HEADINGS = {
    "abstract", "introduction", "conclusion", "conclusions", "references",
    "acknowledgments", "acknowledgements", "appendix", "discussion", "results",
}


@dataclass(frozen=True, slots=True)
class Para:
    """一个 PDF 文本块，通常是自然段。"""

    page: int
    bbox: tuple[float, float, float, float]
    text: str
    size: float
    bold: bool
    # `size` 是不是代理值（OCR 块的像素墨迹高度）。字号判标题那条规则在它上面失效，
    # 原因见 `_heading`。
    size_proxy: bool = False


@dataclass(frozen=True, slots=True)
class Chunk:
    """检索的最小单位。坐标是「页 → 框」的列表，用于前端高亮。"""

    index: int
    text: str
    section: str
    page_start: int
    page_end: int
    n_tokens: int
    rects: tuple[tuple[int, tuple[float, float, float, float]], ...]

    def preview(self, n: int = 90) -> str:
        flat = " ".join(self.text.split())
        return flat[:n] + ("…" if len(flat) > n else "")


# ----------------------------------------------------------------- 抽取

def _join_lines(lines: list[str]) -> str:
    """把同一块里的行拼成一段，处理行尾断词。"""
    out = ""
    for raw in lines:
        piece = raw.strip()
        if not piece:
            continue
        if not out:
            out = piece
        elif out.endswith("-") and piece[:1].islower():
            # signi- / cantly → significantly（半角与全角连字符都算）
            out = out[:-1] + piece
        else:
            out += " " + piece
    return out


def _block_style(block: dict) -> tuple[float, bool]:
    sizes, bold_chars, total = [], 0, 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            txt = span.get("text", "")
            n = len(txt.strip())
            if not n:
                continue
            sizes.append((span.get("size", 0.0), n))
            total += n
            if "Bold" in span.get("font", "") or span.get("flags", 0) & 16:
                bold_chars += n
    if not sizes:
        return 0.0, False
    size = sum(s * n for s, n in sizes) / max(total, 1)
    return size, bold_chars >= total * 0.6


def _size_is_proxy(block: dict) -> bool:
    """这个块的 `size` 是代理值吗（OCR 拼出来的块，字号来自像素墨迹高度）。"""
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("text", "").strip():
                return span.get("font") == "ocr"
    return False


def _boiler_key(text: str) -> str:
    """页眉页脚的比对键：**只**把页码去掉，其余原样。

    页码每页都变，逐字比对永远配不上 —— `2 1 Introduction` 与 `4 1 Introduction`
    是同一个页眉。但不能把数字全抹成 `#`：那样 `Figure 1.4` 和 `Figure 1.5`
    会撞成一个键，图注就被当成页眉删掉了。

    前导页码要**无条件**剥：早先只在后面还跟着编号时才剥，于是教材的
    `464 Bibliography` 一页一个键（541 页里 50 个键，谁都不重复），
    一条页眉都剔不掉 —— 更要命的是它被 `_NUMBERED_HEADING` 认成标题，
    把章节栈在每页打回 `464 Bibliography`，整本 30 页书目只剩第 1 页算数。
    """
    s = " ".join(split_tokens(text))
    s = re.sub(r"^\d{1,3}\s+(?=\S)", "", s)  # 页眉前的页码
    s = re.sub(r"\s+\d{1,3}$", "", s)        # 页脚后的页码
    return "#page" if re.fullmatch(r"\d{1,3}", s) else s


def _repeats_within(pages: list[int], count: int = 3, window: int = 8) -> bool:
    """同一段页边文本是否在 `window` 页的窗口内出现过 `count` 次以上。

    教材的页眉是**隔页**出现的（偶数页才有），连续判据抓不到；
    逐章更换又让"出现次数过半"（541 页 = 163 次）永远达不到。
    窗口判据两头都够得着，而正文没可能在 8 页里一字不差地重复三次。
    """
    if len(pages) < count:
        return False
    for i in range(len(pages) - count + 1):
        if pages[i + count - 1] - pages[i] < window:
            return True
    return False


def page_needs_ocr(blocks: list[dict]) -> bool:
    """这一页的文字层是不是不够用（不够就该走 OCR）。

    对外公开：`references._page_blocks` 也要做同一个判断，否则扫描件的
    参考文献会去读一份空的文字层 —— 切块器认得 `References` 标题，
    参考文献解析却在原文页面上什么也读不到，整份扫描件照样 0 条引文。
    """
    return _page_chars(blocks) < _OCR_MIN_CHARS


def _page_chars(blocks: list[dict]) -> int:
    """一页文字层的有效字符数（只算文本块里的 span，空白不算）。

    注意是**字符数**不是词数：`len(text.split())` 数的是词，一行 41 个字符的
    正文会得到 7 —— 阈值按字符定的时候用错单位，正常页面就会被当成扫描页。
    """
    total = 0
    for block in blocks:
        if block.get("type") != 0:
            continue
        text = "".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        total += len("".join(text.split()))
    return total


def extract_paras(pdf_path: str, *, ocr=None) -> tuple[list[Para], int]:
    """抽出所有自然段，并剔除页眉页脚。返回 (段落列表, 总页数)。

    `ocr` 是可选钩子：`(page) -> dict | None`，只在某页文字层几乎为空时调用。
    为什么不在这里直接绑定某个 OCR 引擎（RapidOCR / Tesseract）：切块器是纯解析逻辑，
    不该知道自己跑在装了 OCR 的机器还是没装的机器上；而且测试能注入假钩子，
    不必为了跑用例去加载一份模型权重。
    """
    doc = pymupdf.open(pdf_path)
    per_page: list[list[tuple[Para, bool]]] = []  # bool: 是否页边区域

    for page_no, page in enumerate(doc, start=1):
        height = page.rect.height or 1.0
        items = []
        blocks = page.get_text("dict")["blocks"]
        if ocr is not None and page_needs_ocr(blocks):
            # 扫描页：文字层是空的，但版面还在。让 OCR 把那页"读"成带 bbox 的
            # blocks，后续的标题识别、页眉剔除、坐标高亮就能原样复用。
            ocr_blocks = ocr(page)
            if ocr_blocks:
                blocks = ocr_blocks.get("blocks", [])
        for block in blocks:
            if block.get("type") != 0:
                continue
            lines = [
                "".join(s.get("text", "") for s in line.get("spans", []))
                for line in block.get("lines", [])
            ]
            text = " ".join(_join_lines(lines).split())
            if not text or ARXIV_STAMP.match(text):
                continue
            x0, y0, x1, y1 = block["bbox"]
            size, bold = _block_style(block)
            para = Para(
                page=page_no,
                bbox=(x0, y0, x1, y1),
                text=text,
                size=size,
                bold=bold,
                size_proxy=_size_is_proxy(block),
            )
            # 页边带放宽到 16% / 90%：教材的页眉在页面 14.9% 处，7.5% 的窄带根本够不着，
            # 于是整本书的页眉没被剔掉，每页还把章节栈打回顶层。
            # 放宽不会误伤正文 —— 正文句子不会在连续多页里一字不差地重复。
            edge = y1 < height * 0.16 or y0 > height * 0.90
            items.append((para, edge))
        per_page.append(items)

    n_pages = len(per_page)
    threshold = max(2, math.ceil(n_pages * 0.3))

    # 页眉页脚识别：归一化文本在多个页面的页边区域重复出现。
    # 页码必须一起归一化掉 —— 教材的页眉写作 `2 1 Introduction`、`3 1 Introduction`，
    # 逐字比对时每页都不一样，于是整本书的页眉一条都剔不掉，
    # 而且每页都把章节栈打回顶层，子节全部丢失父级。
    # 两条判据取并集：
    #   ① 出现次数过半 —— 论文那种"奇数页作者名、偶数页标题"的页眉，每条只隔页出现；
    #   ② 连续 ≥3 页出现 —— 教材的页眉是**每章一变**的（`2 1 Introduction`），
    #      一章才十几页，永远到不了"30% 页数"（541 页 = 163 次）这条线。
    body_size = _body_size([para for items in per_page for para, _ in items])
    seen: dict[str, int] = {}
    pages_of: dict[str, list[int]] = {}
    for page_no, items in enumerate(per_page, start=1):
        for para, edge in items:
            # 大字号的东西**不是**页眉页脚，哪怕它长在页边、每页都出现。
            # 教材每页页眉都写着 `464 Bibliography`，把页码剥掉之后它就和
            # 第 477 页真正的 `Bibliography` 标题同键；不按字号分开，
            # 真标题会被当作页眉删掉，整个书目节从此不存在。
            if edge and para.size <= body_size * 1.12:
                key = _boiler_key(para.text)
                if key:
                    seen[key] = seen.get(key, 0) + 1
                    pages_of.setdefault(key, []).append(page_no)
    boiler = {k for k, c in seen.items() if c >= threshold}
    boiler |= {k for k in pages_of if _repeats_within(pages_of[k])}

    paras: list[Para] = []
    for items in per_page:
        for para, edge in items:
            if edge and para.size <= body_size * 1.12 and _boiler_key(para.text) in boiler:
                continue
            if len(split_tokens(para.text)) == 0:
                continue
            paras.append(para)
    doc.close()
    return paras, n_pages


# ----------------------------------------------------------------- 标题

def _body_size(paras: list[Para]) -> float:
    """正文字号：按字符数加权的众数区间，比算术平均稳。"""
    buckets: dict[int, int] = {}
    for p in paras:
        key = round(p.size * 2) / 2  # 0.5pt 分桶
        buckets[key] = buckets.get(key, 0) + len(p.text)
    return max(buckets.items(), key=lambda kv: kv[1])[0] if buckets else 10.0


def _plausible_heading(tail: str) -> bool:
    """编号后面跟的那截得**像标题**，不能是公式。

    541 页教材踩的坑：显示公式常以数字开头（`0 + ϕ′ 1a [ψ10 + …`），
    正好命中"编号 + 空格 + 任意内容"的标题判据。实测 953 块里九成的章节名
    是公式碎片，阅读器目录和"每条改动都标章节"一起报废。

    判据只用两条，够用且不误伤：首字符是字母（或汉字）；字母占比 ≥ 0.45。
    """
    tail = tail.strip()
    if not tail or not (tail[0].isalpha() or "一" <= tail[0] <= "鿿"):
        return False
    letters = len(_LETTER.findall(tail))
    return letters >= 4 and letters / max(len(re.sub(r"\s", "", tail)), 1) >= 0.45


def _reject_heading_text(text: str) -> bool:
    """四类"看着像标题、其实是别的东西"的硬特征。

    在 541 页教材上量过：406 个章节名里 5 个是假的，其中 4 个以 `0 ` 开头 ——
    显示公式常以「0 …」起头（`0 z < 0 1 z ≥0 rect[z] =`），正好命中
    「编号 + 空白 + 任意内容」那条判据。另外三类同样只靠硬特征排除，
    不去猜语义，免得把真标题一起误伤：

    | 假标题 | 判据 |
    |---|---|
    | `0 z < 0 1 z ≥0 rect[z] =` | 章节号是 0；或含 `= < > ≤ ≥ ^ _` |
    | `0 that controls the overall height…(Interactive figure)` | 含交互图说明 |
    | `1. This allows us to calculate the parameters ϕ[h1]…` | 编号 + 句子，词数超过 12 |

    真标题不会同时踩中这几条：编号从 1 起、不含数学符号、不超过十余词。
    """
    t = text.strip()
    if re.match(r"^0(?:\.|\s)", t):
        return True
    if "Interactive figure" in t:
        return True
    if re.search(r"[=<>≤≥^_]", t):
        return True
    if re.match(r"^\d+\.\s+[A-Z][a-z]+", t) and len(t.split()) > 12:
        return True
    return False


def _heading(para: Para, body_size: float) -> tuple[int, str] | None:
    """判断是否标题，返回 (层级, 标题文本)。"""
    text = para.text.strip()
    n_tok = len(split_tokens(text))
    if n_tok == 0 or n_tok > 25 or _reject_heading_text(text):
        return None

    # 图表里的字比正文小得多，而"编号 + 空格 + 内容"那条判据不看字号，
    # 于是表头单元格和坐标轴刻度全被当成了章节标题。实测 FSA 论文（2508.18224）
    # 目录里冒出三条假标题，字号一看就分得开：
    #
    # | 误判的标题 | 字号 | 真子章节 | 正文 |
    # |---|---|---|---|
    # | `64 Seqlen (K)`（图 9 的坐标轴刻度） | 5.5 pt | 8.4~8.9 pt | 10.0 pt |
    # | `2.78 H100 NVL`（表 3 的单元格） | 4.6 pt | 同上 | 同上 |
    # | `Q3 Q6`（图注里的问答编号） | 3.2 pt | 同上 | 同上 |
    #
    # 所以判据放在所有分支之前：比正文小很多的段落不可能是标题。
    # 门槛取 0.7 而不是 0.9 —— 理由见 `MIN_HEADING_SIZE_RATIO` 那张表，
    # 0.9 会把本来就比正文小一号的子章节一起卡掉。
    # 扫描件跳过 —— 它的 `size` 是 OCR 的墨迹高度，跟字号不是一回事。
    if not para.size_proxy and body_size and para.size < body_size * MIN_HEADING_SIZE_RATIO:
        return None

    bare = text.rstrip(".:;")
    if bare.lower() in _KEYWORD_HEADINGS:
        return 1, bare

    # 页眉页脚会把页码粘在标题头上/尾上：`2 1 Introduction`、`1.1 Supervised learning 3`。
    # 页码每个页面都变，按"重复出现"剔不掉，只能从标题里剥。
    # 剥完要重新过一次编号判据 —— 不成立就说明原本也不是标题。
    body = _TRAILING_PAGE.sub("", _PAGE_PREFIX.sub("", text)).strip()
    m = _NUMBERED_HEADING.match(body)
    if m and len(m.group(2)) > 2 and not body.endswith("."):
        head, tail = m.group(1), m.group(2)
        # 编号和标题之间**一个空白都没有**才算粘连。判据是"分隔段里有没有空白字符"，
        # 不是"分隔段 strip 之后是不是空" —— 后者会把正常的单个空格也判成粘连。
        glued = not re.search(r"\s", body[len(head) : len(body) - len(tail)])
        # 扫描件（`size_proxy`）要放宽两处，都是 OCR 自己的毛病：
        #   ① 编号和标题粘成一个词 —— `1Introduction`、`3.2Attention`、`7Conclusion`；
        #   ② 标题和下一段正文并成一块 —— `3.1 Encoder and Decoder Stacks Encoder:
        #      The encoder is composed of a stack…`。
        # 所以代字号块的硬约束就是"尾巴短"：≤6 词才可能是标题，再长就是正文。
        # 文字层的 PDF 照旧走原判据（必须有分隔空白），一个字都不放宽。
        if para.size_proxy:
            # 代字号的块还要求首字母大写（汉字同理）：`11 ion@google.com`
            # 是首页的作者单位碎片，字母占比 0.86、词数 1，光看前面两条必中。
            ok = len(tail.split()) <= 6 and (tail[0].isupper() or "一" <= tail[0] <= "鿿")
        else:
            ok = not glued
        if ok and _plausible_heading(tail):
            return head.count(".") + 1, (f"{head} {tail}" if glued else body)

    # 字号显著大于正文，且不是以句末标点结尾（排除首句）。
    # 扫描件（`size_proxy`）走不了这条：OCR 的"字号"是一行的像素墨迹高度，
    # 实测同一份论文正文 13~19pt、真标题 13~20pt —— 两个区间几乎完全重叠。
    # 拿 1.12 倍去判等于按噪声挑标题：15 页扫描件里既漏掉大半真标题，
    # 又把参考文献里的条目行（`[13] Zhongqiang Huang and Mary Harper...`）
    # 判成标题，整个 References 节被切成七八段。
    if (
        not para.size_proxy
        and para.size >= body_size * 1.12
        and not text.endswith((".", "?", "!"))
        and _plausible_heading(text)
    ):
        return 1, text

    if para.bold and _UPPER_HEADING.match(text):
        return 1, text

    return None


# ----------------------------------------------------------------- 切块

def _split_oversize(text: str) -> list[str]:
    """超长段落按句子切，返回若干段，每段尽量不超过 TARGET_TOKENS。"""
    sentences = _SENTENCE_SPLIT.split(text)
    out, buf, n = [], "", 0
    for sent in sentences:
        k = len(split_tokens(sent))
        if buf and n + k > TARGET_TOKENS:
            out.append(buf.strip())
            buf, n = "", 0
        buf += (" " if buf else "") + sent
        n += k
    if buf.strip():
        out.append(buf.strip())
    return out


def _toc_pages(paras: list[Para]) -> set[int]:
    """整页都是目录的页码。判据见 `_LEADER_RUN`。"""
    joined: dict[int, str] = defaultdict(str)
    for p in paras:
        joined[p.page] += p.text + " "
    return {page for page, text in joined.items() if len(_LEADER_RUN.findall(text)) >= 3}


def chunk_pdf(pdf_path: str, *, ocr=None) -> list[Chunk]:
    """把 PDF 切成带章节路径的块。`ocr` 见 `extract_paras`。"""
    paras, _ = extract_paras(pdf_path, ocr=ocr)
    if not paras:
        return []

    toc = _toc_pages(paras)
    body_size = _body_size(paras)
    chunks: list[Chunk] = []
    stack: list[tuple[int, str]] = []
    buf: list[Para] = []
    buf_tokens = 0

    def section_path() -> str:
        # 只保留最深的 2 级：孙级路径对模型没有增量信息，只会挤占 token 预算
        return " › ".join(title for _, title in stack[-2:])

    def flush(buf: list[Para], section: str) -> None:
        if not buf:
            return
        text = "\n\n".join(p.text for p in buf)
        if not split_tokens(text):
            return
        rects = tuple((p.page, p.bbox) for p in buf)
        pages = [p.page for p in buf]
        chunks.append(
            Chunk(
                index=len(chunks),
                text=text,
                section=section,
                page_start=min(pages),
                page_end=max(pages),
                n_tokens=len(split_tokens(text)),
                rects=rects,
            )
        )

    for para in paras:
        if para.page in toc:
            continue  # 目录页整页跳过：不进索引，也不许它冒充章节标题
        head = _heading(para, body_size)
        if head:
            level, title = head
            if buf_tokens >= MIN_TOKENS:
                flush(buf, section_path())
                buf, buf_tokens = [], 0
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue

        n = len(split_tokens(para.text))
        if n > MAX_TOKENS:
            if buf:
                flush(buf, section_path())
                buf, buf_tokens = [], 0
            for piece in _split_oversize(para.text):
                flush([Para(para.page, para.bbox, piece, para.size, para.bold)], section_path())
            continue

        if buf_tokens + n > MAX_TOKENS and buf_tokens >= MIN_TOKENS:
            flush(buf, section_path())
            buf, buf_tokens = [], 0

        buf.append(para)
        buf_tokens += n

        if buf_tokens >= TARGET_TOKENS:
            flush(buf, section_path())
            buf, buf_tokens = [], 0

    flush(buf, section_path())
    return chunks
