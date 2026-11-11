"""页面级文本索引：把 PDF 变成「可逐字检索的带坐标词流」。

这是溯源能力的地基。设计目标只有一个：给定任意一段引用文本，
要么返回它在原文中的**精确页码 + 坐标框**，要么明确说"找不到"。

关键设计：
1. 以「词」为单位建索引，不是以「页」或「段」—— 只有到词级才能给出 bbox。
2. 归一化必须覆盖连字（ﬁ→fi）、大小写、标点，否则 v1 这类 PDF 永远匹配不上。
3. 匹配分两级：逐字命中（exact）优先；退而求其次做模糊匹配，
   但模糊匹配必须把「原文真实片段」一起返回，让上层能显式标注不确定性。
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace

import pymupdf

TOKEN_RE = re.compile(r"[a-z0-9]+")

# 一页文字层去掉空白后不足这么多字符，就当它是扫描页、改走 OCR。
# 与 `scan.MIN_PAGE_TEXT_CHARS` 是同一个数，理由见 `_page_words`。
MIN_PAGE_TEXT_CHARS = 30

# U+FB00–FB06 连字：不做映射的话，某种 PDF 生成器输出的 "ﬁrst" 会变成 "rst"
_LIGATURE_MAP = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",
        "ﬆ": "st",
    }
)


def normalize(text: str) -> str:
    """统一到可比较的形态：NFKC 拆连字 + 小写。"""
    return unicodedata.normalize("NFKC", text).translate(_LIGATURE_MAP).lower()


def split_tokens(text: str) -> list[str]:
    """切出纯字母数字 token，丢弃标点与空白。"""
    return TOKEN_RE.findall(normalize(text))


def _page_words(page) -> list:
    """取一页的词流：有文字层就用 PDF 自己的，没有就退到 OCR。

    判据与 `scan.analyze` 同口径（去空白后 ≥30 个字符算有文字层），
    保证"检测时说这页是扫描件"和"索引时真的去 OCR"是同一件事 ——
    两处各写一个阈值，迟早会出现"检测说扫描件、索引却按空页处理"。
    """
    words = page.get_text("words")
    if len("".join(w[4] for w in words)) >= MIN_PAGE_TEXT_CHARS:
        return words

    from app.parsing import scan  # 延迟导入：本模块被 chunker 依赖，放到顶层会绕回去

    return scan.ocr_words(page) or words


@dataclass(frozen=True, slots=True)
class Token:
    text: str  # 归一化后的 token
    raw: str  # 原文词形，用于回显
    page: int  # 1-based 页码
    rect: tuple[float, float, float, float]  # x0, y0, x1, y1（pt）
    # 同一个原文词切出来的 token 共用一个编号。回显时同一编号只拼一次 ——
    # 否则 "Self-attention" 切成 self/attention 两个 token，两个都带着整串 raw，
    # 引用框里就读成 "Self-attention, Self-attention, …"（实测自注意力那篇全是这样）
    word_id: int = -1


@dataclass(frozen=True, slots=True)
class Anchor:
    """一条引用的落地坐标。"""

    page: int
    rect: tuple[float, float, float, float]
    source_text: str  # 原文实际内容，用于向用户展示"这句才是原文"
    exact: bool  # True=逐字命中，False=模糊命中
    similarity: float
    token_span: tuple[int, int]  # 在词流中的起止下标，便于调试
    truncated: bool = False  # 引用尾部被截断（模型常见），已按最长可验证前缀命中
    # 这条引用来自哪个文档（paper_id）。`TextIndex.find` 不知道文档名，
    # 由校验器回填（`guardrails.citation.verify_claims`）—— 跨文献对比要靠它
    # 把每条锚点归到 A 还是 B 名下。
    doc: str = ""


class TextIndex:
    """单个 PDF 的词级索引。"""

    def __init__(self, pdf_path: str) -> None:
        self.path = pdf_path
        self.tokens: list[Token] = []
        self._postings: dict[str, list[int]] = defaultdict(list)

        word_no = 0

        def emit(raw: str, page_no: int, rect: tuple[float, float, float, float]) -> None:
            nonlocal word_no
            word_no += 1
            for piece in split_tokens(raw):
                pos = len(self.tokens)
                self.tokens.append(
                    Token(text=piece, raw=raw, page=page_no, rect=rect, word_id=word_no)
                )
                self._postings[piece].append(pos)

        doc = pymupdf.open(pdf_path)
        for page_no, page in enumerate(doc, start=1):
            # 扫描页的文字层是空的，"从空里找不到锚点"等于抗幻觉在扫描件上全线失效。
            # 所以这一页没有文字层时改走 OCR 的词流 —— 坐标是识别出来的，会偏，
            # 但"这段确实在原文里"这个判断仍然成立，高亮框也还落在附近的字上。
            words = _page_words(page)
            # 行尾断词必须在这里接上，否则索引和切块器说的不是同一种话：
            # PDF 原文是 `In addi-` / `tion to the masked …`，索引会得到
            # ["addi","tion"]，而切块器已经把这块文本修成了 "addition" ——
            # 模型照证据块逐字抄回来，"addition" 在索引里根本不存在，
            # 于是**真的引用被判成幻觉**（实测 BERT 上三条结论全挂）。
            # 接法与 `chunker._join_lines` 保持一致：连字符 + 小写开头才接。
            pending: tuple[str, tuple[float, float, float, float]] | None = None
            for word in words:
                x0, y0, x1, y1, raw = word[0], word[1], word[2], word[3], word[4]
                if pending is not None:
                    prev_raw, (px0, py0, px1, py1) = pending
                    if prev_raw.endswith("-") and raw[:1].islower():
                        raw = prev_raw[:-1] + raw  # addi- + tion → addition
                        x0, y0 = min(x0, px0), min(y0, py0)
                        x1, y1 = max(x1, px1), max(y1, py1)
                    else:
                        emit(prev_raw, page_no, (px0, py0, px1, py1))
                    pending = None
                pending = (raw, (x0, y0, x1, y1))
            if pending is not None:
                emit(pending[0], page_no, pending[1])
        doc.close()

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def texts(self) -> list[str]:
        return [t.text for t in self.tokens]

    # ------------------------------------------------------------------ 检索

    def find(
        self,
        quote: str,
        min_similarity: float = 0.90,
        *,
        max_drop: int = 3,
        min_tokens: int = 6,
    ) -> Anchor | None:
        """在文档中定位一段引用。

        三级策略：
        1. 逐字匹配（最理想）。
        2. 截断容错：模型经常把引用掐在句子中间（"journalis"），
           依次丢弃末尾 1~max_drop 个 token 再试，命中则标记 truncated。
           仍要求剩余 token 数 ≥ min_tokens，避免靠几个通用词蒙混过关。
        3. 模糊匹配：用「最稀有 token」定位候选窗口，避免全文滑窗。

        截断容错是严格的：被丢弃的尾部必须是原文紧接着那几个词的**前缀**
        （"journalis" 之于 "journalistic"）。否则一个"真句子 + 编造尾巴"
        就能靠砍掉尾巴骗过校验 —— 那等于给幻觉开后门。
        """
        query = split_tokens(quote)
        if not query:
            return None

        for drop in range(0, max_drop + 1):
            trimmed = query[: len(query) - drop] if drop else query
            if len(trimmed) < min_tokens:
                break
            anchor = self._find_exact(trimmed)
            if anchor is None:
                continue
            if drop == 0:
                return anchor
            if self._is_truncation_of(trimmed, query[len(trimmed) :], anchor.token_span[0]):
                return replace(anchor, truncated=True)

        return self._find_fuzzy(query, min_similarity)

    def _is_truncation_of(self, matched: list[str], dropped: list[str], start: int) -> bool:
        """判断被丢弃的尾部是否确实是原文的"半截"。

        规则：逐词与原文对齐；完全相同的词允许通过，
        最后一个词允许是原文对应词的前缀（真正的截断）。
        一旦出现对不上的词，判定为编造。
        """
        pos = start + len(matched)
        for i, tok in enumerate(dropped):
            if pos >= len(self.tokens):
                return False
            source = self.tokens[pos].text
            if tok != source:
                return i == len(dropped) - 1 and source.startswith(tok)
            pos += 1
        return False  # 尾部完整对齐却没匹配到，说明是多拼的词，不算截断

    def _find_exact(self, query: list[str]) -> Anchor | None:
        positions = self._postings.get(query[0])
        if not positions:
            return None
        n = len(query)
        texts = self.texts
        for start in positions:
            if start + n > len(texts):
                continue
            if texts[start : start + n] == query:
                return self._build_anchor(start, n, exact=True, similarity=1.0)
        return None

    def _find_fuzzy(self, query: list[str], min_similarity: float) -> Anchor | None:
        n = len(query)
        # 用最稀有的 token 做候选锚点，避免全文滑窗
        rare = min(query, key=lambda t: len(self._postings.get(t, ())))
        candidates = self._postings.get(rare, [])
        if not candidates:
            return None

        offset = query.index(rare)
        texts = self.texts
        best_ratio, best_start = 0.0, -1

        for pos in candidates:
            for start in (pos - offset, pos - offset - 1, pos - offset + 1):
                if start < 0 or start + n > len(texts):
                    continue
                ratio = difflib.SequenceMatcher(
                    None, query, texts[start : start + n], autojunk=False
                ).ratio()
                if ratio > best_ratio:
                    best_ratio, best_start = ratio, start

        if best_start >= 0 and best_ratio >= min_similarity:
            return self._build_anchor(best_start, n, exact=False, similarity=best_ratio)
        return None

    def _build_anchor(self, start: int, n: int, *, exact: bool, similarity: float) -> Anchor:
        span = self.tokens[start : start + n]
        x0 = min(t.rect[0] for t in span)
        y0 = min(t.rect[1] for t in span)
        x1 = max(t.rect[2] for t in span)
        y1 = max(t.rect[3] for t in span)
        # 一个原文词切出的多个 token 只回显一次，否则 "self-attention" 会被拼两遍
        words: list[str] = []
        last_word = -1
        for t in span:
            if t.word_id != last_word:
                words.append(t.raw)
                last_word = t.word_id
        return Anchor(
            page=span[0].page,
            rect=(x0, y0, x1, y1),
            source_text=" ".join(words),
            exact=exact,
            similarity=round(similarity, 6),
            token_span=(start, start + n),
        )
