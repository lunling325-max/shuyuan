"""从论文的 References 节里抽出引用条目，供引用图谱使用。

为什么要自己写，不用现成的参考文献解析库？
1. 它们面向 BibTeX / 标准格式，而我们从 PDF 抽出来的是一坨：
   页码混在正文里、条目被分栏切成两半、连字符断词、大小写风格各异；
2. 我们**不需要**字段齐全的结构化数据 —— 图谱只问一件事：
   "这条引用指向的，是不是库里已有的某篇？" 标题（必要时加 arXiv 号）足以回答。

真实语料里参考文献有三种排版，必须都能吃：

| 样式 | 例子 | 切分方式 |
|---|---|---|
| 数字标记 | `[1] Jimmy Lei Ba, ...` | 按 `[n]` 切 |
| 字母键 | `[XDH+19] Qizhe Xie, ...` | 按 `[KEY]` 切 |
| 无标记（ACL 体） | `Alan Akbik, ... Vollgraf. 2018. Contextual ...` | 按空行切，再把跨栏断开的续段合回去 |

解析失败的条目**照样返回**（`title` 为空），因为图谱要如实显示"有 N 条没解析出来"，
而不是假装只有解析成功的那些存在。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.parsing.chunker import ARXIV_STAMP, Chunk

# 数字标记或字母键，必须顶到行首，否则正文里的 "see [3]" 也会被切开
_MARKED = re.compile(r"(?m)^\s*\[(\d{1,3}|[A-Za-z][A-Za-z0-9+]{1,10})\]\s*")
# 纯数字行是页码，混在参考文献里毫无信息量
_PAGE_NOISE = re.compile(r"(?m)^\s*\d{1,4}\s*$")
# 句点 + 空格才是句子边界；但 "Quoc V. Le" 里的缩写点后面也跟空格，不能切。
# 判据：点号前必须是**小写字母/数字/右括号**，这样 "V." "N." 这类首字母缩写不受影响。
_SEGMENT = re.compile(r"(?<=[a-z0-9\)\]])\s*\.\s+")
_ARXIV_ID = re.compile(r"(?:arxiv[:/\s]*|abs/)(\d{4}\.\d{4,5})", re.IGNORECASE)
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_PURE_YEAR = re.compile(r"^(?:19[5-9]\d|20[0-4]\d)\.?$|^\((?:19[5-9]\d|20[0-4]\d)\)\.?$")
# arXiv 新式编号 YYMM.NNNNN 长得就像年份：`arXiv:2004.08994` 里的 2004 是**月份**。
# 找年份之前先把它抹掉，否则 GPT-4 的参考文献里会有条目被标成 2004 年（正确是 2020）。
_ARXIV_LIKE = re.compile(r"\b\d{4}\.\d{4,5}\b")

# 出处关键字：见到这些，前面攒的就是标题，不是标题的一部分
# 参考文献之后紧跟的节标题：到这里就停。不停的话，附录正文会被当成引用，
# 而且附录里常带**自己的一份参考文献**（GPT-4 的 System Card 就是），
# 编号从 [1] 重新开始，两组混在一起 —— 实测 194 条里 89 条是附录。
_END_SECTION = re.compile(r"^(appendix|supplementary|author contributions|index)", re.IGNORECASE)

# 每页重复的书名页眉：`464 Bibliography` / `Bibliography 465`。
# `_page_blocks` 直接读原始页面块，绕过了切块器的页眉剔除，所以这里要再挡一次 ——
# 不挡的话每条跨页的首条引用都会以 `464 Bibliography Aiken, M., ...` 开头。
_RUNNING_HEAD = re.compile(
    r"^(?:\d{1,3}\s+)?(?:references|bibliography)(?::)?(?:\s+\d{1,3})?$", re.IGNORECASE
)

# 教材体在条目末尾标页码（`... Machine Learning, 224–232. 301`）。
# 这个尾数会让 `_looks_complete` 判成"没写完"（结尾不是句点），
# 于是下一条的作者行被并进上一条 —— 实测 500 条书目只解析出 255 条。
_TRAILING_PAGE_REF = re.compile(r"\.\s+\d{1,3}$")

# 作者-年份体：作者串后面紧跟 `(2019).`。年份要取这个，不是条目里最后一个四位数
# （出处里常写着会议年份 2020，条目本身是 2019 年的预印本）。
_AUTHOR_YEAR = re.compile(r"^[^()]{3,250}?\((\d{4})\)")

_VENUE_HINT = re.compile(
    r"^(?:in\s|proc\.|proceedings|conference|journal|transactions|arxiv|corr|"
    r"advances|pages|pp\.|vol\.|press|workshop|symposium|technical report|"
    r"university|mit|cambridge|springer|acm|ieee|naacl|acl|emnlp|iclr|icml|nips|neurips)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Reference:
    key: str  # 原始标记（"12" 或 "XDH+19"）；无标记样式用序号
    raw: str
    authors: str
    title: str  # 解析不出就是空串 —— 别编
    venue: str
    year: str
    arxiv_id: str

    @property
    def is_parsed(self) -> bool:
        return bool(self.title)


def _clean(text: str) -> str:
    """修 PDF 特有的脏东西：页码行、跨行断词、多余空白。

    断词有两种形态：块内还留着换行的 `Convolu-\\ntional`，以及块已经被压成
    单行的 `augment- ation`。只修前者的话，标题会带着 "augment- ation" 去比对，
    跟原文的 "augmentation" 一个词都对不上。
    """
    text = _PAGE_NOISE.sub("\n", text)
    text = re.sub(r"-\s*\n\s*|(?<=[a-z])-\s+(?=[a-z])", "", text)  # Convolu-\ntional / augment- ation
    text = re.sub(r"\s*\n\s*", " ", text)
    text = _TRAILING_PAGE_REF.sub(".", re.sub(r"\s+", " ", text).strip())
    return text.strip()


def _pick_year(body: str) -> str:
    """取参考文献的**出版年**。

    两个坑，都踩过：

    1. 不能取"第一个像年份的四位数"。arXiv 号 `arXiv:2004.08994` 里的 2004 是
       月份，比真正的年份写在更前面 —— GPT-4 的参考文献里就有条目被标成 2004，
       而它实际是 2020。
    2. 三种版面（数字标记 / 字母标记 / 无标记 ACL）都把年份写在**条目末尾**
       （`..., 2020.` / `..., pages 5998–6008, 2017.`），所以取最后一个才算准。

    抹掉 arXiv 号再取最后一个：实测五份样本 323 条，改动只落在错误年份上。
    """
    author_year = _AUTHOR_YEAR.match(body)
    if author_year:
        return author_year.group(1)
    years = _YEAR.findall(_ARXIV_LIKE.sub(" ", body))
    return years[-1] if years else ""


def parse_entry(key: str, raw: str) -> Reference:
    """把一条引用拆成 作者 / 标题 / 出处。

    难点是"哪一段是标题"：作者段之后可能先出现一个孤零零的年份（ACL 体写作
    `Authors. 2018. Title. Venue.`），也可能直接就是标题。遇到像出处的段就停，
    剩下的整段都算出处 —— 出处里本来就不需要细分。
    """
    body = _clean(_MARKED.sub("", raw, count=1))
    segs = [s.strip() for s in _SEGMENT.split(body) if s.strip()]
    authors = segs[0] if segs else ""
    title = venue = ""

    for i, seg in enumerate(segs[1:], start=1):
        if _PURE_YEAR.match(seg):
            continue  # ACL 体里被切成独立段的年份
        if _VENUE_HINT.match(seg):
            venue = " ".join(segs[i:])
            break
        if not title:
            title = seg
        else:
            break

    if not title:
        # 没找到标题段时，第一段多半就是标题（无标记样式里作者可能写在上一行）。
        # 但只有它**像个标题**才认：否则一串垃圾符号也会被算成"解析成功"，
        # 图谱上"有 N 条没解析出来"的统计就成了摆设。
        if len(re.findall(r"[A-Za-z]{2,}", authors)) >= 2:
            title, authors = authors, ""
    # 年份可以写在标题句尾（"… consistency training, 2019."）。
    # 不剥掉的话，"2019" 会混进标题的词集合，拿去和原文标题比对时白白拉低相似度。
    title = re.sub(r",?\s*\b(?:19[5-9]\d|20[0-4]\d)\.?$", "", title).strip()

    arxiv = _ARXIV_ID.search(body)
    year = _pick_year(body)
    return Reference(
        key=key,
        raw=body[:500],
        authors=authors[:200],
        title=title[:300],
        venue=venue[:200],
        year=year,
        arxiv_id=arxiv.group(1) if arxiv else "",
    )


# ---------------------------------------------------------------- 切分


def _by_markers(blob: str) -> list[tuple[str, str]] | None:
    marks = list(_MARKED.finditer(blob))
    if len(marks) < 3:  # 一两条多半是正文里的误命中，退回按段切
        return None
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(blob)
        out.append((m.group(1), blob[m.end() : end]))
    return out


def _looks_complete(text: str) -> bool:
    """一条引用写完了没有。只用来决定"下一段是新条目还是本条的续行"。

    判据三条缺一不可：有年份、以句点收尾、**正文够长**。
    第三条是踩出来的：ACL 体的条目头就是 `Joseph Turian, Lev Ratinov, and Yoshua
    Bengio. 2010.` —— 有年份、有句点，却是半条。少了长度判据，标题会另起一条，
    于是两条引用被劈成"作者头"和"标题+出处"，Transformer 那条就是这么丢的
    （实测它被粘进了上一条 Turian 的 raw 里）。
    """
    text = _TRAILING_PAGE_REF.sub(".", text.rstrip())
    if not _YEAR.search(text) or not text.endswith("."):
        return False
    # 有明确出处就更可能是写完了；否则要求长度，挡住"作者 + 年份"的半条
    return bool(_VENUE_HINT.search(text)) or len(text) >= 60


def _by_paragraphs(blob: str) -> list[tuple[str, str]]:
    """无标记样式：按空行切，再把没收尾的条目和后一段接回去。"""
    blocks = [" ".join(b.split()) for b in re.split(r"\n\s*\n", blob)]
    entries: list[str] = []
    for b in blocks:
        # 只丢掉纯数字（页码）和几乎没字母的碎块。
        # 门槛不能抬高：像 "2018." 这种短段正是条目的一部分，
        # 丢掉它会让上一条永远"没写完"，于是一路把后面的引用都吞进来。
        if not b or _PAGE_NOISE.fullmatch(b) or len(re.findall(r"[A-Za-z]", b)) < 2:
            continue
        if entries and not _looks_complete(entries[-1]):
            entries[-1] = f"{entries[-1]} {b}"
        else:
            entries.append(b)
    return [(str(i), e) for i, e in enumerate(entries, start=1)]


def _page_blocks(path: str | Path, first: int, last: int) -> str:
    """把参考文献所在页的段落块拼起来，从 "References" 标题处截断。

    为什么不直接拼切块文本？因为切块是按 token 预算切的，**会从条目中间切一刀**。
    实测 BERT：切块拼出来只有 59 条，其中一条是 "Quora question pairs. Christopher
    Clark and Matt Gardner. 2018. ..." —— 两条引用粘成一条，Transformer 那条直接丢了。
    换成页面段落块之后是 196 块，引用齐全。

    切块只用来**定位页码范围**，正文内容一律回原文取。
    """
    doc = pymupdf.open(str(path))
    try:
        parts: list[tuple[int, str]] = []
        for pno in range(first, last + 1):
            for block in doc[pno - 1].get_text("blocks"):
                text = " ".join(block[4].split())
                if not text or len(text) < 3 or ARXIV_STAMP.match(text):
                    continue
                parts.append((pno, text))
    finally:
        doc.close()

    def key(text: str) -> str:
        return re.sub(r"\s+\d{1,3}$", "", re.sub(r"^\d{1,3}\s+(?=\S)", "", text))

    # 页眉页脚：同一段文字在 ≥3 个**不同页面**出现。这条路读的是原始页面块，
    # 绕过了切块器的页眉剔除，于是教材每页的 `Draft: please send errata to …` 和
    # `This work is subject to a Creative Commons …` 还在。不清掉的话，
    # 它们会被当成一条"没写完的引用"，把后面几条的作者行一起吞进去（实测吞掉 3 条）。
    pages_of: dict[str, set[int]] = {}
    for pno, text in parts:
        pages_of.setdefault(key(text), set()).add(pno)
    furniture = {k for k, pages in pages_of.items() if len(pages) >= 3}
    texts = [t for _, t in parts if key(t) not in furniture]

    # 先定位节标题再切，切完才剔运行页眉 —— 顺序反了就找不到标题了
    # （标题行本身就长成 `Bibliography` 的样子）。
    for i, p in enumerate(texts):
        if not _RUNNING_HEAD.match(p):
            continue
        rest = texts[i + 1 :]
        for j, q in enumerate(rest):
            if _END_SECTION.match(q):
                rest = rest[:j]
                break
        return "\n\n".join(q for q in rest if not _RUNNING_HEAD.match(q))
    return "\n\n".join(texts)


def extract_references(chunks: list[Chunk], path: str | Path | None = None) -> list[Reference]:
    """从切块结果（与其对应 PDF）里取参考文献条目。没有 References 节就返回空列表。

    给了 `path` 就从原文页面段落块切（推荐，见 `_page_blocks` 的原因）；
    没给就退回拼切块文本 —— 那条路会被切块边界切断条目，只作为兜底。
    """
    # 论文写 References，教材写 Bibliography —— 只认前者的话，541 页的书在图上一条出边都没有
    ref_chunks = [
        c for c in chunks
        if c.section.strip().lower().startswith(("reference", "bibliograph"))
    ]
    if not ref_chunks:
        return []

    if path is not None:
        blob = _page_blocks(
            path,
            min(c.page_start for c in ref_chunks),
            max(c.page_end for c in ref_chunks),
        )
    else:
        blob = "\n\n".join(c.text for c in ref_chunks)

    parts = _by_markers(blob) or _by_paragraphs(blob)
    return [parse_entry(key, raw) for key, raw in parts if raw.strip()]


# ---------------------------------------------------------------- 标题比对


_PUNCT = re.compile(r"[^0-9a-z一-鿿]+")


def normalize_title(text: str) -> str:
    """标题归一化：去标点、折叠空白、小写。用于"这是不是同一篇"的判断。"""
    return _PUNCT.sub(" ", text.lower()).strip()


def title_similarity(a: str, b: str) -> float:
    """标题 Jaccard。参考文献里的标题常被截断或大小写不同，所以要模糊比。"""
    ta, tb = set(normalize_title(a).split()), set(normalize_title(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
