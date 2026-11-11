"""首屏元信息：作者与机构。

赛题模块一要求"细粒度解析"里列出**标题、作者及机构信息**、摘要……标题早就有了，
作者和机构一直空着 —— 文献库里七篇论文，用户想按"谁写的、哪个组的"分辨，
只能自己点开 PDF 看首页。所以这里补上。

判据只有一条：**版面几何**。首页标题下方到"Abstract"之间的那一块，
按行的形状分成三类 ——
    · 带 `@` 的是邮箱行，跳过（邮箱域名能反推机构，但反推是猜）；
    · 含机构词（University / Google / Institute …）的是机构行；
    · 2~5 个词、每个词首字母大写、不含数字的是人名行。
两条原则照抄标题识别那一路：**只认版面，不信元数据**（arXiv 的元数据常年是错的）；
**认不出就留空，不猜**（扫描件首页没有文本层，宁可不写，也不给一个假机构）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.parsing.layout import Line, page_lines

# 首页机构行的常用词。只收"几乎不可能出现在人名里"的词 ——
# 收几个泛词（AI、Lab）看着更全，但会把 Simon 这种名字...
# 所以宁可漏几个罕见表述（"Max Planck Society"就认不出），也不误把人名当机构。
_ORG_WORDS = frozenset(
    """
    university université universitat universität universidad universita college school
    institute institut institution academy department dept faculty laboratory laboratories
    lab labs research center centre inc corp corporation company ltd gmbh limited group
    foundation hospital society sciences engineering technology polytechnic
    google deepmind microsoft meta facebook openai anthropic nvidia ibm apple amazon
    tencent alibaba baidu huawei bytedance sensetime moore
    tsinghua peking zhejiang shanghai jiaotong fudan harbin wuhan nanjing sichuan
    mit stanford berkeley carnegie oxford cambridge harvard princeton toronto
    """.split()
)

# 人名里允许出现的小写虚词：van der Waals、de la Fontaine、bin Laden
_PARTICLES = frozenset("van von der den de del della da di du bin al el la le dos".split())

# 排版器打在名字旁边的等贡献/脚注记号（CMSY 字体的 ∗ † ‡），以及上标数字
_MARKS = "∗*†‡§¶‖♯#"

# 名字里不该出现的词：这几行是章节名和页脚声明，形状上像人名（两三个大写词）
_NOT_NAMES = frozenset(
    """
    abstract introduction conclusion conclusions references appendix
    acknowledgments acknowledgements figure figuretable table page
    """.split()
)

_MAX_AUTHORS = 40
_MAX_AFFILIATIONS = 12


@dataclass(slots=True)
class FrontMatter:
    """一篇论文首页认出来的作者与机构。认不出就是两个空列表，不是 None。"""

    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.authors or self.affiliations)


def extract_front_matter(path: str | Path) -> FrontMatter:
    """从首页版面认出作者与机构。认不出返回空 FrontMatter（扫描件就是这条路径）。"""
    lines = page_lines(path)
    if not lines:
        return FrontMatter()

    band = _author_band(lines)
    fm = FrontMatter()
    seen_authors: set[str] = set()
    seen_affils: set[str] = set()

    for line in band:
        raw = " ".join(line.text.split())
        if not raw or "@" in raw:
            continue  # 邮箱行：域名能反推机构，但那是猜，不是读
        text = _strip_marks(raw)

        if _affiliation_shaped(text):
            for aff in _split_affiliations(text):
                key = aff.lower()
                if key not in seen_affils and len(fm.affiliations) < _MAX_AFFILIATIONS:
                    seen_affils.add(key)
                    fm.affiliations.append(aff)
            continue

        if _name_shaped(text):
            for name in _split_authors(text):
                key = name.lower()
                if key not in seen_authors and len(fm.authors) < _MAX_AUTHORS:
                    seen_authors.add(key)
                    fm.authors.append(name)

    return fm


# ---------------------------------------------------------------- 版面范围


def _author_band(lines: list[Line]) -> list[Line]:
    """标题下方到"Abstract"之间的那几行。

    上界取标题最后一行（标题本身不能进来：它也是"两三个大写词"）；
    下界取 "Abstract" / "ABSTRACT" / "Abstract—We propose…" 那行；
    整页找不到摘要标题（教材、报告的首封面）就只向下取 6 个标题高 ——
    再往下基本是版权与目录，那里面的机构名（"The MIT Press"）不是作者单位。
    """
    biggest = max(ln.size for ln in lines)
    title = sorted(
        (ln for ln in lines if biggest - ln.size <= 0.8), key=lambda ln: ln.y0
    )
    # 标题可能不止一行：纵向相邻的那几行都算，和标题识别的取法一致
    picked = [title[0]]
    for ln in title[1:]:
        if ln.y0 - picked[-1].y1 > biggest * 2:
            break
        picked.append(ln)
    top = min(ln.y0 for ln in picked)
    bottom = max(ln.y1 for ln in picked)

    limit = bottom + biggest * 6
    for ln in lines:
        if ln.y0 <= bottom:
            continue
        if _is_abstract_heading(ln.text):
            limit = ln.y0
            break
        if _is_section_heading(ln.text) and ln.y1 < limit:
            limit = ln.y0
            break

    # 上界用标题的**底边**而不是顶边：标题第二行（BERT 的 "Language Understanding"）
    # 顶边在底边之上，用顶边切会把它当成一个"两三个大写词"的人名。
    return [ln for ln in lines if ln.y0 >= bottom - 0.5 and ln.y1 <= limit]


_ABSTRACT = re.compile(r"^(abstract|summary)\b", re.IGNORECASE)
_SECTION = re.compile(r"^\d+(\.\d+)*\s+[A-Z]")


def _is_abstract_heading(text: str) -> bool:
    t = text.strip().strip(".:—- ")
    return bool(_ABSTRACT.match(t)) and len(t) < 24


def _is_section_heading(text: str) -> bool:
    return bool(_SECTION.match(text.strip()))


# ---------------------------------------------------------------- 行的形状


def _strip_marks(text: str) -> str:
    """去掉等贡献记号、上标数字，并压掉多余空格。

    `Ashish Vaswani∗` → `Ashish Vaswani`；`1 Google Brain` → `Google Brain`；
    `Aidan N. Gomez∗†` → `Aidan N. Gomez`。只动首尾与记号，词本身一个字不改。
    """
    out = "".join(" " if ch in _MARKS else ch for ch in text)
    words = [w for w in out.split() if not w.isdigit()]
    return " ".join(words).strip(" ,;:/·-")


def _affiliation_shaped(text: str) -> bool:
    """机构行：含机构词，短，且不像一句人话。"""
    words = text.split()
    if not words or len(words) > 8 or len(text) > 80:
        return False
    if _sentence_break(text):  # 两个句子：那是脚注正文（"∗Equal contribution. Listing…"）
        return False
    return any(_bare(w) in _ORG_WORDS for w in words)


def _sentence_break(text: str) -> bool:
    """这行是不是两句话粘在一起（脚注正文的典型形状）。

    **不能简单地看 ". "**：`Tom B. Brown`、`Simon J.D. Prince` 里也有句号加空格，
    那只是缩写。区别在于句号前面那个词长什么样 —— 缩写是短的纯大写
    （`B.` / `J.D.` / `Ph.D.`），除此之外的句号都意味着这行其实是两句话
    （`contribution.` / `Lee.`），而脚注正文正是这个形状，不该当成人名。
    """
    words = text.split()
    for word in words[:-1]:
        if not word.endswith("."):
            continue
        core = word.strip(".,;:")
        if len(core) > 1 and not (core.replace(".", "").isupper() and len(core) <= 4):
            return True
    return False


def _name_shaped(text: str) -> bool:
    """人名行：2~5 个词，每个词首字母大写，不含数字与句读。"""
    if not text or len(text) > 50:
        return False
    # 一行挤着好几位作者时，整行轮廓已经是"A, B and C"了 —— 拆开由 _split_authors
    # 负责，这里先按逗号分句，只要每一句都像人名，整行就算人名行
    chunks = [c for c in re.split(r"[,&]| and ", text) if c.strip()]
    if len(chunks) > 1:
        return all(_name_shaped(c) for c in chunks)
    words = text.split()
    if not 2 <= len(words) <= 5:
        return False
    if _sentence_break(text):  # "Kenton Lee. Abstract" 这种是两句话粘在一行
        return False
    return all(_word_ok(w) for w in words) and _has_a_real_name(text)


def _word_ok(word: str) -> bool:
    core = word.strip(".,;:")
    if not core:
        return False
    low = core.lower()
    if low in _PARTICLES:  # van / de / der：小写是正常的
        return True
    if low in _NOT_NAMES:  # Abstract / Introduction：大写得像名字，其实是章节名
        return False
    if not core[0].isupper():
        return False
    return all(ch.isalpha() or ch in "-'’." for ch in core)


def _has_a_real_name(text: str) -> bool:
    """至少有一个两字母以上的词 —— "A. B." 这种缩写堆不是名字。"""
    return any(len(w.strip(".,;:")) >= 2 and w.strip(".,;:").lower() not in _PARTICLES
               for w in text.split())


def _bare(word: str) -> str:
    return word.strip(".,;:()").lower()


# ---------------------------------------------------------------- 拆行


def _split_authors(text: str) -> list[str]:
    """一行里挤着好几位作者时拆开：`A, B, and C` → 三个人。

    拆完逐个复核形状，有一段不像人名就整行不动 —— 拆错比不拆更糟：
    把 "Google Research, Zurich" 拆成两位作者的名字，文献库里就多了两个人。
    """
    parts = [p.strip() for p in re.split(r"[,&]| and ", text) if p.strip()]
    if len(parts) < 2:
        return [text]
    if all(1 <= len(p.split()) <= 5 and _word_ok_loose(p) for p in parts):
        return parts
    return [text]


def _word_ok_loose(part: str) -> bool:
    words = part.split()
    return all(_word_ok(w) for w in words) and _has_a_real_name(part)


def _split_affiliations(text: str) -> list[str]:
    """`Google Brain, Google Research` 这类并列机构拆开；带编号的按编号拆。

    `1Google Brain 2Google Research` 去掉记号后已经粘成一行，
    只能按"下一个机构词之前"切 —— 这一条留到有人真的这么排版时再补，
    现在的作用是别把整串当成一个机构名。
    """
    parts = [p.strip() for p in re.split(r"[,;]", text) if p.strip()]
    if len(parts) < 2:
        return [text]
    if all(_affiliation_shaped(p) or len(p.split()) <= 6 for p in parts):
        return parts
    return [text]
