"""版本谱系：判断两份 PDF 是不是同一篇文献的不同版本，并说清改了什么。

两层能力，别混在一起看：

1. **是不是同一篇**（E-001 已标定）：5-gram shingle 的 Jaccard。
   同一篇不同版本 ≈ 0.72，不同论文 ≈ 0.00x，判别力差 100 倍，阈值可以放心定。
2. **改了什么**（本模块）：句级对齐后，还要**把噪声和实质改动分开**。

第 2 点才是真难点。朴素句级 diff 在 v1→v7 上吐出 64 个变更块，
其中 60 个是"改写"—— 但绝大多数只是脚注编号、行尾连字符、分栏顺序造成的伪差异。
直接丢给大模型总结，它会一本正经地编出一堆"作者修订了 X"，而 X 其实只是标点。

所以这里做三件事：
- 归一化要比"能对上"更进一步（拆连字、修行尾断词）—— 但**只修确定是伪影的东西**，
  凡是可能削掉真内容的规则一律不要，见 `normalize` 的说明；
- 逐句算相似度，**措辞微调判为 trivial 并丢弃**，只有新增/删除/实质改写才留；
- 相邻的变更合并成一段，避免同一处修改在摘要里被说三遍。
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# ---------------------------------------------------------------- 归一化

_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[0-9])")
MIN_SENT_CHARS = 25

# 阈值来自 E-001 的实测，不是拍脑袋
SAME_VERSION_THRESHOLD = 0.30
DISTINCT_THRESHOLD = 0.05


def normalize(text: str) -> str:
    """NFKC 拆连字 + 修行尾断词 + 折叠空白 + 小写。

    这里**故意不剥上标脚注编号**，尽管它确实是噪声源（"gradients 4. To counteract"）。
    试过两版正则：宽松版在两份 PDF 上命中 91/101 处，其中真脚注不到 5 处，
    把 "BLEU 41.0" "section 3.2" "Figure 2" 一起削成了 ".0" "section .2" "figure"；
    收紧到"句点+空白"之后还剩 7/11 处误伤（"See Figure 2. 3.3"），且删与不删
    在这对文件上给出**完全相同**的 diff（41 处、79 条 trivial）。

    也就是说：它没省下任何工作，却会让 "layer 5 of 6" 和 "layer 6 of 6" 归一化后相同，
    把真改动判成噪声。宁可留一点噪声，也不能对真修订视而不见。
    """
    text = unicodedata.normalize("NFKC", text).translate(_LIGATURES)
    text = re.sub(r"-\s*\n\s*", "", text)  # signifi-\ncantly -> significantly
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ---------------------------------------------------------------- 指纹


def shingles(text: str, k: int = 5) -> set[str]:
    tokens = normalize(text).split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


@dataclass(frozen=True, slots=True)
class Relation:
    """两份文档的关系。`kind` 决定上层该怎么提示用户。"""

    kind: str  # identical | new_version | related | distinct
    similarity: float

    @property
    def is_duplicate(self) -> bool:
        return self.kind in ("identical", "new_version")


def classify(score: float) -> Relation:
    """把相似度落成三态之一。阈值来自 E-001 的实测标定。"""
    if score >= SAME_VERSION_THRESHOLD:
        return Relation("new_version", round(score, 4))
    if score >= DISTINCT_THRESHOLD:
        return Relation("related", round(score, 4))
    return Relation("distinct", round(score, 4))


def relate(text_a: str, text_b: str) -> Relation:
    if normalize(text_a) == normalize(text_b):
        return Relation("identical", 1.0)
    return classify(jaccard(shingles(text_a), shingles(text_b)))


# ---------------------------------------------------------------- 句级差异


@dataclass(frozen=True, slots=True)
class Sentence:
    page: int
    text: str
    section: str = ""


@dataclass(slots=True)
class Change:
    kind: str  # insert | delete | replace
    old_pages: list[int]
    new_pages: list[int]
    old: list[str]
    new: list[str]
    section: str = ""  # 改动落在哪一节 —— 读者关心的是"3.2 改了"，不是"第 4 页改了"

    @property
    def is_reference_churn(self) -> bool:
        """参考文献编号整体位移不是"论文修订"，不该进摘要。"""
        return self.section.strip().lower().startswith("references")

    @property
    def page_label(self) -> str:
        pages = self.new_pages or self.old_pages
        if not pages:
            return "—"
        return f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}–{pages[-1]}"


@dataclass(slots=True)
class VersionDiff:
    old_pages: int
    new_pages: int
    old_sentences: int
    new_sentences: int
    changes: list[Change] = field(default_factory=list)
    trivial_dropped: int = 0

    @property
    def stats(self) -> dict[str, int]:
        counts = {"insert": 0, "delete": 0, "replace": 0}
        for c in self.changes:
            counts[c.kind] += 1
        return {**counts, "trivial_dropped": self.trivial_dropped}


def load_sentences(path: str | Path, *, annotate: bool = True) -> list[Sentence]:
    """按阅读顺序抽句子，并标上它属于哪一节。

    句子本身不带章节信息，但"改了哪一节"才是读者要的答案。
    这里用切块器的 `section` 反查：先按文本包含关系找，找不到再按页码兜底。
    """
    from app.parsing.chunker import ARXIV_STAMP, chunk_pdf

    doc = pymupdf.open(str(path))
    out: list[Sentence] = []
    try:
        for pno, page in enumerate(doc, start=1):
            for raw in _SENT_SPLIT.split(page.get_text()):
                # 投稿戳带版本号和日期，两版必然不同 —— 留着它，
                # 每份 PDF 都会凭空多出一条"删除 arXiv:... "的假改动
                text = " ".join(ARXIV_STAMP.sub(" ", raw).split())
                if len(normalize(text)) >= MIN_SENT_CHARS:
                    out.append(Sentence(page=pno, text=text))
    finally:
        doc.close()

    if not annotate or not out:
        return out

    try:
        chunks = chunk_pdf(str(path))
    except Exception:  # noqa: BLE001 —— 章节标注是锦上添花，失败不能拖垮 diff
        return out

    pool = [(normalize(c.text), c.section, c.page_start, c.page_end) for c in chunks]
    annotated: list[Sentence] = []
    for s in out:
        key = normalize(s.text)[:60]
        section = ""
        if key:
            section = next((sec for txt, sec, _a, _b in pool if key in txt), "")
        if not section:
            section = next(
                (sec for _t, sec, a, b in pool if a <= s.page <= b), ""
            )
        annotated.append(Sentence(page=s.page, text=s.text, section=section))
    return annotated


TRIVIAL_RATIO = 0.90  # 高于此值只算措辞微调（标点、大小写、脚注位置）
MATCH_RATIO = 0.50  # 低于此值就认为两句毫无关系，一个是新增、一个是删除


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b), autojunk=False).ratio()


def _refine(
    old_span: list[Sentence],
    new_span: list[Sentence],
    *,
    drop_trivial: bool,
) -> tuple[list[Change], int]:
    """把一个大变更块拆成逐句的真实改动。

    做法是两边句子做一次贪心最优配对（块内句数通常 < 50，平方复杂度无所谓），
    配上的再按相似度分档：几乎一样→噪声丢弃；差得多→实质改写；配不上的→增/删。
    """
    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(old_span):
        for j, b in enumerate(new_span):
            r = _ratio(a.text, b.text)
            if r >= MATCH_RATIO:
                pairs.append((r, i, j))
    pairs.sort(reverse=True)

    used_old: set[int] = set()
    used_new: set[int] = set()
    matched: list[tuple[float, int, int]] = []
    for r, i, j in pairs:
        if i in used_old or j in used_new:
            continue
        used_old.add(i)
        used_new.add(j)
        matched.append((r, i, j))

    out: list[Change] = []
    trivial = 0
    for r, i, j in sorted(matched, key=lambda t: t[1]):
        if r >= TRIVIAL_RATIO:
            if drop_trivial:
                trivial += 1
                continue
        out.append(
            Change(
                kind="replace",
                old_pages=[old_span[i].page],
                new_pages=[new_span[j].page],
                old=[old_span[i].text],
                new=[new_span[j].text],
                section=new_span[j].section or old_span[i].section,
            )
        )

    # 没配上的句子：一边有、另一边没有 —— 这才是真正的增删
    for i, s in enumerate(old_span):
        if i not in used_old:
            out.append(Change("delete", [s.page], [], [s.text], [], s.section))
    for j, s in enumerate(new_span):
        if j not in used_new:
            out.append(Change("insert", [], [s.page], [], [s.text], s.section))

    out.sort(key=lambda c: (c.new_pages or c.old_pages or [0])[0])
    return out, trivial


def diff_versions(
    old_path: str | Path,
    new_path: str | Path,
    *,
    drop_trivial: bool = True,
) -> VersionDiff:
    """句级对齐 + 噪声过滤 + 相邻合并。"""
    old = load_sentences(old_path)
    new = load_sentences(new_path)
    sm = difflib.SequenceMatcher(
        None, [s.text for s in old], [s.text for s in new], autojunk=False
    )

    raw: list[Change] = []
    dropped = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old_span, new_span = old[i1:i2], new[j1:j2]

        # 大块"改写"不能当原子处理：41 句对 33 句的块里，往往只有 2 句真改了，
        # 其余是分栏顺序或表格抽取顺序变了。进去做句对级再对齐，只留下真动过的。
        parts, trivial_here = _refine(old_span, new_span, drop_trivial=drop_trivial)
        raw.extend(parts)
        dropped += trivial_here

    return VersionDiff(
        old_pages=_page_count(old_path),
        new_pages=_page_count(new_path),
        old_sentences=len(old),
        new_sentences=len(new),
        changes=_merge_adjacent(raw),
        trivial_dropped=dropped,
    )


def _merge_adjacent(changes: list[Change]) -> list[Change]:
    """同一页上挨着的变更合成一条。

    不合并的话，一处改写会被切成三条记录，摘要把同一件事说三遍，
    看起来像"改了很多"，实际是一处。
    """
    merged: list[Change] = []
    for c in changes:
        if merged and _touches(merged[-1], c):
            last = merged[-1]
            last.new_pages = sorted({*last.new_pages, *c.new_pages})
            last.old_pages = sorted({*last.old_pages, *c.old_pages})
            last.old.extend(c.old)
            last.new.extend(c.new)
            continue
        merged.append(c)
    return merged


def _touches(a: Change, b: Change) -> bool:
    # 跨节的改动绝不合并：合并后那条 label 只能写其中一节，等于撒谎
    if a.section != b.section:
        return False
    pa = set(a.new_pages or a.old_pages)
    pb = set(b.new_pages or b.old_pages)
    if not pa & pb:
        return False
    # 同页且都是"有增无减"的插入，可以并；一增一删不能并，那是两件事
    return (a.kind == b.kind) or len(a.old) == 0 or len(b.old) == 0


def _page_count(path: str | Path) -> int:
    doc = pymupdf.open(str(path))
    try:
        return doc.page_count
    finally:
        doc.close()
