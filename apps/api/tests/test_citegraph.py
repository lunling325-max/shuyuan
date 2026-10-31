"""引用图谱的测试。

图谱最容易出的错**不是崩溃，是安静地说错话**：
同一个 PageRank 值算错方向，排名照样画得出来，只是榜首换成了别人；
版本没合并，图上看着"两篇论文"，其实是一篇的两版。所以这里的断言
大多盯着"谁应该排在谁前面""库里两版应该是同一个点"这种可验证的事实。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.citegraph import (
    MIN_SHARED_CITES,
    TITLE_MATCH,
    _layout,
    _paper_year,
    build_graph,
    pagerank,
)

ROOT = Path(__file__).resolve().parents[3]
PDF_V1 = ROOT / "pdf" / "1706.03762v1.pdf"
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"
PDF_GPT3 = ROOT / "pdf" / "2005.14165_GPT-3.pdf"
PDF_GPT4 = ROOT / "pdf" / "2303.08774_GPT-4.pdf"

needs_pdfs = pytest.mark.skipif(
    not (PDF_V1.exists() and PDF_V7.exists() and PDF_BERT.exists()),
    reason="样本 PDF 不在仓库里",
)


# ---------------------------------------------------------------- 假论文


@dataclass
class FakePaper:
    """只带图谱用得到的那点东西。真解析太慢，单元测试不该依赖 PDF。"""

    id: str
    title: str
    arxiv_id: str = ""
    created_at: str = ""
    uploaded_at: str = ""
    path: Path = field(default_factory=Path)
    chunks: list = field(default_factory=list)
    fingerprint: set = field(default_factory=set)


# ---------------------------------------------------------------- PageRank


def test_pagerank_points_toward_the_cited_node():
    """票必须流向"被引用的那一方"。

    方向写反过一次：代码把出边当入边用，结果引用别人最多的那篇排第一，
    而它自己一条引用都没收到。图和数字都在，错得毫无痕迹。
    """
    rank = pagerank(["A", "B", "C"], [("A", "B"), ("C", "B")])
    assert rank["B"] > rank["A"] and rank["B"] > rank["C"]


def test_pagerank_passes_authority_along_the_chain():
    """A 引 B、B 引 C：越往下游越值钱。这正是 PageRank 比"数被引次数"强的地方。"""
    rank = pagerank(list("ABC"), [("A", "B"), ("B", "C")])
    assert rank["C"] > rank["B"] > rank["A"]


def test_pagerank_rewards_authority_over_volume():
    """被枢纽引一次，胜过被两个无名小卒各引一次。

    X 的被引次数（2）比 Y（1）多，但引 X 的 L1、L2 自己没人引，
    引 Y 的 H 被三篇引 —— 票的含金量不同。实测 Y=0.276 > X=0.051。
    """
    ids = ["H", "L1", "L2", "X", "Y", "C1", "C2", "C3"]
    edges = [
        ("C1", "H"), ("C2", "H"), ("C3", "H"),   # H 是三篇的共同引用对象
        ("L1", "X"), ("L2", "X"),                 # 无名小卒引 X
        ("H", "Y"),                               # 枢纽引 Y
        ("X", "C1"), ("Y", "C2"),                 # 给 X、Y 找出边，免得它们当悬挂点
    ]
    rank = pagerank(ids, edges)
    assert sum(1 for s, t in edges if t == "X") > sum(1 for s, t in edges if t == "Y")
    assert rank["Y"] > rank["X"]
    assert rank["H"] == max(rank.values())


def test_pagerank_is_stable_and_normalized():
    edges = [("a", "b"), ("b", "c"), ("c", "a")]
    first = pagerank(["a", "b", "c"], edges)
    second = pagerank(["a", "b", "c"], edges)
    assert first == second
    assert sum(first.values()) == pytest.approx(1.0, abs=1e-6)


def test_pagerank_handles_empty_input():
    assert pagerank([], []) == {}


def test_paper_year_comes_from_the_arxiv_id():
    """不能拿 PDF 生成日期当年份。

    Transformer 的 v7 是 2024 年重新导出的，按生成日期会显示 "(2024)"，
    而它 2017 年就发表了 —— 图谱上写错年份，比不写更糟。
    """
    assert _paper_year(FakePaper(id="a", title="", arxiv_id="1706.03762", created_at="2024-03-11")) == "2017"
    assert _paper_year(FakePaper(id="b", title="", arxiv_id="1810.04805", created_at="2019-01-01")) == "2018"
    assert _paper_year(FakePaper(id="c", title="", arxiv_id="2303.08774", created_at="2024-01-01")) == "2023"


def test_paper_year_falls_back_to_the_pdf_date():
    assert _paper_year(FakePaper(id="d", title="", arxiv_id="", created_at="2019-05-02")) == "2019"
    assert _paper_year(FakePaper(id="e", title="", arxiv_id="", created_at="")) == ""


# ---------------------------------------------------------------- 布局


def test_layout_puts_the_top_node_in_the_centre():
    ranked = ["p", "a", "b", "c"]
    rank = {"p": 0.5, "a": 0.2, "b": 0.2, "c": 0.1}
    pos = _layout(ranked, rank)
    assert pos["p"] == (0.5, 0.5)
    for k in ("a", "b", "c"):
        assert 0.0 <= pos[k][0] <= 1.0 and 0.0 <= pos[k][1] <= 1.0


def test_layout_is_deterministic():
    """两次打开同一个库，节点位置必须一样。"""
    ranked = [f"n{i}" for i in range(12)]
    rank = {k: 1 / (i + 1) for i, k in enumerate(ranked)}
    assert _layout(ranked, rank) == _layout(ranked, rank)


# ---------------------------------------------------------------- 空库 / 阈值


def test_empty_library_produces_an_empty_graph():
    g = build_graph([])
    assert g.nodes == [] and g.edges == []
    assert g.stats == {}


def test_title_threshold_sits_above_the_near_miss_pair():
    """实测：真引用 1.00，最像的假引用 0.50（GPT-2 vs GPT-3 的标题）。

    0.50 那条是"Language models are unsupervised multitask learners"对
    "Language Models are Few-Shot Learners" —— 人会犹豫，图不能犹豫。
    """
    assert 0.5 < TITLE_MATCH < 1.0
    assert MIN_SHARED_CITES >= 2


# ---------------------------------------------------------------- 真 PDF


@needs_pdfs
def test_versions_of_one_paper_collapse_into_one_node():
    """v1 和 v7 是同一篇论文。不合并的话，PageRank 会把票数劈成两半。"""
    from app.api.registry import PaperRegistry
    from app.parsing.chunker import chunk_pdf
    from app.versioning import shingles

    papers = []
    for name, path in (("v1", PDF_V1), ("v7", PDF_V7)):
        chunks = chunk_pdf(str(path))
        papers.append(
            FakePaper(
                id=name,
                title=PaperRegistry._title(path, name),
                arxiv_id=PaperRegistry._arxiv_id(path, name),
                created_at=PaperRegistry._pdf_date(path),
                path=path,
                chunks=chunks,
                fingerprint=shingles(" ".join(c.text for c in chunks)),
            )
        )

    g = build_graph(papers, max_external=0)
    assert g.stats["clusters"] == 1
    assert g.stats["merges"] == 1
    assert len(g.nodes) == 1
    assert len(g.nodes[0].paper_ids) == 2
    # 代表必须是**新**的那版（按 PDF 生成日期）
    assert g.nodes[0].paper_ids[-1] == "v7"


@needs_pdfs
def test_known_citations_are_found_and_ranked():
    """样本库的正确答案是能背下来的：

    BERT 引 Transformer，GPT-3 引 Transformer 和 BERT，GPT-4 三个都引。
    于是 Transformer 被 3 篇引用，排第一；GPT-4 谁都不引用它，垫底。
    """
    from app.api.registry import PaperRegistry
    from app.parsing.chunker import chunk_pdf
    from app.versioning import shingles

    paths = {"v1": PDF_V1, "v7": PDF_V7, "bert": PDF_BERT}
    if PDF_GPT3.exists():
        paths["gpt3"] = PDF_GPT3
    if PDF_GPT4.exists():
        paths["gpt4"] = PDF_GPT4

    papers = []
    for name, path in paths.items():
        chunks = chunk_pdf(str(path))
        papers.append(
            FakePaper(
                id=name,
                title=PaperRegistry._title(path, path.name),
                arxiv_id=PaperRegistry._arxiv_id(path, path.name),
                created_at=PaperRegistry._pdf_date(path),
                path=path,
                chunks=chunks,
                fingerprint=shingles(" ".join(c.text for c in chunks)),
            )
        )

    g = build_graph(papers, max_external=0)
    labels = {n.id: n for n in g.nodes}
    top = g.nodes[0]
    assert top.kind == "paper"
    assert top.label.startswith("Attention Is All You Need")
    assert top.cited_by >= 2
    assert top.rank == max(n.rank for n in g.nodes)

    def edge_exists(src_key: str, dst_key: str) -> bool:
        src = next(n.id for n in g.nodes if f":{src_key}" in n.id)
        dst = next(n.id for n in g.nodes if f":{dst_key}" in n.id)
        return any(e.source == src and e.target == dst for e in g.edges)

    assert edge_exists("bert", "v7")
    if "gpt3" in paths:
        assert edge_exists("gpt3", "v7") and edge_exists("gpt3", "bert")
    if "gpt4" in paths:
        assert edge_exists("gpt4", "bert") and edge_exists("gpt4", "gpt3")
    assert len(labels) == g.stats["clusters"]


@needs_pdfs
def test_external_references_need_at_least_two_citers():
    """只被一篇引用的外部文献不上图 —— 全画出来是一团毛线。"""
    from app.api.registry import PaperRegistry
    from app.parsing.chunker import chunk_pdf
    from app.versioning import shingles

    papers = []
    for name, path in (("v7", PDF_V7), ("bert", PDF_BERT)):
        chunks = chunk_pdf(str(path))
        papers.append(
            FakePaper(
                id=name,
                title=PaperRegistry._title(path, path.name),
                arxiv_id=PaperRegistry._arxiv_id(path, path.name),
                created_at=PaperRegistry._pdf_date(path),
                path=path,
                chunks=chunks,
                fingerprint=shingles(" ".join(c.text for c in chunks)),
            )
        )

    g = build_graph(papers, max_external=8)
    external = [n for n in g.nodes if n.kind == "external"]
    assert external, "两篇论文的共同参考文献应该出现在图上"
    assert all(n.cited_by >= MIN_SHARED_CITES for n in external)
    assert all(n.rank == 0.0 for n in external), "外部文献不参与库内排名"
    # 被省略的部分要如实写出来，不能悄悄少画
    assert any("省略" in s for s in g.notes)


@needs_pdfs
def test_graph_is_deterministic():
    from app.api.registry import PaperRegistry
    from app.parsing.chunker import chunk_pdf
    from app.versioning import shingles

    path = PDF_BERT
    chunks = chunk_pdf(str(path))
    papers = [
        FakePaper(
            id="bert",
            title=PaperRegistry._title(path, path.name),
            arxiv_id=PaperRegistry._arxiv_id(path, path.name),
            path=path,
            chunks=chunks,
            fingerprint=shingles(" ".join(c.text for c in chunks)),
        )
    ]
    a = build_graph(papers)
    b = build_graph(papers)
    assert [(n.id, n.x, n.y, n.rank) for n in a.nodes] == [(n.id, n.x, n.y, n.rank) for n in b.nodes]
