"""检索层的测试。

用真问题考，不用单词考 —— "attention" 这种词满篇都是，命中不算本事。
真正该考的是：**问一句话，第一名的块是不是人也会选的那块。**
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.chunker import chunk_pdf
from app.retrieval import BM25, Hit, reciprocal_rank_fusion
from app.retrieval.bm25 import stem

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"


@pytest.fixture(scope="module")
def bm25():
    return BM25(chunk_pdf(str(PDF_V7)))


@pytest.mark.parametrize(
    "query, expected_section",
    [
        ("why divide by square root of dk", "3.2.1"),
        ("positional encoding sinusoidal", "3.5"),
        ("label smoothing epsilon", "5.4"),
        ("BLEU WMT 2014 English to German", "6.1"),
    ],
)
def test_top_hit_lands_in_the_right_section(bm25, query, expected_section):
    hits = bm25.search(query, k=3)
    assert hits, f"{query!r} 一条都没召回"
    assert expected_section in hits[0].chunk.section, (
        f"{query!r} 的第一名落在 {hits[0].chunk.section!r}"
    )


def test_multihead_heads_query_is_a_genuine_near_tie(bm25):
    """这条**不是**第一名，而是实打实的近似平局，所以只断言进前三。

    "multi-head attention heads" 把 3.2 Attention(6.3297) 和
    3.2.2 Multi-Head Attention(6.2874) 拉开 0.7% —— 名次是抛硬币的结果。
    这不是词干化引入的缺陷（词干化整体把 Hit@1 从 66.7% 提到 83.3%），
    而是短块上 BM25 的固有性质：只要查询词在引言里也出现，短块靠长度归一化就赢了。

    把它断言成"必须第一"等于把一个靠不住的信号写进测试，
    以后任何无关改动都会让它随机变红 —— 那种测试只会被人删掉。
    """
    hits = bm25.search("multi-head attention heads", k=3)
    assert any("3.2.2" in h.chunk.section for h in hits), [h.chunk.section for h in hits]


def test_decisive_queries_are_not_near_ties(bm25):
    """反过来要守住：真正有区分度的查询，第一名必须明显领先。

    如果这条也退化成平局，说明词干化把词形信息磨掉了 —— 那时该回退。
    """
    hits = bm25.search("label smoothing epsilon", k=2)
    # 只召回一条也是强的确定性（区分度最高的情形），不该被当成退化
    assert len(hits) == 1 or hits[0].score > hits[1].score * 1.5, [h.score for h in hits[:2]]


def test_empty_query_returns_nothing(bm25):
    assert bm25.search("   ", k=5) == []


def test_unknown_term_returns_nothing(bm25):
    """库里没有的词，宁可返回空也不能硬凑一条。"""
    assert bm25.search("zzzqqq xylophone", k=5) == []


def test_scores_are_monotonic(bm25):
    hits = bm25.search("attention", k=5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_is_deterministic(bm25):
    a = [(h.chunk.index, h.score) for h in bm25.search("attention mechanism", k=5)]
    b = [(h.chunk.index, h.score) for h in bm25.search("attention mechanism", k=5)]
    assert a == b


def test_rrf_prefers_document_ranked_high_by_both(bm25):
    """两路都排第一的块，融合后必须还是第一。"""
    chunks = bm25.chunks
    lexical = bm25.search("scaled dot product attention", k=5)
    # 伪造一路"向量"结果：把 BM25 的排名整体后移一位，制造分歧
    dense = [
        Hit(chunk=chunks[0], score=0.99, source="dense"),
        *[Hit(chunk=h.chunk, score=0.9, source="dense") for h in lexical[:4]],
    ]
    fused = reciprocal_rank_fusion([lexical, dense], limit=5)
    assert fused[0].chunk.index == lexical[0].chunk.index
    assert fused[0].source == "rrf"


def test_rrf_keeps_unique_documents(bm25):
    lexical = bm25.search("attention", k=5)
    fused = reciprocal_rank_fusion([lexical, lexical], limit=10)
    assert len({h.chunk.index for h in fused}) == len(fused)


# ---------------------------------------------------------------- 单复数归并

@pytest.mark.parametrize(
    "plural, singular",
    [
        ("visualizations", "visualization"),
        ("heads", "head"),
        ("layers", "layer"),
        ("processes", "process"),
        ("cases", "case"),
        ("classes", "class"),
        ("studies", "study"),
    ],
)
def test_stem_maps_plural_to_singular(plural, singular):
    assert stem(plural) == stem(singular)


@pytest.mark.parametrize("word", ["analysis", "various", "class", "attention", "this"])
def test_stem_leaves_non_plurals_alone(word):
    """ss / us / is 结尾不是复数，动了就是制造假匹配。"""
    assert stem(word) == word


def test_stem_is_idempotent():
    """词干化必须幂等，否则同一词在索引侧和查询侧会落到不同形态。"""
    for word in ["heads", "visualizations", "studies", "processes"]:
        assert stem(stem(word)) == stem(word)


def test_plural_query_finds_plural_title(bm25):
    """回归测试：问 "attention visualization"，必须能碰到标题为
    "Attention Visualizations" 的那一节。

    这条曾经整个失手 —— 词法通道对单复数完全失明，
    导致唯一正确的块连候选都进不去，只能眼睁睁看着别的节拿第一。
    """
    hits = bm25.search("attention visualization", k=6)
    sections = [h.chunk.section for h in hits]
    assert any("Attention Visualizations" in s for s in sections), sections
