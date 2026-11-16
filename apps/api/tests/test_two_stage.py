"""两级检索（章节粗筛 → 块级检索）的边界行为。

真正要守住的三件事：
1. 小文档**不走**两级检索 —— 25 块的论文上粗筛只有开销没有收益，还会担召回风险；
2. 粗筛结果不够 k 条时**整条退回全局**，宁可慢也不能少给证据；
3. `restrict=空集合` 必须返回空，调用方据此判断"粗筛没筛出东西"，而不是拿到一堆无关块。
"""

from __future__ import annotations

import pytest

import app.rag.query as Q
from app.parsing.chunker import Chunk
from app.rag.query import retrieve
from app.retrieval.bm25 import BM25
from app.retrieval.sections import build as build_sections


def _chunk(i: int, text: str, section: str) -> Chunk:
    return Chunk(index=i, text=text, section=section, page_start=i + 1, page_end=i + 1,
                 n_tokens=len(text.split()), rects=())


def _big_paper(n_sections: int = 60, per_section: int = 4) -> list[Chunk]:
    """造一份"长文档"：每节一个主题词，主题词只在该节出现。"""
    chunks = []
    for s in range(n_sections):
        name = f"Chapter {s} topic{s}"
        for j in range(per_section):
            chunks.append(_chunk(len(chunks), f"paragraph {j} about topic{s} only", name))
    return chunks


def test_empty_restrict_returns_nothing():
    """空集合表示"没有候选"，不是"不做限制" —— 两者混淆会让粗筛失效得无声无息。"""
    bm = BM25(_big_paper(2, 2))
    assert bm.search("topic1", k=3, restrict=set()) == []
    assert bm.search("topic1", k=3, restrict=None) != []


def test_restrict_limits_scope():
    bm = BM25(_big_paper(3, 2))
    # 每节 2 块：topic1 落在序号 2、3 上
    hits = bm.search("topic1", k=5, restrict={2, 3})
    assert hits and {h.chunk.index for h in hits} <= {2, 3}, [h.chunk.index for h in hits]
    assert bm.search("topic1", k=5, restrict={0, 1}) == []


def test_sections_group_and_rank():
    chunks = _big_paper(5, 3)
    idx = build_sections(chunks)
    assert len(idx) == 5
    top = idx.score("topic3")
    assert top and top[0].name.endswith("topic3"), [h.name for h in top[:2]]
    # 候选必须落在该节的全部块上
    assert set(idx.candidates(["topic3"])) == set(top[0].positions)


def test_small_document_skips_two_stage():
    """25 块的论文不建章节索引：这是硬边界，防止有人把阈值调低后召回悄悄变差。"""
    bm = BM25(_big_paper(2, 2))
    assert Q._section_index(bm) is None


def test_two_stage_keeps_top_hit_on_long_document():
    """两级检索的首条命中必须和全局一致 —— 收窄范围不能改变答案。"""
    chunks = _big_paper(60, 4)
    bm = BM25(chunks)
    queries = ["topic42"]
    two = retrieve("x", retriever=bm, queries=queries, k=3)
    saved = Q.MIN_CHUNKS_FOR_SECTIONS
    try:
        Q.MIN_CHUNKS_FOR_SECTIONS = 10 ** 9
        BM25(chunks)                      # 新实例，避免复用已缓存的章节索引
        flat = retrieve("x", retriever=BM25(chunks), queries=queries, k=3)
    finally:
        Q.MIN_CHUNKS_FOR_SECTIONS = saved
    assert two and flat
    assert two[0].chunk.index == flat[0].chunk.index, (two[0].chunk.index, flat[0].chunk.index)


def test_two_stage_falls_back_when_candidates_are_thin():
    """粗筛只捞到 1 块、而用户要 6 块时必须退回全局 —— 少给证据比慢更糟。"""
    chunks = _big_paper(60, 4)
    bm = BM25(chunks)
    # 只查 topic42 时全局也只有 4 块命中，看不出回退；加一个到处都有的词才有对比
    hits = retrieve("x", retriever=bm, queries=["topic42 paragraph"], k=6)
    assert len(hits) == 6, [h.chunk.index for h in hits]
