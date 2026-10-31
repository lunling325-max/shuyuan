"""检索词扩展与多路召回的测试（离线）。

最该守住的一条：**扩展失败不能拖垮问答**。
扩展是增强项，模型超时 / 返回垃圾 / 网络断了，都必须安静地退回原始问题。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.chunker import chunk_pdf
from app.rag.query import expand_query, retrieve
from app.retrieval import BM25

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"


@pytest.fixture(scope="module")
def bm25():
    return BM25(chunk_pdf(str(PDF_V7)))


def test_expand_returns_queries():
    def fake(user, *, system=None, **_):
        return {"queries": ["scaled dot-product attention", "sqrt dk scaling"]}

    assert expand_query("为什么要除以 sqrt(dk)？", llm_call=fake) == [
        "scaled dot-product attention",
        "sqrt dk scaling",
    ]


def test_expand_falls_back_on_error():
    def boom(*_a, **_k):
        raise RuntimeError("network down")

    assert expand_query("为什么要除以 sqrt(dk)？", llm_call=boom) == ["为什么要除以 sqrt(dk)？"]


def test_expand_falls_back_on_empty_payload():
    def empty(*_a, **_k):
        return {"queries": []}

    assert expand_query("任意问题", llm_call=empty) == ["任意问题"]


def test_retrieve_fuses_and_dedups(bm25):
    def fake(user, *, system=None, **_):
        return {"queries": ["scaled dot product attention", "divide by sqrt dk"]}

    hits = retrieve("为什么要除以 sqrt(dk)？", retriever=bm25, k=5, llm_call=fake)
    assert hits
    assert len({h.chunk.index for h in hits}) == len(hits)
    assert "3.2.1" in hits[0].chunk.section


def test_retrieve_without_expansion_still_works(bm25):
    hits = retrieve("scaled dot product attention", retriever=bm25, k=3, expand=False)
    assert hits and "3.2.1" in hits[0].chunk.section


def test_retrieve_returns_empty_when_nothing_matches(bm25):
    assert retrieve("zzzqqq xylophone", retriever=bm25, k=3, expand=False) == []
