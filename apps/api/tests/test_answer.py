"""问答链路的测试（离线，不调网络）。

关键是证明：模型编出来的那条会被**摘掉**，而不是被"温和改写"后放行。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.chunker import chunk_pdf
from app.parsing.text_index import TextIndex
from app.rag import answer_question, build_evidence
from app.retrieval import BM25

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"

REAL_QUOTE = (
    "We suspect that for large values of dk, the dot products grow large in magnitude"
)
FAKE_QUOTE = "Our model achieves 90.2 percent top-1 accuracy on ImageNet classification"


@pytest.fixture(scope="module")
def env():
    chunks = chunk_pdf(str(PDF_V7))
    return BM25(chunks), TextIndex(str(PDF_V7))


def _fake_llm(payload: dict):
    def call(user: str, *, system: str | None = None, **_) -> dict:
        return payload

    return call


def test_real_quote_is_kept_with_anchor(env):
    bm25, index = env
    payload = {"claims": [{"text": "除以 sqrt(dk) 是为了抵消点积变大", "quotes": [REAL_QUOTE]}]}
    ans = answer_question(
        "为什么要除以 sqrt(dk)？", retriever=bm25, index=index, llm_call=_fake_llm(payload)
    )
    assert len(ans.kept) == 1
    anchor = ans.kept[0].anchors[0]
    assert anchor.exact
    assert anchor.page == 4
    assert "p.4" in ans.render()


def test_fabricated_quote_is_dropped(env):
    bm25, index = env
    payload = {
        "claims": [
            {"text": "真实结论", "quotes": [REAL_QUOTE]},
            {"text": "编造结论：ImageNet 90.2%", "quotes": [FAKE_QUOTE]},
        ]
    }
    ans = answer_question("why divide by square root of dk", retriever=bm25, index=index, llm_call=_fake_llm(payload))
    assert [c.claim.text for c in ans.kept] == ["真实结论"]
    assert [c.claim.text for c in ans.dropped] == ["编造结论：ImageNet 90.2%"]
    assert "编造" not in ans.render()


def test_claim_without_quote_is_dropped(env):
    bm25, index = env
    payload = {"claims": [{"text": "没有证据的断言", "quotes": []}]}
    ans = answer_question("why divide by square root of dk", retriever=bm25, index=index, llm_call=_fake_llm(payload))
    assert ans.kept == []
    assert "证据不足" in ans.render()


def test_no_evidence_short_circuits(env):
    bm25, index = env

    def boom(*_a, **_k):
        raise AssertionError("没有证据时不应该调用 LLM")

    ans = answer_question(
        "zzzqqq xylophone", retriever=bm25, index=index, llm_call=boom
    )
    assert ans.evidence == []
    assert ans.claims == []


def test_evidence_is_not_truncated(env):
    """E-002 的教训：截断证据会逼模型去猜。"""
    bm25, _ = env
    hits = bm25.search("why divide by square root of dk", k=3)
    text = build_evidence(hits)
    assert hits[0].chunk.text in text
    assert len(text) > 2000
