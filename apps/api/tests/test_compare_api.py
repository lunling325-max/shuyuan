"""跨文献对比的测试（全程离线，检索与校验真跑）。

守住的四件事：
1. 证据块按论文分组编号（A#1 / B#1），模型看得到"这段话出自哪一篇"；
2. 模型给的引用必须自带 paper 标签：没标签的一律丢掉，不猜；
3. 校验按各自的原文进行 —— 一条结论同时引用 A、B，两边的锚点都要落地，
   且各自带着出自哪一篇；
4. 没配 key → 200 + degraded，各篇召回的原文照样返回，界面不白屏。

样本用赛题里那对最常被拿来对比的论文：Transformer（v7）与 BERT。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.rag.compare import build_evidence, parse_claims

ROOT = Path(__file__).resolve().parents[3]
PDF_A = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_B = ROOT / "pdf" / "1810.04805_BERT.pdf"


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    from app.api.registry import PaperRegistry

    reg = PaperRegistry(tmp_path_factory.mktemp("papers_cross"))
    reg.add(PDF_A.read_bytes(), PDF_A.name)
    reg.add(PDF_B.read_bytes(), PDF_B.name)
    return reg


@pytest.fixture()
def client(registry, monkeypatch):
    from app.api import deps
    from app.api.routers.cross import router

    monkeypatch.setattr(deps, "registry", registry)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _quote_from(paper, limit: int = 140) -> str:
    """从这篇论文的正文块里取一段，当作模型"逐字引用"的原文。

    **必须先用 find() 自检**：首页那块（标题 + 作者 + 邮箱）在词流里是散开的，
    拿它当前缀永远匹配不上，测试会假红。比硬编码一句安全：换一份 PDF 也不会
    因为句子不同而失败。
    """
    index = paper.text_index
    for chunk in paper.chunks:
        text = chunk.text.strip()
        if len(text) >= 120 and index.find(text[:limit]) is not None:
            return text[:limit]
    raise AssertionError("这份 PDF 没有可用的正文块")


def _fake_llm(payload: dict):
    """同时应付检索词扩展与结论生成两次调用。"""

    def call(user, *, system=None, **kwargs):
        if system and "search queries" in system:
            return {"queries": ["attention mechanism"]}
        return payload

    return call


# ---------------------------------------------------------------- 纯函数


def test_evidence_is_grouped_by_paper():
    class _Chunk:
        index = 0
        page_start = 1
        page_end = 1
        section = "Introduction"
        text = "We propose something."

    class _Hit:
        chunk = _Chunk()
        score = 1.0

    text = build_evidence({"A": [_Hit()], "B": [_Hit()]}, {"A": "Paper A", "B": "Paper B"})
    assert "论文 A：Paper A" in text
    assert "论文 B：Paper B" in text
    assert "[A#1] p.1 · Introduction" in text
    assert "[B#1] p.1 · Introduction" in text


def test_quotes_without_paper_label_are_dropped():
    raw = {
        "claims": [
            {
                "text": "有标签的结论",
                "quotes": [
                    {"paper": "a", "quote": "attention is all you need"},
                    {"paper": "B", "quote": "bert is bidirectional"},
                    {"quote": "没有标签的引用"},
                    "一句话也不是对象",
                ],
            }
        ]
    }
    claims = parse_claims(raw, {"A": "id-a", "B": "id-b"})
    assert len(claims) == 1
    assert [q.doc for q in claims[0].quotes] == ["id-a", "id-b"]


# ---------------------------------------------------------------- 接口


def test_compare_anchors_both_papers(client, registry, monkeypatch):
    papers = registry.all()
    paper_a = next(p for p in papers if p.filename == PDF_A.name)
    paper_b = next(p for p in papers if p.filename == PDF_B.name)

    payload = {
        "claims": [
            {
                "text": "两篇都建立在注意力机制之上，但 BERT 是双向编码器。",
                "quotes": [
                    {"paper": "A", "quote": _quote_from(paper_a)},
                    {"paper": "B", "quote": _quote_from(paper_b)},
                ],
            }
        ]
    }
    monkeypatch.setattr("app.llm.client.chat_json", _fake_llm(payload))

    r = client.post(
        "/api/papers/compare",
        json={
            "paper_ids": [paper_a.id, paper_b.id],
            "question": "这两篇的架构有什么不同？",
            "k": 2,
        },
    )
    assert r.status_code == 200
    body = r.json()

    assert [p["label"] for p in body["papers"]] == ["A", "B"]
    assert body["degraded"] is False, body["notes"]
    assert set(body["evidence"]) == {"A", "B"}
    assert body["evidence"]["A"] and body["evidence"]["B"]

    claim = body["claims"][0]
    assert claim["status"] == "supported"
    assert {a["paper"] for a in claim["anchors"]} == {"A", "B"}
    # 每条锚点都能定位到页码与坐标，这才叫"可核对"
    for anchor in claim["anchors"]:
        assert anchor["page"] >= 1
        assert len(anchor["bbox"]) == 4


def test_fabricated_quotes_are_blocked(client, registry, monkeypatch):
    papers = registry.all()
    payload = {
        "claims": [
            {
                "text": "两篇论文都获得了图灵奖。",
                "quotes": [
                    {"paper": "A", "quote": "Both papers were awarded the Turing Award in 2020."},
                ],
            }
        ]
    }
    monkeypatch.setattr("app.llm.client.chat_json", _fake_llm(payload))

    body = client.post(
        "/api/papers/compare",
        json={"paper_ids": [p.id for p in papers], "question": "这两篇拿了什么奖？"},
    ).json()

    claim = body["claims"][0]
    assert claim["status"] == "unsupported"
    assert claim["anchors"] == []
    assert body["stats"]["unsupported"] == 1


def test_missing_key_degrades_with_evidence(client, registry, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    papers = registry.all()

    body = client.post(
        "/api/papers/compare",
        json={"paper_ids": [p.id for p in papers], "question": "架构差异在哪？"},
    ).json()

    assert body["degraded"] is True
    assert body["claims"] == []
    assert body["evidence"]["A"] and body["evidence"]["B"]
    assert "DEEPSEEK_API_KEY" in " ".join(body["notes"])


def test_same_paper_twice_is_400(client, registry):
    pid = registry.all()[0].id
    r = client.post(
        "/api/papers/compare",
        json={"paper_ids": [pid, pid], "question": "对比它自己？"},
    )
    assert r.status_code == 400
