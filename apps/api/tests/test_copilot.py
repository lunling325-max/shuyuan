"""写作 Copilot 接口的测试（全程离线）。

重点不是"框架写得好不好"，而是三条底线：
1. 模型编出来的证据编号一律丢弃；
2. 引用文本来自证据块原文，不是模型生成的；
3. 模型不可用仍返回 200，references 保留真实文献；库空才是 400。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"

IDEA = "用稀疏注意力降低长文档 RAG 的检索成本"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ.setdefault("PAPER_STORE_DIR", str(tmp_path_factory.mktemp("papers_copilot")))
    from app.api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def paper_ids(client) -> list[str]:
    ids = []
    for path in (PDF_V7, PDF_BERT):
        r = client.post(
            "/api/papers",
            files={"file": (path.name, path.read_bytes(), "application/pdf")},
        )
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    return ids


@pytest.fixture(scope="module")
def library_ids(paper_ids) -> set[str]:
    """共享 registry 里可能还有别的测试传进来的版本（如 v1），引用允许落在任何真实论文上。"""
    import app.api.main as main

    return {p.id for p in main.registry.all()}


def _fake_llm(queries: list[str], outline: object):
    def call(user, *, system=None, **kwargs):
        if system and "search queries" in system:
            return {"queries": queries}
        return {"outline": outline}

    return call


def test_keeps_only_real_evidence_numbers_and_builds_quotes(
    client, library_ids, monkeypatch
):
    outline = [
        {
            "title": "1 引言",
            "points": [
                {"text": "稀疏注意力能降低长文档 RAG 的检索成本", "evidence": [1, 999]},
                {"text": "需要报告压缩率与准确率的权衡", "evidence": []},
            ],
        },
        {
            "title": "2 方法",
            "points": [{"text": "设计稀疏注意力掩码", "evidence": [1, 1]}],
        },
    ]
    monkeypatch.setattr(
        "app.llm.client.chat_json",
        _fake_llm(["scaled dot product attention", "sparse attention"], outline),
    )

    r = client.post("/api/copilot/outline", json={"idea": IDEA})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["idea"] == IDEA
    assert body["degraded"] is False

    # 999 不存在 —— 丢掉；1 存在 —— 保留；重复编号去重
    assert body["outline"][0]["points"][0]["evidence"] == [1]
    assert body["outline"][0]["points"][1]["evidence"] == []
    assert body["outline"][1]["points"][0]["evidence"] == [1]

    references = body["references"]
    assert [ref["n"] for ref in references] == [1], "references 按编号去重"
    ref = references[0]
    assert ref["paper_id"] in library_ids
    assert ref["title"]
    assert ref["page"] >= 1
    assert 0 < len(ref["quote"]) <= 200

    # 引用必须是真实原文前缀，而不是模型写的句子
    chunk_texts = client.get(f"/api/papers/{ref['paper_id']}/chunks").json()
    assert any(ref["quote"] in c["text"] for c in chunk_texts)


def test_invented_only_numbers_yield_empty_evidence(client, paper_ids, monkeypatch):
    outline = [
        {"title": "1 引言", "points": [{"text": "完全没有依据的要点", "evidence": [42, 999]}]}
    ]
    monkeypatch.setattr(
        "app.llm.client.chat_json",
        _fake_llm(["scaled dot product attention"], outline),
    )

    body = client.post("/api/copilot/outline", json={"idea": IDEA}).json()

    assert body["outline"][0]["points"][0]["evidence"] == []
    assert body["references"] == []


def test_max_points_caps_points_per_section(client, paper_ids, monkeypatch):
    outline = [
        {
            "title": "1 引言",
            "points": [{"text": f"要点 {i}", "evidence": [1]} for i in range(6)],
        }
    ]
    monkeypatch.setattr(
        "app.llm.client.chat_json",
        _fake_llm(["scaled dot product attention"], outline),
    )

    body = client.post(
        "/api/copilot/outline", json={"idea": IDEA, "max_points": 2}
    ).json()

    assert len(body["outline"][0]["points"]) == 2


def test_llm_outage_degrades_but_returns_real_references(client, library_ids, monkeypatch):
    from app.llm.client import LLMError

    def down(*args, **kwargs):
        raise LLMError("缺少环境变量 DEEPSEEK_API_KEY")

    monkeypatch.setattr("app.llm.client.chat_json", down)

    r = client.post("/api/copilot/outline", json={"idea": IDEA})

    assert r.status_code == 200, "模型不可用不能 500"
    body = r.json()
    assert body["degraded"] is True
    assert body["outline"] == []
    assert body["references"], "断网时仍要能展示库里可引用的原文"
    for ref in body["references"]:
        assert ref["paper_id"] in library_ids
        assert ref["title"]
        assert ref["page"] >= 1
        assert 0 < len(ref["quote"]) <= 200


def test_retrieval_covers_the_whole_library():
    """不是只查第一篇：两篇论文的证据都要能进来。"""
    import app.api.main as main
    from app.rag.copilot import collect_evidence

    evidence = collect_evidence(
        main.registry.all(),
        "attention models",
        limit=16,
        per_paper=2,
        llm_call=_fake_llm(["attention", "bert"], []),
    )
    titles = {e.title.lower() for e in evidence}
    assert any("bert" in t for t in titles), titles
    assert any("attention" in t for t in titles), titles


def test_empty_library_returns_400(client, monkeypatch, tmp_path):
    import app.api.main as main
    from app.api.registry import PaperRegistry

    monkeypatch.setattr(main, "registry", PaperRegistry(tmp_path))

    r = client.post("/api/copilot/outline", json={"idea": IDEA})

    assert r.status_code == 400
    assert "上传" in r.json()["detail"]
