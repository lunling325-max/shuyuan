"""中英对照接口的测试（全程离线）。

守住四条：
1. 段落按空行切、译文与原文一一配对；
2. 模型返回段数不符时只配对 min 长度，缺失译文留空并降级；
3. 模型/网络不可用时返回 200 + degraded，不能 500；
4. 404 只表示"论文/块不存在"，不能和降级混为一谈。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    # app.api.main 可能已被别的测试模块导入；setdefault 不覆盖已有配置
    os.environ.setdefault("PAPER_STORE_DIR", str(tmp_path_factory.mktemp("papers_bilingual")))
    from app.api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def paper_id(client) -> str:
    data = PDF_V7.read_bytes()
    r = client.post("/api/papers", files={"file": (PDF_V7.name, data, "application/pdf")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.fixture(scope="module")
def chunks(client, paper_id) -> list[dict]:
    return client.get(f"/api/papers/{paper_id}/chunks").json()


@pytest.fixture
def target(chunks) -> dict:
    """选一个 2~10 段的块：既能测配对，又不会让假模型返回上百条译文。"""
    from app.rag.bilingual import split_paragraphs

    for chunk in chunks:
        if 2 <= len(split_paragraphs(chunk["text"])) <= 10:
            return chunk
    pytest.fail("样本里找不到多段落块，无法测试逐段配对")


@pytest.fixture(autouse=True)
def _clear_cache():
    from app.rag import bilingual

    bilingual.clear_cache()
    yield
    bilingual.clear_cache()


def _url(paper_id: str, chunk: int) -> str:
    return f"/api/papers/{paper_id}/bilingual?chunk={chunk}"


def test_translates_paragraph_by_paragraph(client, paper_id, target, monkeypatch):
    from app.rag.bilingual import split_paragraphs

    paragraphs = split_paragraphs(target["text"])
    zh = [f"第 {i} 段译文" for i in range(1, len(paragraphs) + 1)]
    seen: dict[str, str] = {}

    def fake(user, *, system=None, **kwargs):
        seen["user"] = user
        seen["system"] = system or ""
        return {"zh": zh}

    monkeypatch.setattr("app.llm.client.chat_json", fake)
    r = client.get(_url(paper_id, target["index"]))

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunk"] == target["index"]
    assert body["section"] == target["section"]
    assert body["page_start"] == target["page_start"]
    assert body["page_end"] == target["page_end"]
    assert [p["en"] for p in body["paragraphs"]] == paragraphs
    assert [p["zh"] for p in body["paragraphs"]] == zh
    assert body["degraded"] is False
    assert body["model"]
    # 提示词必须把段数说清楚，并交代公式/变量名保留原文
    assert str(len(paragraphs)) in seen["user"]
    assert "不许合并" in seen["system"]
    assert "softmax" in seen["system"] and "d_model" in seen["system"]


def test_fewer_translations_pair_up_to_min_and_degrade(client, paper_id, target, monkeypatch):
    from app.rag.bilingual import split_paragraphs

    paragraphs = split_paragraphs(target["text"])
    assert len(paragraphs) >= 2

    def fake(user, *, system=None, **kwargs):
        return {"zh": ["只翻了一段"]}

    monkeypatch.setattr("app.llm.client.chat_json", fake)
    body = client.get(_url(paper_id, target["index"])).json()

    assert [p["en"] for p in body["paragraphs"]] == paragraphs
    assert body["paragraphs"][0]["zh"] == "只翻了一段"
    assert all(p["zh"] == "" for p in body["paragraphs"][1:]), "不许硬塞假译文"
    assert body["degraded"] is True


def test_more_translations_are_truncated_to_input_and_degrade(client, paper_id, target, monkeypatch):
    from app.rag.bilingual import split_paragraphs

    paragraphs = split_paragraphs(target["text"])

    def fake(user, *, system=None, **kwargs):
        return {"zh": [f"译文{i}" for i in range(len(paragraphs) + 3)]}

    monkeypatch.setattr("app.llm.client.chat_json", fake)
    body = client.get(_url(paper_id, target["index"])).json()

    assert len(body["paragraphs"]) == len(paragraphs)
    assert [p["zh"] for p in body["paragraphs"]] == [f"译文{i}" for i in range(len(paragraphs))]
    assert body["degraded"] is True, "多出来的译文说明模型没有按段数输出"


def test_llm_outage_returns_200_with_original_text(client, paper_id, target, monkeypatch):
    from app.llm.client import LLMError
    from app.rag.bilingual import split_paragraphs

    def down(*args, **kwargs):
        raise LLMError("缺少环境变量 DEEPSEEK_API_KEY")

    monkeypatch.setattr("app.llm.client.chat_json", down)
    r = client.get(_url(paper_id, target["index"]))

    assert r.status_code == 200, "断网不能让中英对照 500"
    body = r.json()
    assert body["degraded"] is True
    assert body["model"]
    assert [p["en"] for p in body["paragraphs"]] == split_paragraphs(target["text"])
    assert all(p["zh"] == "" for p in body["paragraphs"])


def test_successful_translation_is_cached_per_chunk(client, paper_id, target, monkeypatch):
    from app.rag.bilingual import split_paragraphs

    paragraphs = split_paragraphs(target["text"])
    calls = {"n": 0}

    def fake(user, *, system=None, **kwargs):
        calls["n"] += 1
        return {"zh": ["缓存的译文"] * len(paragraphs)}

    monkeypatch.setattr("app.llm.client.chat_json", fake)
    first = client.get(_url(paper_id, target["index"]))
    second = client.get(_url(paper_id, target["index"]))

    assert first.status_code == second.status_code == 200
    assert calls["n"] == 1, "同一个块的重复请求不该再调模型"
    assert first.json() == second.json()


def test_unknown_paper_is_404(client):
    assert client.get(_url("deadbeef", 0)).status_code == 404


def test_chunk_out_of_range_is_404(client, paper_id, chunks):
    assert client.get(_url(paper_id, len(chunks))).status_code == 404
    assert client.get(_url(paper_id, -1)).status_code == 404
