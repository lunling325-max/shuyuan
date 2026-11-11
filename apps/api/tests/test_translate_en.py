"""英译中（划词翻译）接口的测试（全程离线）。

守住五条底线：
1. 正常翻译：返回原文 + 译文，degraded=false；
2. 译文丢掉 LaTeX/引用编号：仍然返回译文，但 degraded=true 且 notes 写明丢了什么；
3. 模型不可用（抛异常）→ 200 + degraded，原文照样返回，绝不 500；
4. 空输入 → 400；
5. 超长输入 → 413（划词的上限比中译英小）。

和 test_translate.py 一样：裸 `FastAPI()` + `include_router`，不碰 main.py。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.rag.translate_en import MAX_INPUT_CHARS


@pytest.fixture()
def client():
    from app.api.routers.translate import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_translates_a_selected_snippet(client, monkeypatch):
    seen: list[str] = []

    def fake(user, *, system=None, **kwargs):
        seen.append(user)
        return {"zh": "我们提出 Transformer，完全基于注意力机制。"}

    monkeypatch.setattr("app.llm.client.chat_json", fake)

    src = "We propose the Transformer, based entirely on attention mechanisms."
    r = client.post("/api/translate/en-zh", json={"text": src})
    assert r.status_code == 200
    body = r.json()
    assert body["en"] == src
    assert body["zh"].startswith("我们提出 Transformer")
    assert body["degraded"] is False
    assert body["notes"] == []
    # 原文真的进了 prompt：划词翻译不能只发"翻译一下"
    assert src in seen[0]


def test_missing_formula_is_flagged_but_translation_kept(client, monkeypatch):
    src = "The scaled dot-product attention divides by $\\sqrt{d_k}$ as in [12]."

    # 模型把公式和引用编号都吞了
    monkeypatch.setattr(
        "app.llm.client.chat_json",
        lambda user, **kw: {"zh": "缩放点积注意力按论文所述做了归一化。"},
    )

    body = client.post("/api/translate/en-zh", json={"text": src}).json()
    assert body["degraded"] is True
    assert body["zh"]  # 有译文就照样给，不能因为丢了片段就什么都不给
    note = " ".join(body["notes"])
    assert "丢失" in note
    assert "$\\sqrt{d_k}$" in note or "[12]" in note


def test_model_failure_degrades_instead_of_500(client, monkeypatch):
    def boom(user, **kwargs):
        raise RuntimeError("DEEPSEEK_API_KEY 没配")

    monkeypatch.setattr("app.llm.client.chat_json", boom)

    src = "Multi-head attention allows the model to jointly attend to information."
    r = client.post("/api/translate/en-zh", json={"text": src})
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["zh"] == ""
    assert body["en"] == src  # 原文永远不丢
    assert body["notes"]


def test_empty_input_is_400(client):
    assert client.post("/api/translate/en-zh", json={"text": "   "}).status_code == 400


def test_too_long_input_is_413(client):
    body = {"text": "attention " * (MAX_INPUT_CHARS // 5)}
    assert client.post("/api/translate/en-zh", json=body).status_code == 413
