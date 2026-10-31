"""中译英接口的测试（全程离线）。

守住五条底线：
1. 按段落顺序逐段翻译，段落数、顺序与原文一致；
2. 译文丢掉 LaTeX/引用标记时，该段退回原文并写进 notes；
3. 模型不可用 → 200 + degraded，绝不 500；
4. 空输入 → 400；
5. 超长输入 → 413。

测试用裸 `FastAPI()` + `include_router`，不依赖 main.py 里的 registry 与静态托管。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.rag.translate_zh import MAX_INPUT_CHARS, split_paragraphs


@pytest.fixture()
def client():
    """只装翻译路由：这个模块不依赖 registry，隔离得越干净越好。"""
    from app.api.routers.translate import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_translates_paragraph_by_paragraph_in_order(client, monkeypatch):
    text = "第一段介绍研究背景。\n\n第二段描述方法。\n\n第三段报告结果。"
    calls: list[tuple[str, str]] = []

    def fake(user, *, system=None, **kwargs):
        calls.append((user, system or ""))
        return {"en": f"Paragraph {len(calls)}."}

    monkeypatch.setattr("app.llm.client.chat_json", fake)

    r = client.post("/api/translate/zh-en", json={"text": text})

    assert r.status_code == 200, r.text
    body = r.json()
    assert [p["zh"] for p in body["paragraphs"]] == split_paragraphs(text)
    assert [p["en"] for p in body["paragraphs"]] == [
        "Paragraph 1.",
        "Paragraph 2.",
        "Paragraph 3.",
    ]
    assert body["degraded"] is False
    assert body["notes"] == []
    assert body["model"]
    assert len(calls) == 3, "必须一段一次调用，顺序翻译"
    assert "第三段报告结果" in calls[2][0], "每段只带自己的原文"
    assert "We propose" in calls[0][1], "system prompt 必须要求学术摘要体"
    assert "JSON" in calls[0][1], "必须约定只输出 JSON"


def test_single_newlines_are_used_when_no_blank_line(client, monkeypatch):
    """从 PDF 复制出来的摘要常是单换行分隔，也要按段落翻，而不是揉成一段。"""
    text = "第一行中文。\n第二行中文。"
    count = {"n": 0}

    def fake(user, *, system=None, **kwargs):
        count["n"] += 1
        return {"en": f"Line {count['n']}."}

    monkeypatch.setattr("app.llm.client.chat_json", fake)

    body = client.post("/api/translate/zh-en", json={"text": text}).json()

    assert [p["zh"] for p in body["paragraphs"]] == ["第一行中文。", "第二行中文。"]
    assert [p["en"] for p in body["paragraphs"]] == ["Line 1.", "Line 2."]


def test_dropped_latex_or_citation_falls_back_to_source(client, monkeypatch):
    text = "我们提出 $x^2$ 方法[1]。\n\n第二段保留[2]标记。"
    calls = {"n": 0}

    def fake(user, *, system=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # 公式和引用都被模型吃掉了 —— 这一段必须退回原文
            return {"en": "We propose a method."}
        return {"en": "The second paragraph keeps [2]."}

    monkeypatch.setattr("app.llm.client.chat_json", fake)

    body = client.post("/api/translate/zh-en", json={"text": text}).json()

    first, second = body["paragraphs"]
    assert first["en"] == first["zh"], "丢片段的段落必须退回原文，不能假装翻好了"
    assert second["en"] == "The second paragraph keeps [2]."
    assert body["degraded"] is True
    note = " ".join(body["notes"])
    assert "$x^2$" in note
    assert "[1]" in note


def test_inline_code_must_survive_translation(client, monkeypatch):
    """行内代码同样属于不可翻译片段，丢了也要回退。"""
    text = "调用 `softmax()` 完成归一化。"
    monkeypatch.setattr(
        "app.llm.client.chat_json",
        lambda user, **kwargs: {"en": "Call softmax to normalize."},
    )

    body = client.post("/api/translate/zh-en", json={"text": text}).json()

    assert body["paragraphs"][0]["en"] == body["paragraphs"][0]["zh"]
    assert body["degraded"] is True
    assert "`softmax()`" in " ".join(body["notes"])


def test_llm_outage_returns_200_with_original_text(client, monkeypatch):
    from app.llm.client import LLMError

    def down(*args, **kwargs):
        raise LLMError("缺少环境变量 DEEPSEEK_API_KEY")

    monkeypatch.setattr("app.llm.client.chat_json", down)

    r = client.post("/api/translate/zh-en", json={"text": "第一段。\n\n第二段。"})

    assert r.status_code == 200, "断网不能让翻译接口 500"
    body = r.json()
    assert body["degraded"] is True
    assert [p["zh"] for p in body["paragraphs"]] == ["第一段。", "第二段。"]
    assert all(p["en"] == "" for p in body["paragraphs"])
    assert "模型不可用" in " ".join(body["notes"])


def test_invalid_model_output_falls_back_per_paragraph(client, monkeypatch):
    """模型返回的结构不对时，只回退该段，不影响其它段。"""
    calls = {"n": 0}

    def fake(user, *, system=None, **kwargs):
        calls["n"] += 1
        return {"nope": "没有 en 字段"} if calls["n"] == 1 else {"en": "Second."}

    monkeypatch.setattr("app.llm.client.chat_json", fake)

    body = client.post("/api/translate/zh-en", json={"text": "第一段。\n\n第二段。"}).json()

    assert body["paragraphs"][0]["en"] == "第一段。"
    assert body["paragraphs"][1]["en"] == "Second."
    assert body["degraded"] is True
    assert "第 1 段" in body["notes"][0]


def test_empty_input_is_400(client):
    for payload in ({"text": ""}, {"text": "   \n\n  "}):
        r = client.post("/api/translate/zh-en", json=payload)
        assert r.status_code == 400, r.text


def test_overlong_input_is_413(client, monkeypatch):
    def fake(user, *, system=None, **kwargs):
        raise AssertionError("超长输入不该走到模型调用")

    monkeypatch.setattr("app.llm.client.chat_json", fake)

    r = client.post(
        "/api/translate/zh-en", json={"text": "字" * (MAX_INPUT_CHARS + 1)}
    )

    assert r.status_code == 413
    assert str(MAX_INPUT_CHARS) in r.json()["detail"]
