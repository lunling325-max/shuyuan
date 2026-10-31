"""流式问答的测试（全程离线，不打真接口）。

重点不是"模型写得好不好"，而是协议和失败语义：
1. 事件顺序固定，token 拼起来必须等于模型完整原文；
2. 流结束后仍要跑真守卫：编造的引用不产生锚点、结论标 unsupported；
3. 模型挂了 / 没配 key，HTTP 仍是 200，但只能收到 error，收不到 done；
4. retrieve 模式完全不碰模型。

测试同样 monkeypatch `app.llm.client.chat_json` 与 `stream_chat`，
但检索、BM25、TextIndex、守卫校验都是真跑的。
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"

# 与 tests/test_api.py 同一句原文：这条在 TextIndex 里能逐字命中。
VERBATIM = (
    "We suspect that for large values of dk, the dot products grow large in magnitude, "
    "pushing the softmax function into regions where it has extremely small gradients"
)

FABRICATED = (
    "Attention is clearly the most important invention of the twenty-first century "
    "and every modern system depends on it in ways never described in this paper"
)


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """往 registry 里灌一份真实 PDF；不依赖 main.py 的上传路由。"""
    from app.api.registry import PaperRegistry

    reg = PaperRegistry(tmp_path_factory.mktemp("papers_stream"))
    reg.add(PDF_V7.read_bytes(), PDF_V7.name)
    return reg


@pytest.fixture()
def paper_id(registry) -> str:
    return registry.all()[0].id


@pytest.fixture()
def client(registry, monkeypatch):
    """裸 FastAPI + include_router，只挂被测路由。"""
    from app.api import deps
    from app.api.routers.stream import router

    monkeypatch.setattr(deps, "registry", registry)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------- 工具


def _events(text: str) -> list[tuple[str, dict]]:
    """把整个 SSE 响应体解析成 [(event, data), ...]。

    不按 `\\n\\n` 硬切字符串以外的细节留给这里，测试只断言语义。
    """
    out: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
        if event is None:
            continue
        out.append((event, json.loads("\n".join(data_lines))))
    return out


def _names(events: list[tuple[str, dict]]) -> list[str]:
    return [name for name, _data in events]


def _fake_expand(queries: list[str] | None = None):
    """检索词扩展的桩：只回答扩展问题，其余调用视为测试写错了。"""

    def call(user, *, system=None, **kwargs):
        assert system and "search queries" in system, "这里只应发生检索词扩展"
        return {"queries": queries or ["scaled dot product attention sqrt dk"]}

    return call


def _stream(client, paper_id: str, *, question: str = "为什么除以 sqrt(dk)？", **body):
    payload = {"question": question, **body}
    with client.stream(
        "POST", f"/api/papers/{paper_id}/qa/stream", json=payload
    ) as response:
        text = "".join(response.iter_text())
        return response.status_code, text


def _fake_stream(raw: str, *, pieces: int = 7):
    """把完整原文切成若干段模拟 token。"""
    size = max(1, len(raw) // pieces)
    chunks = [raw[i : i + size] for i in range(0, len(raw), size)]

    def call(user, *, system=None, temperature=0.2, model=None):
        assert user.startswith("问题：")
        assert "证据块：" in user
        for chunk in chunks:
            yield chunk

    return call


# ---------------------------------------------------------------- 路由协议


def test_events_are_ordered_and_tokens_reassemble_to_raw_model_output(
    client, paper_id, monkeypatch
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("app.llm.client.chat_json", _fake_expand())

    raw = json.dumps(
        {
            "claims": [
                {"text": "大 dk 时点积过大会把 softmax 推进梯度极小区", "quotes": [VERBATIM]},
                {"text": "编造出来的结论", "quotes": [FABRICATED]},
            ]
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr("app.llm.client.stream_chat", _fake_stream(raw))

    status, text = _stream(client, paper_id)

    assert status == 200
    events = _events(text)
    names = _names(events)
    assert names[:3] == ["meta", "queries", "evidence"]
    assert names[-2:] == ["claims", "done"]
    token_events = [data for name, data in events if name == "token"]
    assert len(token_events) > 1, "应当逐 token 发送，而不是攒完再发"
    assert "".join(data["text"] for data in token_events) == raw

    meta = events[0][1]
    assert meta["paper_id"] == paper_id
    assert meta["title"]
    assert meta["question"] == "为什么除以 sqrt(dk)？"

    queries = events[1][1]["queries"]
    assert queries == ["scaled dot product attention sqrt dk"]

    hits = events[2][1]["hits"]
    assert hits
    assert set(hits[0]) == {"chunk_index", "page", "section", "score", "preview"}

    claims_event = events[-2][1]
    claims = claims_event["claims"]
    assert [c["status"] for c in claims] == ["supported", "unsupported"]
    assert claims[0]["anchors"], "逐字引用必须落回原文锚点"
    assert claims[0]["anchors"][0]["exact"] is True
    assert claims[1]["anchors"] == [], "编造的引用不能产生任何锚点"
    assert claims[1]["reason"], "被拦下时必须说明原因"
    assert claims_event["stats"] == {"supported": 1, "weak": 0, "unsupported": 1}

    done = events[-1][1]
    assert isinstance(done["took_ms"], int)
    assert done["took_ms"] >= 0


def test_retrieve_mode_never_touches_the_model(client, paper_id, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("retrieve 模式不应调用任何模型接口")

    monkeypatch.setattr("app.llm.client.chat_json", boom)
    monkeypatch.setattr("app.llm.client.stream_chat", boom)

    status, text = _stream(client, paper_id, mode="retrieve")

    assert status == 200
    events = _events(text)
    assert _names(events) == ["meta", "queries", "evidence", "done"]
    assert events[1][1]["queries"] == ["为什么除以 sqrt(dk)？"]
    assert events[2][1]["hits"], "retrieve 模式仍要给出召回证据"


def test_llm_error_midstream_ends_with_error_not_done(client, paper_id, monkeypatch):
    from app.llm.client import LLMError

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("app.llm.client.chat_json", _fake_expand())

    def broken(user, *, system=None, temperature=0.2, model=None):
        yield "先吐一半"
        raise LLMError("HTTP 500: upstream down")

    monkeypatch.setattr("app.llm.client.stream_chat", broken)

    status, text = _stream(client, paper_id)

    assert status == 200, "SSE 已开始，失败不能再改状态码"
    events = _events(text)
    assert _names(events) == ["meta", "queries", "evidence", "token", "error"]
    assert events[-1][1]["stage"] == "llm"
    assert "HTTP 500" in events[-1][1]["message"]
    assert "done" not in _names(events)


def test_missing_key_folds_into_llm_error_but_keeps_evidence(
    client, paper_id, monkeypatch
):
    from app.llm.client import LLMError

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def down(*args, **kwargs):
        raise LLMError("缺少环境变量 DEEPSEEK_API_KEY")

    monkeypatch.setattr("app.llm.client.chat_json", down)

    def should_not_stream(*args, **kwargs):
        raise AssertionError("没配 key 时不应开始 token 流")

    monkeypatch.setattr("app.llm.client.stream_chat", should_not_stream)

    status, text = _stream(client, paper_id)

    assert status == 200
    events = _events(text)
    assert _names(events) == ["meta", "queries", "evidence", "error"]
    assert events[-1][1]["stage"] == "llm"
    assert "DEEPSEEK_API_KEY" in events[-1][1]["message"]
    assert events[2][1]["hits"], "断网演示的兜底：证据必须先给出来"
    assert "test-key" not in text, "任何事件都不能泄露密钥"


def test_unknown_paper_is_still_404_before_streaming(client):
    r = client.post("/api/papers/not-exist/qa/stream", json={"question": "hi"})
    assert r.status_code == 404


def test_empty_question_is_rejected_by_the_reused_schema(client, paper_id):
    assert client.post(
        f"/api/papers/{paper_id}/qa/stream", json={"question": ""}
    ).status_code == 422


# ---------------------------------------------------------------- 纯逻辑


def test_parse_model_claims_strips_markdown_fence():
    from app.rag.streaming import parse_model_claims

    raw = '```json\n{"claims": [{"text": "结论", "quotes": ["原文"]}]}\n```'
    claims = parse_model_claims(raw, doc_id="d1")

    assert len(claims) == 1
    assert claims[0].text == "结论"
    assert claims[0].quotes[0].text == "原文"


def test_sse_keeps_chinese_readable():
    from app.rag.streaming import sse

    frame = sse("token", {"text": "注意"})

    assert "注意" in frame
    assert "\\u" not in frame
    assert frame.startswith("event: token\ndata: ")
    assert frame.endswith("\n\n")


# ---------------------------------------------------------------- 客户端


class _FakeResponse:
    """最小可迭代响应；`failure` 用于模拟读到一半连接断开。"""

    def __init__(self, lines: list[bytes], failure: BaseException | None = None):
        self.lines = lines
        self.failure = failure
        self.closed = False

    def __iter__(self):
        yield from self.lines
        if self.failure is not None:
            raise self.failure

    def close(self):
        self.closed = True


def test_stream_chat_parses_sse_deltas_and_stops_at_done(monkeypatch):
    from app.llm.client import stream_chat

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data)
        captured["timeout"] = timeout
        captured["resp"] = _FakeResponse(
            [
                b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                b": keepalive\n",
                b'data: {"choices":[{"delta":{"content":"He"}}]}\n',
                b"data: not-json\n",
                b'data: {"choices":[{"delta":{"content":"llo"}}]}\n',
                b"data: [DONE]\n",
                b'data: {"choices":[{"delta":{"content":"after-done"}}]}\n',
            ]
        )
        return captured["resp"]

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    tokens = list(stream_chat("问题", system="系统提示"))

    assert tokens == ["He", "llo"]
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "系统提示"}
    assert captured["timeout"], "必须设置超时，否则半死连接会永久挂起"
    assert captured["resp"].closed is True


def test_stream_chat_without_key_raises_llm_error(monkeypatch):
    from app.llm.client import LLMError, stream_chat

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(LLMError):
        list(stream_chat("hi"))


def test_stream_chat_midstream_disconnect_is_identifiable(monkeypatch):
    from app.llm.client import LLMError, LLMStreamError, stream_chat

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(
            [b'data: {"choices":[{"delta":{"content":"half"}}]}\n'],
            failure=OSError("connection reset by peer"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(LLMStreamError) as excinfo:
        list(stream_chat("hi"))

    assert isinstance(excinfo.value, LLMError), "上层只捕获 LLMError 也应接得住"


def test_stream_chat_http_error_body_is_bounded(monkeypatch):
    from app.llm.client import LLMError, stream_chat

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://example.invalid", 500, "boom", None, io.BytesIO(b"x" * 1000)
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(LLMError) as excinfo:
        list(stream_chat("hi"))

    message = str(excinfo.value)
    assert message.startswith("HTTP 500:")
    assert len(message) <= len("HTTP 500: ") + 400
