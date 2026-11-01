"""网络层的测试（全程离线）。

守五件事：
1. 连接建不起来（握手超时、TLS EOF）会**换一种协议**再发一次，而不是直接降级；
2. 哪个协议成功过就记住，下次直接用，不重复付一次失败的代价；
3. 请求已经发出去了才失败（读超时、4xx/5xx、半截内容）**不重发** ——
   被限流时 arXiv 的 429 要等五十多秒才回，重发只是往封禁窗口里多扔一个请求；
4. 429 被翻译成一句人能读懂的话（"被限流"而不是"网络坏了"）；
5. Content-Length 说多少就得收到多少。
"""

from __future__ import annotations

import gzip
import io
import ssl
import urllib.request
from urllib.error import HTTPError, URLError

import pytest

from app.tools import net

URL = "https://export.arxiv.org/api/query?search_query=cat:cs.CL"


@pytest.fixture(autouse=True)
def _no_wait(monkeypatch):
    """3 秒间隔是给真 arXiv 的，测试里一笔勾销。"""
    monkeypatch.setattr(net, "_throttle", lambda _url: None)
    monkeypatch.setattr(net, "_prefer_legacy", {})


def _fake_get(outcomes, seen):
    def get(url, legacy, timeout, headers, max_bytes):
        seen.append(legacy)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return get


class _Response(io.BytesIO):
    """够 `_get` 用：context manager、read、headers。"""

    def __init__(self, body: bytes, length: int | None = None, encoding: str = ""):
        super().__init__(body)
        self.headers = {
            "Content-Length": str(len(body) if length is None else length),
            "Content-Encoding": encoding,
        }


class _Opener:
    def __init__(self, outcome):
        self.outcome = outcome

    def open(self, request, timeout=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if isinstance(self.outcome, _Response):
            return self.outcome
        return _Response(self.outcome)


# ---------------------------------------------------------------- 连接阶段失败：重试


def test_handshake_timeout_retries_with_the_other_protocol(monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(
        net, "_get",
        _fake_get([net.ConnectFailed("handshake operation timed out"), b"ok"], seen),
    )

    assert net.fetch_bytes(URL) == b"ok"
    # 先默认协议，失败了换 1.2 —— 两次是不同协议
    assert seen == [False, True]


def test_tls_eof_from_open_becomes_a_retryable_failure(monkeypatch):
    """实测的错误长这样：`<urlopen error _ssl.c:993: The handshake operation timed out>`。"""
    opened: list[bool] = []

    def opener_for(legacy):
        opened.append(legacy)
        if len(opened) == 1:
            return _Opener(URLError(TimeoutError("The handshake operation timed out")))
        return _Opener(b"ok")

    monkeypatch.setattr(net, "opener", opener_for)

    assert net.fetch_bytes(URL) == b"ok"
    assert opened == [False, True]


def test_the_winning_protocol_is_remembered(monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(
        net, "_get",
        _fake_get([net.ConnectFailed("EOF"), b"first", b"second"], seen),
    )

    net.fetch_bytes(URL)
    net.fetch_bytes(URL)
    # 第三次请求直接走赢的那种协议，不再先撞墙
    assert seen == [False, True, True]


def test_single_attempt_host_fails_after_one_shot(monkeypatch):
    """下载用它：下载层自己连试三次，网络层再乘一次就是九次。"""
    seen: list[bool] = []
    monkeypatch.setattr(
        net, "_get", _fake_get([net.ConnectFailed("reset"), b"ok"], seen)
    )

    with pytest.raises(net.ConnectFailed):
        net.fetch_bytes(URL, attempts=1)
    assert seen == [False]


def test_modern_only_env_skips_the_legacy_attempt(monkeypatch):
    seen: list[bool] = []
    monkeypatch.setenv("ARXIV_TLS13", "1")
    monkeypatch.setattr(
        net, "_get", _fake_get([net.ConnectFailed("EOF"), b"ok"], seen)
    )

    assert net.fetch_bytes(URL) == b"ok"
    assert seen == [False, False]


# ---------------------------------------------------------------- 代理与压缩


def test_system_proxy_is_bypassed_by_default(monkeypatch):
    """注册表里那个 127.0.0.1:7897 不该被 urllib 悄悄用上。"""
    monkeypatch.delenv("RESEARCH_PROXY", raising=False)
    handlers = net.proxy_handlers()
    assert isinstance(handlers[0], urllib.request.ProxyHandler)
    assert handlers[0].proxies == {}


def test_research_proxy_can_restore_system_or_pick_one(monkeypatch):
    monkeypatch.setenv("RESEARCH_PROXY", "system")
    assert net.proxy_handlers() == []          # 交回 urllib 默认行为

    monkeypatch.setenv("RESEARCH_PROXY", "http://127.0.0.1:1080")
    proxies = net.proxy_handlers()[0].proxies
    assert proxies["https"] == "http://127.0.0.1:1080"


def test_gzip_body_is_decoded(monkeypatch):
    raw = gzip.compress(b"<feed>" + b"x" * 9000 + b"</feed>")
    monkeypatch.setattr(
        net, "opener", lambda legacy=False: _Opener(_Response(raw, encoding="gzip"))
    )

    body = net.fetch_bytes(URL)
    assert body.startswith(b"<feed>") and len(body) > 9000


# ---------------------------------------------------------------- 已发出：不重试


def test_read_timeout_is_not_retried(monkeypatch):
    """被限流时 arXiv 要按五十多秒才回 429，我们 15 秒就超时了。

    这种"慢"再重发一次，只是往封禁窗口里多扔一个请求。
    """
    seen: list[bool] = []
    monkeypatch.setattr(
        net, "_get",
        _fake_get([TimeoutError("The read operation timed out"), b"ok"], seen),
    )

    with pytest.raises(TimeoutError):
        net.fetch_bytes(URL)
    assert seen == [False]


def test_http_error_is_not_retried(monkeypatch):
    seen: list[bool] = []
    error = HTTPError(URL, 500, "boom", {}, None)
    monkeypatch.setattr(net, "_get", _fake_get([error], seen))

    with pytest.raises(HTTPError):
        net.fetch_bytes(URL)
    assert seen == [False], "5xx 重发一遍还是 5xx，不该白白多打一次"


def test_rate_limit_becomes_a_readable_message(monkeypatch):
    """429 走真实 `_get`：要验的就是它把"被限流"和"网络坏了"分开。"""
    monkeypatch.setattr(
        net, "opener",
        lambda legacy=False: _Opener(HTTPError(URL, 429, "Too Many Requests", {}, None)),
    )

    with pytest.raises(OSError, match="限流（HTTP 429）"):
        net.fetch_bytes(URL)


def test_truncated_body_is_not_retried_here(monkeypatch):
    """半截内容由上层按语义重试（下载 3 次），网络层不掺和。"""
    seen: list[bool] = []
    monkeypatch.setattr(
        net, "_get",
        _fake_get([OSError("下载被截断：只收到 10/20 字节"), b"ok"], seen),
    )

    with pytest.raises(OSError, match="截断"):
        net.fetch_bytes(URL)
    assert seen == [False]


def test_short_body_against_declared_length_is_a_failure(monkeypatch):
    """断流时 `read()` 不报错 —— 长度比对是我们唯一的证据。"""
    class _OpenerWithShortBody(_Opener):
        def open(self, request, timeout=None):
            return _Response(self.outcome, length=20)

    monkeypatch.setattr(
        net, "opener", lambda legacy=False: _OpenerWithShortBody(b"x" * 10)
    )

    with pytest.raises(OSError, match="截断"):
        net.fetch_bytes(URL)
