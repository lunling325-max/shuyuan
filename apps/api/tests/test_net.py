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


# ---------------------------------------------------------------- 分块传输

def test_chunked_response_is_read_in_full():
    """真 chunked 响应不许用 `read(-1)`。

    `http.client` 只有 amt is None 时才走"剥掉分块框架、读完整个流"的正规路径；
    amt=-1 会命中 `amt <= chunk_left` 分支，先读满第一个 chunk，再用
    `_safe_read(-1)` 把剩余裸字节（后续 chunk 的长度行、CRLF）一起吞进来。
    体里插进框架字节，gzip 解压必然 `Error -3 while decompressing data`。

    线上就是这么坏的：OpenAlex 与 Zenodo 走 chunked 且不发 Content-Length，
    5 个检索源只有这两个全灭；Crossref 带 Content-Length，一直正常（E-021）。
    用假响应（io.BytesIO）测不出来 —— 它的 `read(-1)` 乖乖读完全部。

    **注意这条只在 Python ≤ 3.11 上会红。** 3.12 的 `_read_chunked` 开头补了
    `if amt is not None and amt < 0: amt = None`（上游修复），所以本机 venv（3.12）
    就算写回 `read(-1)` 也照样通过 —— 而镜像用 `python:3.11-slim`。
    正因为如此，下面另外两条用"看 read 的实参"来守，不依赖解释器版本。
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    payload = b'{"results": [' + b",".join(b'{"id": %d}' % i for i in range(200)) + b"]}"
    compressed = gzip.compress(payload)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):   # noqa: N802  —— 父类要求这个名字
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            step = 512
            for i in range(0, len(compressed), step):
                piece = compressed[i:i + step]
                self.wfile.write(b"%x\r\n" % len(piece) + piece + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

        def log_message(self, *args):   # 别把访问日志刷进测试输出
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/works?search=x"
        assert net.fetch_bytes(url, timeout=10) == payload
    finally:
        server.shutdown()
        server.server_close()


class _RecordingResponse(_Response):
    """记下 `read` 实际收到的实参 —— 这条断言和解释器版本无关。"""

    def __init__(self, body: bytes, **kw):
        super().__init__(body, **kw)
        self.reads: list = []

    def read(self, amt=None):
        self.reads.append(amt)
        return super().read(amt)


def test_read_is_called_with_no_amount_when_there_is_no_limit(monkeypatch):
    """不限长时必须 `read()`，**不能是 `read(-1)`**。

    Python ≤ 3.11 的 `_read_chunked(-1)` 会先读满第一个 chunk，再用
    `_safe_read(-1)` 把后面的 chunk 框架字节一起吞进来，gzip 解压随即报
    `Error -3 while decompressing data`。3.12 补上了这个洞，所以只在容器里
    （`python:3.11-slim`）复现 —— 这正是"本地全绿、上线两个源全灭"的原因。
    """
    response = _RecordingResponse(gzip.compress(b"x" * 5000), encoding="gzip")
    monkeypatch.setattr(net, "opener", lambda legacy=False: _Opener(response))

    assert net.fetch_bytes(URL) == b"x" * 5000
    assert response.reads == [None]


def test_read_is_capped_one_byte_past_the_limit(monkeypatch):
    """给了上限就多读一个字节：刚好判断"超没超"，又不至于把内存吃满。

    这里故意不发 Content-Length（chunked/流式都这样），只看 `read` 的实参。
    真发了 Content-Length 时，读到上限就停会触发"下载被截断"的长度比对 ——
    那是既有语义（超限的大文件报截断），这条测试不碰它。
    """
    response = _RecordingResponse(b"y" * 500)
    response.headers = {}       # 不发 Content-Length
    monkeypatch.setattr(net, "opener", lambda legacy=False: _Opener(response))

    assert net.fetch_bytes(URL, max_bytes=100) == b"y" * 101
    assert response.reads == [101]


def test_broken_gzip_stream_becomes_a_plain_oserror(monkeypatch):
    """zlib.error 不是 OSError，不显式接住会以 `error Error -3 ...` 原文冒到界面。"""
    monkeypatch.setattr(
        net, "opener",
        lambda legacy=False: _Opener(_Response(b"\x1f\x8b\x08\x00garbage", encoding="gzip")),
    )

    with pytest.raises(OSError, match="gzip 解压失败"):
        net.fetch_bytes(URL)
