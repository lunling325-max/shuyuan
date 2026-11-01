"""所有"去外面取数据"的地方共用的一个网络层。

为什么单独抽出来：arXiv、OpenAlex、Crossref、Semantic Scholar 都要同一套东西 ——
统一的 `User-Agent`、统一的超时、同一个复用连接、以及**认出半截响应**。
写四遍就会有一遍漏掉长度校验，而断流时 `read()` 是不报错的（见 EXPERIMENTS E-013）。

TLS 版本：**先走默认协议（通常是 1.3），连接建不起来就换 1.2 重发一次**，
胜出的协议按主机记住。哪个版本更快是随网络环境变的，所以不写死：

| 场景 | TLS 1.3 | TLS 1.2 |
|---|---|---|
| 下载 PDF（E-013 实测） | 约 16 KB/s | 约 95 KB/s |
| `export.arxiv.org` 检索（E-016 实测） | 1.7~4.9 秒，3/5 成功 | 起手就 30 秒超时 |

结论：默认 1.3 + 失败换手，两种环境都活得下去。设 `ARXIV_TLS13=1` 就只用 1.3。

**不走系统代理。** 这台机器注册表里配着 `127.0.0.1:7897`，urllib 默认会全程
走它；同样的请求 curl 直连 0.3~1.6 秒、走代理 3 秒起、偶尔整段超时。
设 `RESEARCH_PROXY=system` 用回系统代理，设成 `http://host:port` 则用它。

**同一个主机 3 秒一次**：arXiv 使用条款明确要求，实测违反后直接 `429`，
之后一段时间全部被拒 —— 看上去像"arXiv 挂了"，其实是自己在打自己。
按主机排队，所以多来源检索里不同站仍然并行，只有同一站的连续请求会被拉开。
"""

from __future__ import annotations

import gzip
import io
import os
import ssl
import threading
import time
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

USER_AGENT = (
    "ResearchAgent/0.1 (arxiv-tools; local competition demo; mailto:demo@example.com)"
)

_OPENERS: dict[bool, urllib.request.OpenerDirector] = {}
# 每个主机上次哪种 TLS 走得通：True = 锁 1.2，False = 默认协议。
# 默认先走默认协议；握手失败才换 1.2 重来（见模块开头的实测表）。
_prefer_legacy: dict[str, bool] = {}
DEFAULT_ATTEMPTS = 2

# arXiv 的 API 使用条款要求"每次请求间隔至少 3 秒"。不遵守的下场是实测过的：
# 连发几次之后直接 429 Too Many Requests，而且此后一段时间全部被拒 ——
# 表现成"arXiv 挂了"，其实是自己在打自己。按主机记最后一次放行时间，
# 同一主机的并发请求排队（不同主机互不影响，多来源检索照常并行）。
MIN_INTERVAL_SECONDS = 3.0
_host_locks: dict[str, threading.Lock] = {}
_next_allowed: dict[str, float] = {}
_guard = threading.Lock()


def _throttle(url: str) -> None:
    host = urlsplit(url).netloc.lower()
    with _guard:
        lock = _host_locks.setdefault(host, threading.Lock())
    with lock:
        wait = _next_allowed.get(host, 0.0) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _next_allowed[host] = time.monotonic() + MIN_INTERVAL_SECONDS


def modern_only() -> bool:
    return os.environ.get("ARXIV_TLS13", "").strip().lower() in ("1", "true", "yes")


def tls_context(legacy: bool = False) -> ssl.SSLContext:
    """`legacy=True` 把最高版本锁到 1.2；默认交给 OpenSSL 协商（通常是 1.3）。"""
    context = ssl.create_default_context()
    if legacy:
        context.maximum_version = ssl.TLSVersion.TLSv1_2
    return context


def proxy_handlers() -> list[urllib.request.BaseHandler]:
    """默认直连；`RESEARCH_PROXY` 可以改回系统代理或指定一个代理。

    urllib 默认会读环境变量**和 Windows 注册表**里的代理。这台机器两者都配着
    本地代理，于是每一次请求都多绕一跳 —— 实测同一条 Atom feed，直连 1.7 秒、
    走代理 3 秒起，还更容易整段超时。
    """
    mode = (os.environ.get("RESEARCH_PROXY") or "").strip()
    if not mode or mode.lower() in ("direct", "none", "off", "0"):
        return [urllib.request.ProxyHandler({})]   # 空字典 = 不用任何代理
    if mode.lower() == "system":
        return []                                  # 交回默认行为
    return [urllib.request.ProxyHandler({"http": mode, "https": mode})]


def opener(legacy: bool = False) -> urllib.request.OpenerDirector:
    """复用同一个 opener：每次新建都要重新做一次 DNS + 握手。"""
    cached = _OPENERS.get(legacy)
    if cached is None:
        cached = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=tls_context(legacy)),
            *proxy_handlers(),
        )
        _OPENERS[legacy] = cached
    return cached


class ConnectFailed(OSError):
    """连接（含 TLS 握手）就没建起来。

    这类失败发生在**请求发出去之前**：没给对方添负担，也没进对方的计数，
    换一种协议重来是安全的。和"请求已经发出、等着读响应时超时"要分开 ——
    后者多半是被限流按住（实测 arXiv 的 429 要等五十多秒才回），
    重发只是往封禁窗口里多扔一个请求。
    """


def _gunzip(body: bytes, max_bytes: int | None) -> bytes:
    """解开 gzip；读到 `max_bytes + 1` 就停，别让压缩炸弹把内存吃光。"""
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as fh:
        return fh.read((max_bytes + 1) if max_bytes is not None else -1)


def _get(
    url: str, legacy: bool, timeout: float, headers: dict[str, str] | None, max_bytes: int | None
) -> bytes:
    # 默认点名要 gzip：这台机器上 Python 的 TLS 客户端吞吐只有 curl 的十分之一
    # （实测 6~10 KB/s 对 40~120 KB/s），Atom feed 压完只剩十分之一大小，
    # 少传的那部分就是少等的那些秒。
    sent = {"Accept-Encoding": "gzip", "User-Agent": USER_AGENT}
    sent.update(headers or {})
    request = urllib.request.Request(url, headers=sent)
    try:
        response = opener(legacy).open(request, timeout=timeout)
    except HTTPError as exc:
        if exc.code == 429:
            # 说清楚是"被限流"而不是"网络坏了"：前者等一会儿就好，后者要排查
            raise OSError(
                f"{urlsplit(url).netloc} 限流（HTTP 429）：对方要求降低请求频率，稍等一会儿再试"
            ) from exc
        raise
    except (URLError, ssl.SSLError, TimeoutError, ConnectionError) as exc:
        raise ConnectFailed(str(exc) or type(exc).__name__) from exc

    with response:
        data = response.read((max_bytes + 1) if max_bytes is not None else -1)
        declared = response.headers.get("Content-Length")
        if declared is not None and declared.isdigit() and len(data) < int(declared):
            raise OSError(f"下载被截断：只收到 {len(data)}/{declared} 字节")
        encoding = (response.headers.get("Content-Encoding") or "").strip().lower()
    if encoding == "gzip":
        try:
            data = _gunzip(data, max_bytes)
        except (OSError, EOFError) as exc:
            raise OSError(f"gzip 解压失败：{exc}") from exc
    return data


def fetch_bytes(
    url: str,
    *,
    timeout: float = 15,
    headers: dict[str, str] | None = None,
    max_bytes: int | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
) -> bytes:
    """取回响应体；声明了 Content-Length 就必须一斤不少。

    断流时 `read()` 会"正常"返回半截内容，不抛任何异常 —— 这里用长度比对
    把它变成一次普通的失败，交给上层重试，而不是让半截 JSON/PDF 往下流。

    `max_bytes` 给 PDF 那种有大小上限的下载用：多读一个字节就能判断超没超，
    不会因为一份异常大的响应把内存吃满。

    **TLS 版本是试出来的，不是配出来的。** 先按这个主机上次成功的协议发
    （没见过就先用默认协议），连接建不起来就换另一个协议再试一次，
    并把胜出的协议记住。实测 `export.arxiv.org` 有约三分之一的请求会
    `UNEXPECTED_EOF_WHILE_READING` 或握手超时 —— 换手重发基本就好了，
    而让整页空白是最差的结果。

    重试只发生在"请求还没发出去"的阶段（`ConnectFailed`）。
    HTTP 错误码、半截内容、读超时都不重发：

    | 失败 | 重发有用吗 |
    |---|---|
    | 握手/连接超时、TLS EOF | 有用，换协议常常就好 |
    | 读超时 | 没用还可能更糟 —— 被限流时 429 要等 50 多秒才回 |
    | 4xx / 5xx | 没用，同样的请求只会得到同样的错误 |
    | 半截内容 | 交给上层按语义决定（下载层连试三次） |
    """
    host = urlsplit(url).netloc.lower()
    first = _prefer_legacy.get(host, False)

    last_exc: Exception | None = None
    for attempt in range(max(1, attempts)):
        # 第一次用记住的协议，之后轮流换 —— 换手常常就好了
        legacy = first if attempt % 2 == 0 else not first
        if modern_only():
            legacy = False
        _throttle(url)   # 重发也是一次新请求，同样要守 3 秒间隔
        try:
            data = _get(url, legacy, timeout, headers, max_bytes)
        except ConnectFailed as exc:
            last_exc = exc
            continue
        _prefer_legacy[host] = legacy
        return data

    if last_exc is None:   # pragma: no cover —— attempts 至少是 1，走不到
        raise RuntimeError(f"取不到 {url}")
    raise last_exc
