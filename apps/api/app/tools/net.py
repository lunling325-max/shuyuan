"""所有"去外面取数据"的地方共用的一个网络层。

为什么单独抽出来：arXiv、OpenAlex、Crossref、Semantic Scholar 都要同一套东西 ——
统一的 `User-Agent`、统一的超时、同一个复用连接、以及**认出半截响应**。
写四遍就会有一遍漏掉长度校验，而断流时 `read()` 是不报错的（见 EXPERIMENTS E-013）。

TLS 版本：默认把最高版本锁到 1.2。这不是"为了兼容"而降级，是实测结论 ——
arXiv 在 Fastly 后面，本机 Python(OpenSSL) 协商 TLS 1.3 时只有约 16 KB/s，
锁 1.2 之后是 95 KB/s 以上。设 `ARXIV_TLS13=1` 可以关掉这个上限。
"""

from __future__ import annotations

import os
import ssl
import urllib.request

USER_AGENT = (
    "ResearchAgent/0.1 (arxiv-tools; local competition demo; mailto:demo@example.com)"
)

_OPENER: urllib.request.OpenerDirector | None = None


def tls_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    if os.environ.get("ARXIV_TLS13", "").strip().lower() not in ("1", "true", "yes"):
        context.maximum_version = ssl.TLSVersion.TLSv1_2
    return context


def opener() -> urllib.request.OpenerDirector:
    """复用同一个 opener：每次新建都要重新做一次 DNS + 握手。"""
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=tls_context())
        )
    return _OPENER


def fetch_bytes(
    url: str,
    *,
    timeout: float = 15,
    headers: dict[str, str] | None = None,
    max_bytes: int | None = None,
) -> bytes:
    """取回响应体；声明了 Content-Length 就必须一斤不少。

    断流时 `read()` 会"正常"返回半截内容，不抛任何异常 —— 这里用长度比对
    把它变成一次普通的失败，交给上层重试，而不是让半截 JSON/PDF 往下流。

    `max_bytes` 给 PDF 那种有大小上限的下载用：多读一个字节就能判断超没超，
    不会因为一份异常大的响应把内存吃满。
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   **(headers or {})})
    with opener().open(request, timeout=timeout) as response:
        data = response.read((max_bytes + 1) if max_bytes is not None else -1)
        declared = response.headers.get("Content-Length")
        if declared is not None and declared.isdigit() and len(data) < int(declared):
            raise OSError(f"下载被截断：只收到 {len(data)}/{declared} 字节")
    return data
