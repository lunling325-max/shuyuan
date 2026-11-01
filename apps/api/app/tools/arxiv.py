"""arXiv 检索与 PDF 下载：把外部数据源收敛成两个可注入的纯函数。

为什么这样切分：

1. **网络就两个动作**。检索换一份 Atom 列表，下载换一段 PDF 字节。
   其余（版本号归一化、XML 命名空间、换行压平、50MB 上限、%PDF 校验）
   都是确定性的字符串/字节处理，放在这里才能脱离网络单测。
2. **fetch 是可注入的**。默认实现用标准库 `urllib` 真发请求；测试传入假
   fetch，一个网络包都不发。断网演示时请求失败也不抛到路由层 ——
   检索降级为空列表，下载则抛出可映射成 502/400 的领域异常。
3. **版本号必须在这里去掉**。`1706.03762v7` 和 `1706.03762` 指向同一篇论文；
   入库文件名、后续引用图谱只认不带版本的主 id，否则 v1/v7 会被当成两篇。
4. **arXiv 会拒绝没有 User-Agent 的请求**，而且 API 偶尔很慢，所以默认
   fetch 固定带 UA、固定超时，避免演示时卡死。

检索失败返回 `degraded=True` 而不是抛异常：断网时前端仍然要能打开页面，
把"没查到"和"查不了"用同一个 200 响应表达，界面才不会白屏。

**下载为什么重试、为什么把 TLS 锁在 1.2**：在这台机器上实测过同一份
`1706.03762`（2.1MB），arXiv 走 Fastly，握手与吞吐都不稳定 ——

| 条件 | 结果 |
|---|---|
| 默认 TLS（协商到 1.3） | 512KB 用了 31.7 s，约 16 KB/s |
| 锁 TLS 1.2 | 512KB 用了 5.4 s，约 95 KB/s；全量 2.1MB 一次 10.5 s |
| 同一个 urllib 换成清华 PyPI 镜像 | 848KB / 0.4 s，约 2 MB/s（排除"本机网络慢"） |

而且重试是必需的：锁了 1.2 之后仍有约三分之一的请求一路卡到 socket 超时
（`The read operation timed out`）。所以下载默认连试 3 次、逐次退避，
三次都失败才报 502，并在错误里写清"试了几次、可以怎么绕过去"。
检索不重试 —— 它只有 8KB，失败也已经降级成 200，没必要让用户等三倍时间。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import quote
from xml.etree import ElementTree

from app.tools import net

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf"
USER_AGENT = net.USER_AGENT   # 兼容旧引用；实际请求头由 net 统一加
TIMEOUT_SECONDS = 15              # 检索：8KB 的小请求，短超时才能快速降级
DOWNLOAD_TIMEOUT_SECONDS = 60     # 下载：2MB 的 PDF，按实测留足余量
DOWNLOAD_ATTEMPTS = 3
SEARCH_ATTEMPTS = 2               # 检索只再试一次：它本来就该快速降级
RETRY_BACKOFF_SECONDS = 1.0
MAX_PDF_BYTES = 50 * 1024 * 1024

# fetch 的契约只有一条：给 URL，返回响应体。测试用它替换掉真实网络。
Fetch = Callable[[str], bytes]

ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivError(Exception):
    """工具层错误的基类，路由据此映射 HTTP 状态码。"""


class InvalidArxivId(ArxivError, ValueError):
    """arXiv id 格式不合法。路由映射 400。"""


class FetchError(ArxivError):
    """网络层失败（断网、超时、非 2xx）。路由映射 502。"""


class NotPdfError(ArxivError):
    """下载到的内容不是 PDF。路由映射 400。"""


class PdfTooLargeError(ArxivError):
    """PDF 超过 50MB 上限。路由映射 400。"""


@dataclass(frozen=True, slots=True)
class ArxivEntry:
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    published: str = ""
    abstract: str = ""
    pdf_url: str = ""
    abs_url: str = ""


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    results: list[ArxivEntry]
    degraded: bool
    note: str = ""


# 新式 1706.03762 / 老式 hep-th/9901001、math.GT/0309136；版本号单独一组，归一化时丢掉
_ID_RE = re.compile(
    r"(?P<base>(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?"
)


def normalize_arxiv_id(raw: str) -> str:
    """把 arXiv id 归一化成不带版本号的主 id，非法就抛 InvalidArxivId。

    顺手容忍从网页复制来的几种形态：`arXiv:` 前缀、abs/pdf URL、`.pdf` 后缀、
    查询串。它们指向的是同一篇论文，让用户先手动清理没有意义。
    """
    if not isinstance(raw, str):
        raise InvalidArxivId("arXiv id 必须是字符串")

    value = raw.strip()
    if value.lower().startswith("arxiv:"):
        value = value[len("arxiv:"):].strip()
    value = value.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    for marker in ("/abs/", "/pdf/"):
        if marker in value:
            value = value.rsplit(marker, 1)[1]
            break
    if value.lower().endswith(".pdf"):
        value = value[:-4]
    value = value.strip().rstrip("/")

    match = _ID_RE.fullmatch(value)
    if match is None:
        raise InvalidArxivId(f"不是合法的 arXiv id：{raw!r}")
    return match.group("base")


def default_fetch(url: str) -> bytes:
    """真实网络请求。UA、超时、半截响应识别都在 `net` 里统一做。

    超时按目标 URL 分两档：PDF 是几 MB 的实体，给 60 秒；Atom 是一条 8KB 的
    XML，15 秒就该有结果，慢到那个程度说明链路有问题，早点降级比挂着强。
    这么分而不是给 `Fetch` 加一个 `timeout` 参数 —— 那样每个注入的假 fetch
    都得跟着改签名，而注入契约（给 URL，返回字节）越窄越稳。
    """
    timeout = (
        DOWNLOAD_TIMEOUT_SECONDS if url.startswith(PDF_URL) else TIMEOUT_SECONDS
    )
    return net.fetch_bytes(url, timeout=timeout, max_bytes=MAX_PDF_BYTES)


def _fetch(url: str, fetch: Fetch | None) -> bytes:
    # 在调用时解析默认实现，而不是写成参数默认值 —— 测试才能 monkeypatch 它
    if fetch is not None:
        return fetch(url)
    return default_fetch(url)


def _text(node: ElementTree.Element | None) -> str:
    """取元素的全部文本并压成单行，摘要里的换行和缩进不该带到 JSON 里。"""
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def parse_atom(data: bytes) -> list[ArxivEntry]:
    """解析 arXiv Atom feed。

    命名空间必须显式带上：arXiv 的标签全部在 Atom 命名空间里，直接找
    `entry` 会一个都找不到。单条 entry 字段缺失时就地兜底为空串，
    不让一条脏数据毁掉整次检索。
    """
    root = ElementTree.fromstring(data)
    entries: list[ArxivEntry] = []

    for node in root.findall(f"{ATOM}entry"):
        raw_id = _text(node.find(f"{ATOM}id"))
        if not raw_id:
            continue
        try:
            arxiv_id = normalize_arxiv_id(raw_id.rsplit("/", 1)[-1])
        except InvalidArxivId:
            continue  # 无法识别的 id 不入结果，免得前端拿它去下载时 400

        authors = [
            _text(author.find(f"{ATOM}name"))
            for author in node.findall(f"{ATOM}author")
        ]
        authors = [name for name in authors if name]

        pdf_url = ""
        abs_url = ""
        for link in node.findall(f"{ATOM}link"):
            href = (link.get("href") or "").strip()
            if not href:
                continue
            if link.get("title") == "pdf" and not pdf_url:
                pdf_url = href
            elif link.get("rel") == "alternate" and not abs_url:
                abs_url = href

        entries.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                title=_text(node.find(f"{ATOM}title")),
                authors=authors,
                published=_text(node.find(f"{ATOM}published"))[:10],
                abstract=_text(node.find(f"{ATOM}summary")),
                pdf_url=pdf_url or f"{PDF_URL}/{arxiv_id}",
                abs_url=abs_url or f"https://arxiv.org/abs/{arxiv_id}",
            )
        )

    return entries


def _search_candidates(normalized_query: str) -> list[str]:
    """把用户输入翻成 arXiv 的检索式，先整句、再退化成按词搜。

    为什么必须先整句：arXiv 的 `all:` 只绑定紧随其后的**一个**词，
    `all:attention is all you need` 实际是"五个词各自匹配"，相关性排序里
    真正那篇《Attention Is All You Need》根本进不了前五（实测）。写成
    `all:"attention is all you need"` 之后它稳定出现在第 2 位、`ti:"…"` 第 1 位。

    但整句对"我只有主题、没有标题"的检索太苛刻（短语必须原样出现），
    所以整句 0 命中时再按词搜一次 —— 那时候宁可宽，也不要空。

    引号和反斜杠是 arXiv 查询语法的一部分，用户输入里出现就把它们抹掉，
    否则一个多余的 `"` 会把整个检索式变成语法错误。
    """
    cleaned = " ".join(normalized_query.replace('"', " ").replace("\\", " ").split())
    candidates = [f'all:"{cleaned}"']
    if " " in cleaned:
        candidates.append(f"all:{cleaned}")
    return candidates


def _fetch_feed(query: str, limit: int, fetch: Fetch | None) -> list[ArxivEntry]:
    """取一次检索结果；握手/读超时就重试一次，仍失败抛给调用方去降级。

    重试是因为实测里 `export.arxiv.org` 的 TLS 握手会整段卡死
    （`handshake operation timed out`），隔一两秒再发就有了。
    """
    url = (
        f"{API_URL}?search_query={quote(query, safe='')}"
        f"&start=0&max_results={limit}"
    )
    last_exc: Exception | None = None
    for attempt in range(1, SEARCH_ATTEMPTS + 1):
        try:
            return parse_atom(_fetch(url, fetch))
        except Exception as exc:  # noqa: BLE001 —— 网络与解析异常五花八门
            last_exc = exc
            if attempt < SEARCH_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc if last_exc is not None else RuntimeError("arXiv 检索失败")


def search_arxiv(
    query: str,
    limit: int = 5,
    *,
    fetch: Fetch | None = None,
) -> SearchResult:
    """按标题/关键词检索 arXiv；任何网络或解析失败都降级为空结果。"""
    normalized_query = " ".join(str(query).split())
    limit = max(1, min(int(limit), 20))

    if not normalized_query:
        return SearchResult(
            query=normalized_query,
            results=[],
            degraded=False,
            note="查询为空，没有可检索的关键词",
        )

    candidates = _search_candidates(normalized_query)
    try:
        for candidate in candidates:
            results = _fetch_feed(candidate, limit, fetch)
            if results:
                return SearchResult(
                    query=normalized_query,
                    results=results,
                    degraded=False,
                    # 整句没命中、退回按词搜才有结果时，说清楚"这批不是精确匹配"
                    note="" if candidate == candidates[0] else "按整句没有精确匹配，下面是把词拆开搜的结果",
                )
    except Exception as exc:  # noqa: BLE001 —— 断网/超时/DNS 的异常类型不可枚举
        reason = str(exc) or type(exc).__name__
        return SearchResult(
            query=normalized_query,
            results=[],
            degraded=True,
            note=f"arXiv 检索失败：{reason}",
        )

    return SearchResult(
        query=normalized_query,
        results=[],
        degraded=False,
        note="整句和拆成关键词都没有匹配到" if len(candidates) > 1 else "",
    )


def _is_complete_pdf(data: bytes) -> bool:
    """是不是一份下载完整的 PDF。

    判据只有两条，都不看内容质量：文件头是 `%PDF`，结尾 2KB 内出现 `%%EOF`。
    后者是关键 —— 实测断流时 `read()` 会**正常返回半截内容**，不抛任何异常，
    直到 PyMuPDF 报 "Failed to open file … as type pdf"，用户看到的是一句
    莫名其妙的解析错误。一份合法的 PDF 一定以 `%%EOF` 收尾，缺了就说明这次
    传输没走完，该重试的是下载，不是解析。

    不含 `%PDF` 头的（HTML 限流页之类）直接判 True —— 那不是"没下完"，
    是内容不对，交给上层按 400 处理，不该在这里重试三次。
    """
    if not data.lstrip()[:5].startswith(b"%PDF"):
        return True
    return b"%%EOF" in data[-2048:]


def _download_with_retry(url: str, fetch: Fetch | None) -> bytes:
    """下载失败就退避重试，三次都不行才抛 FetchError。

    重试的边界很窄：只重试**网络层**失败（超时、连接被重置、半截内容）。
    内容不对（根本不是 PDF）不重试 —— 再试三次还是那个文件，白白让用户多等。
    """
    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            data = _fetch(url, fetch)
            if _is_complete_pdf(data):
                return data
            last_exc = ValueError(f"只收到 {len(data)} 字节，且结尾没有 %%EOF")
        except Exception as exc:  # noqa: BLE001 —— 网络异常类型不可枚举
            last_exc = exc
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    reason = str(last_exc) or type(last_exc).__name__
    raise FetchError(
        f"下载 arXiv PDF 失败（连试 {DOWNLOAD_ATTEMPTS} 次）：{reason}。"
        "arXiv 偶尔会长时间不给数据或只给半截，稍等一会儿再点「入库」通常就好；"
        "也可以把本地 PDF 直接拖进文献库，不经过下载这一步"
    ) from last_exc


def download_pdf(arxiv_id: str, *, fetch: Fetch | None = None) -> bytes:
    """下载并校验一篇 arXiv PDF。

    先确认内容确实是 PDF 再交给解析器：把 HTML 错误页喂给 PyMuPDF，
    报出来的异常信息对用户毫无意义。大小上限在读取之后检查 ——
    default_fetch 已经限制读取量，超限文件不会把内存吃满。
    """
    normalized = normalize_arxiv_id(arxiv_id)
    url = f"{PDF_URL}/{normalized}"

    data = _download_with_retry(url, fetch)

    if len(data) > MAX_PDF_BYTES:
        raise PdfTooLargeError(
            f"PDF 超过 {MAX_PDF_BYTES // 1024 // 1024}MB 上限"
        )
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise NotPdfError("下载到的内容不是 PDF（文件头不是 %PDF）")
    return data
