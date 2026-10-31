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
   fetch 固定带 UA、固定 15 秒超时，避免演示时卡死。

检索失败返回 `degraded=True` 而不是抛异常：断网时前端仍然要能打开页面，
把"没查到"和"查不了"用同一个 200 响应表达，界面才不会白屏。
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import quote
from xml.etree import ElementTree

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf"
USER_AGENT = "ResearchAgent/0.1 (arxiv-tools; local competition demo)"
TIMEOUT_SECONDS = 15
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
    """真实网络请求。带 UA 与超时；最多读 50MB+1，超限能在下载后立刻发现。"""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read(MAX_PDF_BYTES + 1)


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

    url = (
        f"{API_URL}?search_query=all:{quote(normalized_query, safe='')}"
        f"&start=0&max_results={limit}"
    )

    try:
        data = _fetch(url, fetch)
    except Exception as exc:  # noqa: BLE001 —— 断网/超时/DNS 的异常类型不可枚举
        reason = str(exc) or type(exc).__name__
        return SearchResult(
            query=normalized_query,
            results=[],
            degraded=True,
            note=f"arXiv 检索失败：{reason}",
        )

    try:
        results = parse_atom(data)
    except Exception as exc:  # noqa: BLE001 —— 上游返回坏 XML 时同样不能 500
        reason = str(exc) or type(exc).__name__
        return SearchResult(
            query=normalized_query,
            results=[],
            degraded=True,
            note=f"arXiv 返回内容无法解析：{reason}",
        )

    return SearchResult(query=normalized_query, results=results, degraded=False)


def download_pdf(arxiv_id: str, *, fetch: Fetch | None = None) -> bytes:
    """下载并校验一篇 arXiv PDF。

    先确认内容确实是 PDF 再交给解析器：把 HTML 错误页喂给 PyMuPDF，
    报出来的异常信息对用户毫无意义。大小上限在读取之后检查 ——
    default_fetch 已经限制读取量，超限文件不会把内存吃满。
    """
    normalized = normalize_arxiv_id(arxiv_id)
    url = f"{PDF_URL}/{normalized}"

    try:
        data = _fetch(url, fetch)
    except Exception as exc:  # noqa: BLE001
        reason = str(exc) or type(exc).__name__
        raise FetchError(f"下载 arXiv PDF 失败：{reason}") from exc

    if len(data) > MAX_PDF_BYTES:
        raise PdfTooLargeError(
            f"PDF 超过 {MAX_PDF_BYTES // 1024 // 1024}MB 上限"
        )
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise NotPdfError("下载到的内容不是 PDF（文件头不是 %PDF）")
    return data
