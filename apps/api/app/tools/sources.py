"""多来源文献检索：一次查询，同时问 arXiv / OpenAlex / Crossref / Semantic Scholar / Zenodo。

为什么不是"再写一个 arXiv 客户端"：

1. **各站的长处不一样**。arXiv 有全文 PDF 但只覆盖预印本；OpenAlex 覆盖面最广、
   带开放获取（OA）PDF 链接；Crossref 是 DOI 的权威登记处，正式发表的版本
   往往只有它认得出；Semantic Scholar 的摘要与引用信息最全；Zenodo 收数据集、
   预印本与"论文配的代码/材料"，别的站基本不收。
   只查 arXiv，用户搜一篇 NeurIPS 论文常常一无所获 —— 这正是当初的反馈。
2. **一个站挂了不能拖垮整次检索**。各来源并发请求，每站各自降级：
   某站超时/限流只影响它自己那一栏，其余照常返回。超过 `DEADLINE_SECONDS`
   还没回来的站按"超时"处理 —— 等满所有站的超时叠加起来，用户只会以为页面挂了。
3. **能入库的和只能看的要分开**。入库要的是 PDF；只有拿到 PDF 链接、
   并且能定位到 arXiv id（本项目的入库通道就是 arXiv）的条目才给「入库」，
   其余的给原文链接 —— 假装都能入库，用户点下去只会收到一个错。

`Hit.arxiv_id` 非空即"可入库"。OpenAlex / Crossref / S2 的 arXiv 条目都会
把 id 从 `10.48550/arXiv.xxxx` 这类 DOI 或 `arxiv.org/pdf/xxxx` 链接里解出来。
"""

from __future__ import annotations

import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import quote

from app.tools import arxiv as arxiv_tool
from app.tools import net

Fetch = Callable[[str], bytes]

OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"
S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ZENODO_URL = "https://zenodo.org/api/records"

SOURCES = ("arxiv", "openalex", "crossref", "semanticscholar", "zenodo")
SOURCE_ZH = {
    "arxiv": "arXiv",
    "openalex": "OpenAlex",
    "crossref": "Crossref",
    "semanticscholar": "Semantic Scholar",
    "zenodo": "Zenodo",
}

DEFAULT_LIMIT = 25
MAX_LIMIT = 50
TIMEOUT_SECONDS = 20
# 整次检索的总时限：超时没回来的站按"没取到"处理，其余结果照发。
# 20 秒是这台机器上最慢一站（arXiv，含一次重试）的实测上限，再等就没有意义了。
DEADLINE_SECONDS = 20

# 从各种链接/DOI 里把 arXiv id 抠出来：pdf/abs 链接、DataCite 的 10.48550/arXiv.*
_ARXIV_IN_URL = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})",
    re.IGNORECASE,
)
_ARXIV_IN_DOI = re.compile(r"10\.48550/arxiv\.(?P<id>.+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Hit:
    source: str
    source_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: str = ""
    venue: str = ""
    abstract: str = ""
    pdf_url: str = ""
    abs_url: str = ""
    arxiv_id: str = ""

    @property
    def can_ingest(self) -> bool:
        return bool(self.arxiv_id)


@dataclass(frozen=True, slots=True)
class SourceReport:
    name: str
    label: str
    ok: bool = True
    degraded: bool = False
    note: str = ""
    count: int = 0
    took_ms: int = 0


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    limit: int
    hits: list[Hit]
    sources: list[SourceReport]


def _arxiv_id_from(*values: str) -> str:
    for value in values:
        if not value:
            continue
        for pattern in (_ARXIV_IN_URL, _ARXIV_IN_DOI):
            match = pattern.search(value)
            if match:
                try:
                    return arxiv_tool.normalize_arxiv_id(match.group("id"))
                except arxiv_tool.InvalidArxivId:
                    continue
    return ""


def _one_author(value: str) -> str:
    return " ".join(str(value).split())


def _fetch_json(url: str, *, headers: dict[str, str] | None = None,
                fetch: Fetch | None = None) -> dict:
    raw = fetch(url) if fetch is not None else net.fetch_bytes(
        url, timeout=TIMEOUT_SECONDS, headers=headers
    )
    return json.loads(raw.decode("utf-8", "replace"))


# ---------------------------------------------------------------- arXiv


def _search_arxiv(query: str, limit: int, fetch: Fetch | None) -> tuple[list[Hit], str]:
    result = arxiv_tool.search_arxiv(query, limit=limit, fetch=fetch)
    hits = [
        Hit(
            source="arxiv",
            source_id=entry.arxiv_id,
            title=entry.title,
            authors=entry.authors,
            year=entry.published[:4],
            abstract=entry.abstract,
            pdf_url=entry.pdf_url,
            abs_url=entry.abs_url,
            arxiv_id=entry.arxiv_id,
        )
        for entry in result.results
    ]
    if result.degraded:
        raise RuntimeError(result.note or "arXiv 不可达")
    return hits, result.note


# ---------------------------------------------------------------- OpenAlex


def _search_openalex(query: str, limit: int, fetch: Fetch | None) -> tuple[list[Hit], str]:
    url = (
        f"{OPENALEX_URL}?search={quote(query, safe='')}"
        f"&per-page={limit}&mailto=demo@example.com"
    )
    data = _fetch_json(url, fetch=fetch)
    hits: list[Hit] = []
    for work in data.get("results") or []:
        best = work.get("best_oa_location") or {}
        primary = work.get("primary_location") or {}
        doi = work.get("doi") or ""
        pdf_url = best.get("pdf_url") or primary.get("pdf_url") or ""
        abs_url = primary.get("landing_page_url") or doi
        hits.append(
            Hit(
                source="openalex",
                source_id=(work.get("id") or "").rsplit("/", 1)[-1],
                title=work.get("display_name") or "",
                authors=[
                    _one_author((a.get("author") or {}).get("display_name") or "")
                    for a in (work.get("authorships") or [])
                ],
                year=str(work.get("publication_year") or ""),
                venue=(primary.get("source") or {}).get("display_name") or "",
                pdf_url=pdf_url,
                abs_url=abs_url,
                arxiv_id=_arxiv_id_from(pdf_url, doi),
            )
        )
    return hits, ""


# ---------------------------------------------------------------- Crossref


def _search_crossref(query: str, limit: int, fetch: Fetch | None) -> tuple[list[Hit], str]:
    url = (
        f"{CROSSREF_URL}?query.bibliographic={quote(query, safe='')}"
        f"&rows={limit}&mailto=demo@example.com"
        f"&select=DOI,title,author,issued,container-title,type,link,URL"
    )
    data = _fetch_json(url, fetch=fetch)
    items = ((data.get("message") or {}).get("items")) or []
    hits: list[Hit] = []
    for item in items:
        doi = item.get("DOI") or ""
        pdf_url = ""
        for link in item.get("link") or []:
            if (link.get("content-type") or "").startswith("application/pdf"):
                pdf_url = link.get("URL") or ""
                break
        issued = (item.get("issued") or {}).get("date-parts") or [[""]]
        hits.append(
            Hit(
                source="crossref",
                source_id=doi,
                title=(item.get("title") or [""])[0],
                authors=[
                    _one_author(" ".join(filter(None, [a.get("given"), a.get("family")])))
                    for a in (item.get("author") or [])
                ],
                year=str((issued[0] or [""])[0]),
                venue=(item.get("container-title") or [""])[0],
                pdf_url=pdf_url,
                abs_url=(f"https://doi.org/{doi}" if doi else item.get("URL") or ""),
                arxiv_id=_arxiv_id_from(doi, pdf_url),
            )
        )
    return hits, ""


# ---------------------------------------------------------------- Semantic Scholar


def _search_semanticscholar(
    query: str, limit: int, fetch: Fetch | None
) -> tuple[list[Hit], str]:
    url = (
        f"{S2_URL}?query={quote(query, safe='')}&limit={limit}"
        "&fields=title,year,authors,abstract,externalIds,openAccessPdf,venue"
    )
    headers = {}
    key = _s2_api_key()
    if key:
        headers["x-api-key"] = key
    try:
        data = _fetch_json(url, headers=headers or None, fetch=fetch)
    except OSError as exc:  # 429 在 urllib 里是 HTTPError，属于 OSError
        if "429" in str(exc):
            # 不带 key 也能查，但配额是全站共享的，从机房 IP 出去几乎必然被限流。
            # 这句话是给运维看的，不是给终端用户看的 —— 所以要把"怎么解决"写进去。
            key_note = "" if _s2_api_key() else "：未配 SEMANTIC_SCHOLAR_API_KEY，公共配额已被占满"
            raise RuntimeError(f"限流（429）{key_note}；配 key 后为 1 次/秒") from exc
        raise
    hits: list[Hit] = []
    for paper in data.get("data") or []:
        ids = paper.get("externalIds") or {}
        oa = (paper.get("openAccessPdf") or {}).get("url") or ""
        hits.append(
            Hit(
                source="semanticscholar",
                source_id=paper.get("paperId") or "",
                title=paper.get("title") or "",
                authors=[_one_author((a.get("name") or "")) for a in (paper.get("authors") or [])],
                year=str(paper.get("year") or ""),
                venue=paper.get("venue") or "",
                abstract=paper.get("abstract") or "",
                pdf_url=oa,
                abs_url=(f"https://www.semanticscholar.org/paper/{paper.get('paperId')}"
                         if paper.get("paperId") else ""),
                arxiv_id=_arxiv_id_from(str(ids.get("ArXiv") or ""), oa, str(ids.get("DOI") or "")),
            )
        )
    return hits, ""


def _s2_api_key() -> str:
    import os

    return os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()


# ---------------------------------------------------------------- Zenodo

_TAG_RE = re.compile(r"<[^>]+>")


def _plain(fragment: str, limit: int = 1200) -> str:
    """Zenodo 的 description 是一段 HTML，压成纯文本再进 JSON。

    不压的话，卡片里会直接显示 `<p>` 和 `&nbsp;` —— 用户看到的是源码。
    """
    text = html.unescape(_TAG_RE.sub(" ", fragment or ""))
    return " ".join(text.split())[:limit]


def _search_zenodo(query: str, limit: int, fetch: Fetch | None) -> tuple[list[Hit], str]:
    """Zenodo 收的是数据集、预印本、论文配的代码与材料 —— 别的站基本不收这些。

    PDF 在 `files[]` 里，不是每条记录都有：只有上传了 PDF 的才有得下载。
    取第一条 `.pdf` 类型的文件；没有就只给记录页链接。
    """
    url = f"{ZENODO_URL}?q={quote(query, safe='')}&size={limit}"
    data = _fetch_json(url, fetch=fetch)
    hits: list[Hit] = []
    for record in (data.get("hits") or {}).get("hits") or []:
        meta = record.get("metadata") or {}
        links = record.get("links") or {}
        pdf_url = ""
        for file in record.get("files") or []:
            key = (file.get("key") or "").lower()
            if key.endswith(".pdf") or (file.get("type") or "").lower() == "pdf":
                pdf_url = ((file.get("links") or {}).get("self") or "")
                break
        date = str(meta.get("publication_date") or meta.get("created") or "")
        hits.append(
            Hit(
                source="zenodo",
                source_id=str(record.get("id") or ""),
                title=_one_author(meta.get("title") or ""),
                authors=[
                    _one_author((c.get("name") or ""))
                    for c in (meta.get("creators") or [])
                ],
                year=date[:4],
                venue=(meta.get("resource_type") or {}).get("title") or "Zenodo",
                abstract=_plain(meta.get("description") or ""),
                pdf_url=pdf_url,
                abs_url=links.get("self_html") or links.get("html") or
                        (f"https://zenodo.org/records/{record.get('id')}" if record.get("id") else ""),
                arxiv_id=_arxiv_id_from(pdf_url, str(record.get("doi") or "")),
            )
        )
    return hits, ""


_ADAPTERS = {
    "arxiv": _search_arxiv,
    "openalex": _search_openalex,
    "crossref": _search_crossref,
    "semanticscholar": _search_semanticscholar,
    "zenodo": _search_zenodo,
}


# ---------------------------------------------------------------- 汇总


def _dedupe_keys(hit: Hit) -> list[str]:
    """一篇论文可能有几个"身份"：arXiv id、DOI、抹平写法差异的标题。

    返回的是**全部**身份的集合，只要有任意一个撞上，就是同一篇。
    只用一个 key 会漏：OpenAlex 那条能解出 arXiv id（`arxiv:1706.03762`），
    Crossref 那条只有 DOI（`doi:10.48550/arxiv.1706.03762`），
    各用一个 key 就永远撞不上，同一篇文章会在列表里出现两遍。

    标题兜底是为了那些什么标识都没有的条目；抹掉标点、冠词和大小写差异
    之后相同，基本就是同一篇。
    """
    keys: list[str] = []
    if hit.arxiv_id:
        keys.append("arxiv:" + hit.arxiv_id)
    if hit.source_id:
        if hit.source_id.lower().startswith("10."):
            keys.append("doi:" + hit.source_id.lower())
        elif _ARXIV_IN_URL.search(hit.source_id):
            keys.append("arxiv:" + _arxiv_id_from(hit.source_id))
    title = re.sub(r"[^0-9a-z一-鿿]+", "", hit.title.lower())
    if title:
        keys.append("title:" + title)
    return keys or ["raw:" + hit.source_id + hit.title]


def _merge(first: Hit, later: Hit) -> Hit:
    """保留先出现的那条，把后出现的那条里"先来的没有的"字段补上。

    先出现的来源顺序是固定的（arXiv → OpenAlex → Crossref → S2），
    所以保留下来的元数据是可预期的，不会因为哪个站这次先回包而变化。
    """
    return Hit(
        source=first.source,
        source_id=first.source_id or later.source_id,
        title=first.title or later.title,
        authors=first.authors or later.authors,
        year=first.year or later.year,
        venue=first.venue or later.venue,
        abstract=first.abstract or later.abstract,
        pdf_url=first.pdf_url or later.pdf_url,
        abs_url=first.abs_url or later.abs_url,
        # 前面那条不能入库、后面那条能 —— 补上 id，用户就能少跳一次
        arxiv_id=first.arxiv_id or later.arxiv_id,
    )


def search_sources(
    query: str,
    limit: int = DEFAULT_LIMIT,
    *,
    sources: list[str] | None = None,
    fetch: Fetch | None = None,
) -> SearchResponse:
    """并发查多个来源，各自降级，按来源顺序去重合并。

    `fetch` 是给测试用的注入点：一个函数接管所有出网请求，一台机器上一个包都不发。
    """
    normalized = " ".join(str(query).split())
    limit = max(1, min(int(limit), MAX_LIMIT))
    wanted = [s for s in (sources or list(SOURCES)) if s in _ADAPTERS]
    if not normalized or not wanted:
        return SearchResponse(
            query=normalized,
            limit=limit,
            hits=[],
            sources=[SourceReport(name=s, label=SOURCE_ZH[s], ok=False,
                                  note="没有可用的检索词或来源") for s in wanted],
        )

    def run(name: str) -> tuple[SourceReport, list[Hit]]:
        started = time.monotonic()
        try:
            hits, note = _ADAPTERS[name](normalized, limit, fetch)
            return (
                SourceReport(
                    name=name,
                    label=SOURCE_ZH[name],
                    ok=True,
                    count=len(hits),
                    note=note,
                    took_ms=int((time.monotonic() - started) * 1000),
                ),
                hits,
            )
        except Exception as exc:  # noqa: BLE001 —— 每个站失败的样子都不一样
            return (
                SourceReport(
                    name=name,
                    label=SOURCE_ZH[name],
                    ok=False,
                    degraded=True,
                    note=str(exc) or type(exc).__name__,
                    took_ms=int((time.monotonic() - started) * 1000),
                ),
                [],
            )

    # 并发跑，但只等 DEADLINE_SECONDS。等所有站各自超时叠加起来，
    # 最坏情况要比最长的那一站还长，界面上就是一段看不出在干什么的空白。
    pool = ThreadPoolExecutor(max_workers=len(wanted))
    try:
        futures = {name: pool.submit(run, name) for name in wanted}
        done, _pending = wait(futures.values(), timeout=DEADLINE_SECONDS)
        results = []
        for name in wanted:
            future = futures[name]
            if future in done:
                results.append(future.result())
            else:
                results.append((
                    SourceReport(
                        name=name,
                        label=SOURCE_ZH[name],
                        ok=False,
                        degraded=True,
                        note=f"超过 {DEADLINE_SECONDS} 秒还没返回",
                        took_ms=DEADLINE_SECONDS * 1000,
                    ),
                    [],
                ))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    hits: list[Hit] = []
    seen: dict[str, int] = {}
    for _report, source_hits in results:
        for hit in source_hits:
            keys = _dedupe_keys(hit)
            index = next((seen[k] for k in keys if k in seen), None)
            if index is not None:
                merged = _merge(hits[index], hit)
                hits[index] = merged
                for key in _dedupe_keys(merged):
                    seen.setdefault(key, index)
                continue
            seen_index = len(hits)
            hits.append(hit)
            for key in keys:
                seen.setdefault(key, seen_index)

    return SearchResponse(
        query=normalized,
        limit=limit,
        hits=hits,
        sources=[report for report, _ in results],
    )
