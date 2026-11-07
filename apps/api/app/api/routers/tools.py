"""工具路由：多来源文献检索 + "检索 → 下载 → 入库"闭环。

为什么这样划边界：

1. **检索失败不能 500**。断网演示时页面还得继续工作，所以网络失败统一
   折叠成 200 + 每站各自的 `degraded/note` + 空列表；前端据此显示
   "暂时连不上某站"，而不是弹一个红色报错。一个站挂了，其余站照常返回。
2. **入库复用现成的 registry**。内容哈希去重、切块、版本相似度判定都在
   `app.api.registry` 里，工具端点只负责把下载到的字节喂进去 ——
   自己再建一个注册表就会产生第二份解析缓存，同一篇论文被解析两遍。
3. **状态码表达失败的类型**：id 非法和非 PDF 是客户端问题（400），
   下载失败是上游问题（502），解析不出正文是内容问题（422）。
   这样前端才能区分"改个输入"和"等会儿再试"。
4. **入库通道仍然只有 arXiv**。多来源检索里，只有能定位到 arXiv id 的条目
   才带「入库」按钮；其余给原文链接。假装 OpenAlex 的任意 PDF 都能入库，
   用户点下去只会收到一个 400。
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api import deps
from app.api.schemas import PaperOut
from app.tools.arxiv import (
    FetchError,
    InvalidArxivId,
    NotPdfError,
    PdfTooLargeError,
    download_pdf,
    normalize_arxiv_id,
    search_arxiv,
)
from app.tools.sources import DEFAULT_LIMIT, MAX_LIMIT, SOURCE_ZH, search_sources

router = APIRouter()


class ArxivSearchResultOut(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published: str
    abstract: str
    pdf_url: str
    abs_url: str


class ArxivSearchOut(BaseModel):
    query: str
    results: list[ArxivSearchResultOut] = Field(default_factory=list)
    degraded: bool = False
    note: str = ""


class SearchHitOut(BaseModel):
    source: str
    source_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    venue: str = ""
    abstract: str = ""
    pdf_url: str = ""
    abs_url: str = ""
    arxiv_id: str = ""          # 非空 = 这篇能走 arXiv 通道入库
    can_ingest: bool = False


class SourceReportOut(BaseModel):
    name: str
    label: str
    ok: bool = True
    degraded: bool = False
    note: str = ""
    count: int = 0
    took_ms: int = 0


class SearchOut(BaseModel):
    query: str
    limit: int
    hits: list[SearchHitOut] = Field(default_factory=list)
    sources: list[SourceReportOut] = Field(default_factory=list)
    degraded: bool = False
    note: str = ""


class ArxivIngestRequest(BaseModel):
    arxiv_id: str


class ArxivIngestOut(BaseModel):
    paper: PaperOut
    duplicate_of: str | None = None
    fetched_bytes: int
    # 库里已经有这篇了，这次一个字节都没下载
    already_present: bool = False


@router.get("/api/tools/arxiv/search", response_model=ArxivSearchOut)
def arxiv_search(q: str, limit: int = 5) -> ArxivSearchOut:
    # 不用 Query(ge=1, le=20)：那会把越界变成 422。前端已经在传各种 limit，
    # 夹到合法区间比拒收更符合演示时"别中断"的目标。
    clamped = max(1, min(limit, 20))
    result = search_arxiv(q, limit=clamped)
    return ArxivSearchOut(
        query=result.query,
        results=[ArxivSearchResultOut(**asdict(entry)) for entry in result.results],
        degraded=result.degraded,
        note=result.note,
    )


@router.get("/api/tools/search", response_model=SearchOut)
def multi_search(q: str, sources: str = "", limit: int = DEFAULT_LIMIT) -> SearchOut:
    """一次查询，并发问 arXiv / OpenAlex / Crossref / Semantic Scholar。

    `sources` 是逗号分隔的站名，留空就查全部；认不出的名字直接忽略
    （演示时不值得为拼错一个站名返回 422）。`limit` 是**每个站**取多少条，
    不是总条数 —— 去重之后总数通常少于各站之和，这是正常的。

    没有"全部"：任何来源都有深不见底的翻页，承诺"全部"就是承诺一个
    永远不会到来的响应。25 条/站是本机实测能在 5 秒内出结果的上限。
    """
    # 认不出的站名过滤掉；一个都不认识就退回"查全部"，而不是返回空列表
    wanted = [s.strip() for s in sources.split(",") if s.strip() in SOURCE_ZH] or None
    result = search_sources(
        q, limit=max(1, min(limit, MAX_LIMIT)), sources=wanted
    )
    failed = [s for s in result.sources if not s.ok]
    return SearchOut(
        query=result.query,
        limit=result.limit,
        hits=[
            SearchHitOut(**asdict(hit), can_ingest=hit.can_ingest)
            for hit in result.hits
        ],
        sources=[SourceReportOut(**asdict(report)) for report in result.sources],
        degraded=bool(failed) and bool(result.hits),
        note=(
            "，".join(f"{s.label} 暂不可用（{s.note}）" for s in failed)
            if failed else ""
        ),
    )


@router.post(
    "/api/tools/arxiv/ingest",
    response_model=ArxivIngestOut,
    status_code=201,
)
def arxiv_ingest(req: ArxivIngestRequest) -> ArxivIngestOut:
    """下载一篇 arXiv PDF 并入库。

    **先查库再下载。** 以前是先无条件下载，再靠内容哈希认出"这篇已经在库里"：
    界面上是"点了一下、转 88 秒、然后什么都没变"（实测这次 87.8 秒全花在
    下载同一份 557KB 的文件上）。arXiv 号是精确标识，比对一次是毫秒级的。
    """
    try:
        arxiv_id = normalize_arxiv_id(req.arxiv_id)
    except InvalidArxivId as exc:
        raise HTTPException(400, str(exc)) from exc

    for paper in deps.registry.all():
        if paper.arxiv_id == arxiv_id:
            return ArxivIngestOut(
                paper=deps.paper_out(paper, None),
                already_present=True,
                fetched_bytes=0,
            )

    try:
        data = download_pdf(arxiv_id)
    except InvalidArxivId as exc:
        raise HTTPException(400, str(exc)) from exc
    except NotPdfError as exc:
        raise HTTPException(400, str(exc)) from exc
    except PdfTooLargeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FetchError as exc:
        raise HTTPException(502, str(exc)) from exc

    # 前端展示与后续引用都认不带版本号的文件名；registry 会用内容哈希覆盖 id
    try:
        paper = deps.registry.add(data, f"{arxiv_id}.pdf")
    except Exception as exc:  # noqa: BLE001 —— 解析器对坏 PDF 的异常类型不可枚举
        raise HTTPException(422, f"PDF 解析失败：{exc}") from exc

    if not paper.chunks:
        raise HTTPException(422, "该 PDF 未能抽取到可用正文（可能为纯扫描件）")

    duplicate = deps.registry.find_similar(paper)
    return ArxivIngestOut(
        paper=deps.paper_out(paper, duplicate),
        duplicate_of=duplicate[0].id if duplicate is not None else None,
        fetched_bytes=len(data),
    )
