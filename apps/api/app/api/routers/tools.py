"""工具路由：arXiv 检索与"检索 → 下载 → 入库"闭环。

为什么这样划边界：

1. **检索失败不能 500**。断网演示时页面还得继续工作，所以网络失败统一
   折叠成 200 + `degraded=true` + `results=[]` + 一句 `note`；前端据此显示
   "暂时连不上 arXiv"，而不是弹一个红色报错。
2. **入库复用现成的 registry**。内容哈希去重、切块、版本相似度判定都在
   `app.api.registry` 里，工具端点只负责把下载到的字节喂进去 ——
   自己再建一个注册表就会产生第二份解析缓存，同一篇论文被解析两遍。
3. **状态码表达失败的类型**：id 非法和非 PDF 是客户端问题（400），
   下载失败是上游问题（502），解析不出正文是内容问题（422）。
   这样前端才能区分"改个输入"和"等会儿再试"。
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


class ArxivIngestRequest(BaseModel):
    arxiv_id: str


class ArxivIngestOut(BaseModel):
    paper: PaperOut
    duplicate_of: str | None = None
    fetched_bytes: int


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


@router.post(
    "/api/tools/arxiv/ingest",
    response_model=ArxivIngestOut,
    status_code=201,
)
def arxiv_ingest(req: ArxivIngestRequest) -> ArxivIngestOut:
    try:
        arxiv_id = normalize_arxiv_id(req.arxiv_id)
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
        raise HTTPException(422, "这份 PDF 里没有抽到可用的正文（可能是纯扫描件）")

    duplicate = deps.registry.find_similar(paper)
    return ArxivIngestOut(
        paper=deps.paper_out(paper, duplicate),
        duplicate_of=duplicate[0].id if duplicate is not None else None,
        fetched_bytes=len(data),
    )
