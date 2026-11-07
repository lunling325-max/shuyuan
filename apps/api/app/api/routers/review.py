"""领域综述路由：只做 HTTP 边界，算法全部在 app.rag.review。

为什么响应模型定义在这个文件里，而不是塞进 schemas.py？
schemas.py 是"全站共用契约"，改它要牵动 main.py 的 import 列表；
综述是新增模块，这里自带一套局部模型，挂载时只需要 include 一个 router，
不会让别的功能跟着承担改动风险。

为什么 registry 要在请求时从 deps 取，而不是 import 时绑定？
测试（以及将来多实例部署）会替换掉 deps.registry；模块级绑定会把替换
挡在门外，出现"测试里换成了空库、路由却还在读旧库"的假通过。

降级边界：库空是请求本身无法完成（400）；模型不可用由 rag 层消化成
200 + degraded，这里不做二次包装。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api import deps
from app.rag.review import build_review

router = APIRouter()


class ReviewRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=1000)
    # 不用 ge/le：契约要求越界时"夹住"而不是 422，夹取逻辑在 rag 层
    max_papers: int = Field(default=6, description="最多使用几篇论文，越界会夹到 1..12")


class ReviewPaperOut(BaseModel):
    id: str
    title: str
    year: str


class ReviewPointOut(BaseModel):
    text: str
    evidence: list[int] = Field(
        default_factory=list, description="该要点依据的证据编号；空数组表示没有依据"
    )


class ReviewSectionOut(BaseModel):
    heading: str
    points: list[ReviewPointOut]


class ReviewReferenceOut(BaseModel):
    n: int
    paper_id: str
    title: str
    page: int
    quote: str = Field(description="从证据块正文里逐字截取，压成单行，最多 200 字")
    version: int = Field(0, description="arXiv 的 vN；0 表示取不到版本号")
    scanned: bool = Field(False, description="这一版是扫描件，正文来自 OCR")


class ReviewFutureOut(BaseModel):
    idea: str
    status: str = Field(description="open | solved；solved 表示已被库里其他论文解决")
    evidence: list[int] = Field(default_factory=list)
    why: str = ""


class ReviewOut(BaseModel):
    topic: str
    papers: list[ReviewPaperOut] = Field(description="本次实际用到的论文，按相关度排序")
    sections: list[ReviewSectionOut]
    references: list[ReviewReferenceOut]
    future: list[ReviewFutureOut]
    degraded: bool = Field(
        description="True=模型不可用或输出不合结构；references 仍是真实检索证据"
    )
    note: str = ""


@router.post("/api/review", response_model=ReviewOut)
def review(req: ReviewRequest) -> ReviewOut:
    """基于引用图谱所在文献库，自动生成领域综述并提炼 Future Work。

    检索、编号校验、引用截取、降级都在 app.rag.review；路由层只负责
    "库空是 400"和把内部结构映射成前端契约。
    """
    papers = deps.registry.all()
    if not papers:
        raise HTTPException(400, "文献库是空的，请先上传论文再生成综述")

    result = build_review(req.topic, papers, max_papers=req.max_papers)
    return ReviewOut(
        topic=result.topic,
        papers=[
            ReviewPaperOut(id=p.id, title=p.title, year=p.year) for p in result.papers
        ],
        sections=[
            ReviewSectionOut(
                heading=s.heading,
                points=[
                    ReviewPointOut(text=point.text, evidence=point.evidence)
                    for point in s.points
                ],
            )
            for s in result.sections
        ],
        references=[
            ReviewReferenceOut(
                n=e.n,
                paper_id=e.paper_id,
                title=e.title,
                page=e.page,
                quote=e.quote,
                version=e.version,
                scanned=e.scanned,
            )
            for e in result.references
        ],
        future=[
            ReviewFutureOut(
                idea=f.idea,
                status=f.status,
                evidence=f.evidence,
                why=f.why,
            )
            for f in result.future
        ],
        degraded=result.degraded,
        note=result.note,
    )
