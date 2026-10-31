"""HTTP 层的数据契约。

这里的每个字段都是**前端真的会用到的**，不是把内部对象原样序列化出去：
- 锚点必须带 bbox，否则前端画不出高亮框，溯源就退化成一句"见第 3 页"；
- 被拦下的结论也要返回，界面上要能显示"这条我拒答了"——那才是抗幻觉的证据；
- 证据块带 preview 而不是全文，避免问答响应里塞几十 KB 没人看的文本。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnchorOut(BaseModel):
    page: int
    bbox: list[float] = Field(description="[x0, y0, x1, y1]，PDF 坐标系（pt）")
    source_text: str = Field(description="原文真实片段，前端应优先展示它")
    exact: bool = Field(description="True=逐字命中，False=模糊命中，需人工确认")
    similarity: float
    truncated: bool = False


class ClaimOut(BaseModel):
    text: str
    status: str = Field(description="supported | weak | unsupported")
    reason: str = ""
    anchors: list[AnchorOut] = Field(default_factory=list)


class EvidenceOut(BaseModel):
    rank: int
    section: str
    page_start: int
    page_end: int
    preview: str
    score: float
    source: str


class StatsOut(BaseModel):
    supported: int = 0
    weak: int = 0
    unsupported: int = 0


class AnswerOut(BaseModel):
    question: str
    claims: list[ClaimOut]
    evidence: list[EvidenceOut]
    stats: StatsOut
    took_ms: int
    mode: str = Field(description="answer=走了大模型；retrieve=只返回召回证据")


class PaperOut(BaseModel):
    id: str
    filename: str
    title: str
    pages: int
    chunks: int
    uploaded_at: str
    created_at: str = Field(default="", description="PDF 元数据里的生成日期，可能为空")
    # 只在"刚上传"的响应里填：和库里哪一篇撞了、有多像
    duplicate_of: str | None = None
    duplicate_title: str | None = None
    relation: str | None = Field(default=None, description="identical | new_version | related")
    similarity: float | None = None


class DiffChangeOut(BaseModel):
    kind: str = Field(description="insert | delete | replace")
    section: str
    page: str
    old_pages: list[int]
    new_pages: list[int]
    old: list[str]
    new: list[str]
    reference_churn: bool = Field(description="参考文献编号位移，不是论文修订")


class DiffSummaryItem(BaseModel):
    section: str
    kind: str
    what: str


class CompareOut(BaseModel):
    old: PaperOut
    new: PaperOut
    relation: str
    similarity: float
    stats: dict[str, int]
    changes: list[DiffChangeOut]
    summary: list[DiffSummaryItem]
    summary_ok: bool = Field(description="False 表示大模型不可用，summary 为空不是“没差异”")
    summary_note: str = Field(default="", description="摘要没生成时的原因，直接展示给用户")


class GraphNodeOut(BaseModel):
    id: str
    label: str
    kind: str = Field(description="paper=库里的论文（可点开）| external=只在参考文献里出现")
    year: str
    paper_ids: list[str] = Field(description="库内对应的版本 id，最新版在最后")
    versions: int
    cites: int = Field(description="它引用了库内几篇论文")
    cited_by: int = Field(description="库内几篇论文引用了它")
    rank: float = Field(description="库内 PageRank；外部文献不参与排名，恒为 0")
    x: float = Field(description="归一化坐标 0..1，后端算好，保证每次打开位置一致")
    y: float


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    via: str = Field(description="arxiv=按 arXiv 号认出来的 | title=按标题认出来的")
    score: float


class GraphOut(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]
    stats: dict[str, int]
    notes: list[str] = Field(description="“省略了 N 篇”这类说明，界面要原样显示")


class ChunkOut(BaseModel):
    index: int
    section: str
    page_start: int
    page_end: int
    n_tokens: int
    text: str = Field(description="块全文，阅读器要拿它做高亮；预览请用 preview 字段")
    preview: str
    rects: list[list[float]] = Field(description="该块在原文中的若干 bbox，前端可用于整段高亮")


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=6, ge=1, le=20)
    expand: bool = True
    mode: str = Field(default="answer", pattern="^(answer|retrieve)$")


class BilingualParagraphOut(BaseModel):
    en: str
    zh: str = Field(description="译文；降级或段数对不上时为空串，前端只显示原文")


class BilingualOut(BaseModel):
    chunk: int
    section: str
    page_start: int
    page_end: int
    paragraphs: list[BilingualParagraphOut]
    degraded: bool = Field(
        description="True=大模型不可用或译文段数与原文不一致；原文仍然完整返回"
    )
    model: str


class CopilotOutlineRequest(BaseModel):
    idea: str = Field(min_length=1, max_length=1000)
    max_points: int = Field(default=4, ge=2, le=4, description="每个章节最多几条要点")


class CopilotPointOut(BaseModel):
    text: str
    evidence: list[int] = Field(
        default_factory=list, description="该要点依据的证据块编号；空数组表示没有依据"
    )


class CopilotSectionOut(BaseModel):
    title: str
    points: list[CopilotPointOut]


class CopilotReferenceOut(BaseModel):
    n: int
    paper_id: str
    title: str
    page: int
    quote: str = Field(description="从证据块正文里逐字截取，最多 200 字")


class CopilotOutlineOut(BaseModel):
    idea: str
    outline: list[CopilotSectionOut]
    references: list[CopilotReferenceOut]
    degraded: bool = Field(description="True=大模型不可用，outline 为空但 references 仍是真实文献")
