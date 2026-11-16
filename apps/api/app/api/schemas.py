"""HTTP 层的数据契约。

这里的每个字段都是**前端真的会用到的**，不是把内部对象原样序列化出去：
- 锚点必须带 bbox，否则前端画不出高亮框，溯源就退化成一句"见第 3 页"；
- 被拦下的结论也要返回，界面上要能显示"这条我拒答了"——那才是抗幻觉的证据；
- 证据块带 preview 而不是全文，避免问答响应里塞几十 KB 没人看的文本。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
    # 两级检索的过程量：长文档先按章节粗筛，这两个数字说明"这次只看了多少块"
    corpus_chunks: int = Field(default=0, description="全篇块数")
    candidate_chunks: int = Field(default=0, description="章节粗筛后的候选块数；0 表示未启用粗筛")
    mode: str = Field(description="answer=走了大模型；retrieve=只返回召回证据")
    # 这篇是扫描件：引文是从 OCR 文本里摘的，个别字符可能与纸面不同，
    # 界面上必须说清楚 —— 否则用户会以为论文原文就长这样
    scanned: bool = False


class PaperOut(BaseModel):
    id: str
    filename: str
    title: str
    pages: int
    chunks: int
    uploaded_at: str
    created_at: str = Field(default="", description="PDF 元数据里的生成日期，可能为空")
    arxiv_id: str = Field(default="", description="如 1706.03762（不带版本号）")
    arxiv_version: int = Field(default=0, description="arXiv 的 vN；0 表示取不到，不是 v1")
    arxiv_date: str = Field(
        default="",
        description="首页投稿戳上的日期 YYYY-MM-DD（≠ PDF 生成日期，arXiv GenPDF 不写元数据日期）",
    )
    scanned: bool = Field(default=False, description="首页没有文本层，内容来自 OCR")
    abstract: str = Field(default="", description="摘要正文；认不出就是空串，不编")
    # 首页版面上认出来的作者与机构（赛题模块一要求"作者及机构信息"）。
    # 认不出就是空列表：扫描件首页没有文本层，这里宁可留空也不猜。
    authors: list[str] = Field(default_factory=list, description="首页版面认出的作者")
    affiliations: list[str] = Field(default_factory=list, description="首页版面认出的机构")
    # 只在"刚上传"的响应里填：和库里哪一篇撞了、有多像
    duplicate_of: str | None = None
    duplicate_title: str | None = None
    relation: str | None = Field(default=None, description="identical | new_version | related")
    similarity: float | None = None
    # 谁更新：只有两份 PDF 的生成日期都取得到才敢判断。
    # 空串的含义是"不知道"，不是"一样新" —— 前端不许把空串当成"保留现有"。
    recommendation: str = Field(
        default="", description="keep_existing | replace_existing | '' (无法判断)"
    )
    # 依据要和建议一起回去：前端得照着依据说话。凭版本号判的和凭日期判的
    # 不能共用一句文案 —— v7 顶掉 v1 时说成"日期更晚"，而 v1 恰好是后导出的。
    recommendation_by: str = Field(
        default="", description="判定依据：arxiv_version | date | '' (无法判断)"
    )


class BatchItemOut(BaseModel):
    """批量上传的单条结果：一个文件失败不能拖垮另外九个。

    `ok=false` 时 `error` 一定非空，`paper` 一定为 None —— 前端据此决定
    是画一行绿色还是红色，不需要再判一次状态码。
    """

    filename: str
    ok: bool
    error: str = ""
    skipped: bool = Field(default=False, description="因为 on_duplicate=keep 被跳过")
    paper: PaperOut | None = None


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
    # 新旧是怎么判出来的：arxiv_version | date | uploaded_at。前端按它选措辞，
    # 别把"上传得晚"说成"版本更新"。
    order_by: str = Field(default="", description="arxiv_version | date | uploaded_at")
    # 有一侧是扫描件：字符级差异里混着 OCR 识别的误差，不能当成作者改动
    ocr_side: str = Field(default="", description="old | new | both —— 哪一侧是扫描件")


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
    # 先去掉首尾空白再校验长度：只写空格的问题以前能过 min_length=1，
    # 于是整条链路照跑一遍、返回 200 + 空结论（实测 833 ms 白跑一次）
    model_config = ConfigDict(str_strip_whitespace=True)

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
    model_config = ConfigDict(str_strip_whitespace=True)
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


class TopoNodeOut(BaseModel):
    id: str
    label: str
    kind: str = "module"


class TopoEdgeOut(BaseModel):
    src: str
    dst: str
    label: str = ""


class TopologyOut(BaseModel):
    """架构拓扑图：结构 + 三份可直接粘贴的脚本（DOT / Mermaid / TikZ）。"""

    idea: str
    title: str = ""
    nodes: list[TopoNodeOut] = []
    edges: list[TopoEdgeOut] = []
    dot: str = ""
    mermaid: str = ""
    tikz: str = ""
    degraded: bool = False


class ReferenceOut(BaseModel):
    """一条参考文献。`parsed=False` 的条目照样返回 —— 解析不出来是常态，
    藏起来反而让人以为这篇论文没有参考文献，前端需要显示"这条没解析出标题"。"""

    key: str = Field(description="正文里的引用标记，如 \"12\" 或 \"XDH+19\"")
    raw: str = Field(description="原始条目文本，解析失败时前端只展示这个")
    authors: str = ""
    title: str = Field(default="", description="解析不出就是空串，别编")
    venue: str = ""
    year: str = ""
    arxiv_id: str = ""
    parsed: bool = Field(default=False, description="False=只认出标记，标题没解出来")

    @classmethod
    def of(cls, ref) -> "ReferenceOut":
        return cls(
            key=ref.key,
            raw=ref.raw,
            authors=ref.authors,
            title=ref.title,
            venue=ref.venue,
            year=ref.year,
            arxiv_id=ref.arxiv_id,
            parsed=ref.is_parsed,
        )
