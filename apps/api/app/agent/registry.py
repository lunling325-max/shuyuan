"""工具注册表：把项目里已有的能力包装成模型可选、服务端可校验的"工具"。

为什么不让 planner 直接 import 一堆函数、再写一长串 if/elif？
1. **校验必须集中**。模型给出的工具名和参数是**不可信输入**，未知工具、
   缺少参数、类型不对都必须在这里被拦下，而不是等执行到一半才抛异常。
2. **前端要看到"这个 Agent 会什么"**。每个工具自带中文描述与参数 schema，
   `/api/agent/tools` 直接序列化出去，不需要前端再维护一份能力清单。
3. **复用已有能力，不另起炉灶**。检索、问答校验、版本 diff、引用图谱、
   arXiv 检索全部调用现有模块；这一层只做"参数 -> 调用 -> 观测摘要"的适配。

观测摘要（`ToolOutcome.summary`）是有意截断的：它既要喂回模型继续决策，
也要放进 trace 给前端展示。完整结构化数据放在 `data` 里，摘要只保留
模型下一步真正需要看的部分，避免上下文被一篇论文的全文撑爆。

工具抛异常**不往上抛**：它会被折叠成 `ok=False` 的观测。Agent 的价值在于
"某一步失败了还能继续"，而不是一个坏索引把整次运行打挂。真正的
"库为空"属于请求无法完成，由路由层在调用 Agent 之前返回 400。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.citegraph import build_graph
from app.rag.answer import answer_question
from app.tools.arxiv import search_arxiv as search_arxiv_tool
from app.versioning import classify, diff_versions, jaccard
from app.versioning_summary import DiffSummary, summarize_diff

if TYPE_CHECKING:  # 只为类型标注，工具层不要在 import 时绑定具体注册表实现
    from app.api.registry import Paper

MAX_QUOTE_CHARS = 200
MAX_OBSERVATION_CHARS = 1200
MAX_REASON_CHARS = 240


@dataclass(frozen=True, slots=True)
class ObservedEvidence:
    """本轮工具真正看到过的证据块。

    `quote` 由服务端从 chunk 正文逐字截取，模型只能给 (paper_id, chunk_index)
    指针。这和 `app.rag.review` 是同一条铁律：引用文字不能由模型生成。
    """

    paper_id: str
    title: str
    chunk_index: int
    page: int
    section: str
    text: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.paper_id, self.chunk_index)

    @property
    def quote(self) -> str:
        flat = " ".join(self.text.split())
        return flat[:MAX_QUOTE_CHARS]

    def pointer(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "quote": self.quote,
        }


@dataclass(slots=True)
class ToolOutcome:
    """一次工具调用的结果：成功与否、给人/模型看的摘要、结构化数据、证据。"""

    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    evidence: list[ObservedEvidence] = field(default_factory=list)
    dropped_reason: str = ""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[["ToolContext", BaseModel], ToolOutcome]
    read_only: bool = True


@dataclass
class ToolContext:
    """一次 Agent 运行共享的上下文。

    只持有论文列表和可注入的函数，不持有全局 registry —— 测试可以直接
    构造 SimpleNamespace 假论文，不需要先把 PDF 落盘。
    """

    papers: list["Paper"]
    llm_call: Callable[..., dict] | None = None
    arxiv_fetch: Callable[[str], bytes] | None = None
    _by_id: dict[str, "Paper"] = field(init=False, repr=False)
    _graph: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.papers = list(self.papers)
        self._by_id = {paper.id: paper for paper in self.papers}

    def paper(self, paper_id: str) -> "Paper":
        try:
            return self._by_id[paper_id]
        except KeyError as exc:
            raise ValueError(f"库内不存在该文献：{paper_id}") from exc

    def graph(self):
        """整个运行只建一次图。建图要解析每篇的参考文献，重复调用很贵。"""
        if self._graph is None:
            self._graph = build_graph(self.papers)
        return self._graph


class _StrictArgs(BaseModel):
    """参数模型统一禁止多余字段。

    模型经常"顺手"补一个自己臆想的参数（比如给 search_library 加 `year`）。
    Pydantic 默认会忽略它，看似宽容，实际会让模型误以为这个参数生效了。
    这里选择拒绝，让 planner 把整步丢弃并在 notes 里写清原因。

    `strict=True` 同样是刻意选择：默认的 lax 模式会把 `"3"` 自动转成 3，
    但工具参数是模型生成的不可信输入，偷偷强转等于替模型"圆谎"。
    """

    model_config = ConfigDict(extra="forbid", strict=True)


class SearchLibraryArgs(_StrictArgs):
    query: str = Field(min_length=1, max_length=500, description="全库 BM25 检索式")
    k: int = Field(default=5, ge=1, le=10, description="最多返回几个证据块")


class GetPaperArgs(_StrictArgs):
    paper_id: str = Field(min_length=1, max_length=200, description="文献 id")


class AnswerQuestionArgs(_StrictArgs):
    paper_id: str = Field(min_length=1, max_length=200, description="文献 id")
    question: str = Field(min_length=1, max_length=500, description="针对这篇论文的问题")
    k: int = Field(default=6, ge=1, le=20, description="检索证据块数")


class CompareVersionsArgs(_StrictArgs):
    paper_a: str = Field(min_length=1, max_length=200, description="第一篇文献 id")
    paper_b: str = Field(min_length=1, max_length=200, description="第二篇文献 id")
    summarize: bool = Field(default=True, description="是否让模型把差异总结成人话")


class SearchArxivArgs(_StrictArgs):
    query: str = Field(min_length=1, max_length=200, description="arXiv 标题/关键词")
    limit: int = Field(default=5, ge=1, le=10, description="最多返回几条")


class CitationGraphArgs(_StrictArgs):
    max_nodes: int = Field(default=12, ge=1, le=30, description="返回 PageRank 最高的几篇库内论文")


class FindMainPapersArgs(_StrictArgs):
    top: int = Field(default=3, ge=1, le=10, description="返回几篇基石论文")


class ListPapersArgs(_StrictArgs):
    pass


class ScanPdfArgs(_StrictArgs):
    paper_id: str = Field(min_length=1, max_length=200, description="文献 id")


class FindTablesArgs(_StrictArgs):
    paper_id: str = Field(min_length=1, max_length=200, description="文献 id")
    keyword: str = Field(
        default="",
        max_length=100,
        description="可选：只保留含这个词的行（模型名、指标名、数据集名）；留空则列出全部表格",
    )


class ToolRegistry:
    """工具名 -> ToolSpec。注册顺序就是 `/api/agent/tools` 的展示顺序。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        # 同名覆盖而不是报错：测试要用坏 handler 替换真实工具，模拟执行异常
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.args_model.model_json_schema(),
                "read_only": spec.read_only,
            }
            for spec in self._tools.values()
        ]

    def validate(
        self, name: object, raw_args: object
    ) -> tuple[ToolSpec | None, BaseModel | None, str]:
        """校验工具名与参数。返回 (spec, 解析后的参数, 丢弃原因)。

        原因串既写进 notes 也写进 trace 的 dropped_reason —— 这是给评委看
        "Agent 确实拦下了坏步骤"的证据，不能悄悄吞掉。
        """
        if not isinstance(name, str) or not name.strip():
            return None, None, "工具名必须是非空字符串"
        spec = self._tools.get(name)
        if spec is None:
            return None, None, f"未知工具：{name}"

        args = {} if raw_args is None else raw_args
        if not isinstance(args, dict):
            return spec, None, "参数必须是 JSON 对象"
        try:
            parsed = spec.args_model.model_validate(args)
        except ValidationError as exc:
            return spec, None, _format_validation_error(exc)
        return spec, parsed, ""

    def execute(self, name: object, raw_args: object, ctx: ToolContext) -> ToolOutcome:
        spec, parsed, reason = self.validate(name, raw_args)
        if reason:
            return ToolOutcome(
                ok=False,
                summary=f"步骤被丢弃：{reason}",
                dropped_reason=reason,
            )
        assert spec is not None and parsed is not None
        try:
            outcome = spec.handler(ctx, parsed)
        except Exception as exc:  # noqa: BLE001 —— 工具的异常类型不可枚举
            detail = _one_line(exc)
            return ToolOutcome(
                ok=False,
                summary=f"工具执行失败：{detail}",
                data={"error": detail},
            )
        if not isinstance(outcome, ToolOutcome):
            return ToolOutcome(ok=False, summary="工具返回了无法识别的结果")
        outcome.summary = truncate(outcome.summary, MAX_OBSERVATION_CHARS)
        return outcome


# ---------------------------------------------------------------- 通用小工具


def truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _one_line(value: object) -> str:
    return truncate(str(value) or type(value).__name__, MAX_REASON_CHARS)


def _format_validation_error(exc: ValidationError) -> str:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
    msg = str(first.get("msg") or "参数不合法")
    return truncate(f"参数 {loc} 不合法：{msg}", MAX_REASON_CHARS)


def _paper_year(paper: "Paper") -> str:
    created = (getattr(paper, "created_at", "") or "").strip()
    if len(created) >= 4 and created[:4].isdigit():
        return created[:4]
    match = re.match(r"^(\d{2})(0[1-9]|1[0-2])\.", getattr(paper, "arxiv_id", "") or "")
    if match:
        yy = int(match.group(1))
        return str(2000 + yy if yy < 90 else 1900 + yy)
    uploaded = (getattr(paper, "uploaded_at", "") or "").strip()
    return uploaded[:4] if len(uploaded) >= 4 and uploaded[:4].isdigit() else ""


def _evidence_from_chunk(paper: "Paper", chunk) -> ObservedEvidence:
    return ObservedEvidence(
        paper_id=paper.id,
        title=paper.title,
        chunk_index=chunk.index,
        page=chunk.page_start,
        section=chunk.section or "",
        text=chunk.text,
    )


def _relation(paper_a: "Paper", paper_b: "Paper"):
    """指纹可能因为假论文缺字段而失败；版本关系是增强信息，失败就按 distinct。"""
    try:
        return classify(jaccard(paper_a.fingerprint, paper_b.fingerprint))
    except Exception:  # noqa: BLE001
        return classify(0.0)


# ---------------------------------------------------------------- 工具实现


def _search_library(ctx: ToolContext, args: SearchLibraryArgs) -> ToolOutcome:
    """全库 BM25：跨所有论文检索，再按统一键排序。

    排序键写死成 `(-score, paper_id, chunk_index)`。同分时不能依赖论文在
    列表里的顺序 —— 上传时间只精确到秒，两次运行可能给出不同结果，
    前端回链就会跳到别的段落。
    """
    scored: list[tuple[float, "Paper", Any]] = []
    for paper in ctx.papers:
        if not getattr(paper, "chunks", None):
            continue
        try:
            hits = paper.bm25.search(args.query, k=args.k)
        except Exception:  # noqa: BLE001 —— 单篇索引坏了不该拖垮全库检索
            continue
        for hit in hits:
            scored.append((float(hit.score), paper, hit.chunk))
    scored.sort(key=lambda item: (-item[0], item[1].id, item[2].index))
    picked = scored[: args.k]

    evidence: list[ObservedEvidence] = []
    results: list[dict[str, Any]] = []
    for score, paper, chunk in picked:
        observed = _evidence_from_chunk(paper, chunk)
        evidence.append(observed)
        results.append(
            {
                "paper_id": paper.id,
                "title": paper.title,
                "chunk_index": chunk.index,
                "page": chunk.page_start,
                "section": chunk.section or "",
                "score": round(score, 4),
                "quote": observed.quote,
            }
        )

    lines = [f"全库检索「{args.query}」命中 {len(results)} 块："]
    for item in results:
        lines.append(
            f"[{item['paper_id']} #{item['chunk_index']}] p.{item['page']} "
            f"score={item['score']} · {item['quote'][:120]}"
        )
    if not results:
        lines.append("没有命中。可更换英文关键词，或通过 list_papers 查看库内标题。")
    return ToolOutcome(
        ok=True,
        summary="\n".join(lines),
        data={"query": args.query, "results": results},
        evidence=evidence,
    )


def _get_paper(ctx: ToolContext, args: GetPaperArgs) -> ToolOutcome:
    paper = ctx.paper(args.paper_id)

    # "是否多版本"必须按正文指纹判断，不能看文件名：v1/v7 的命名是人为的
    versions: list[dict[str, Any]] = []
    for other in ctx.papers:
        if other.id == paper.id:
            continue
        relation = _relation(paper, other)
        if relation.is_duplicate:
            versions.append(
                {
                    "paper_id": other.id,
                    "title": other.title,
                    "relation": relation.kind,
                    "similarity": relation.similarity,
                }
            )
    versions.sort(key=lambda item: (-item["similarity"], item["paper_id"]))

    data = {
        "paper_id": paper.id,
        "title": paper.title,
        "year": _paper_year(paper),
        "pages": paper.pages,
        "chunks": len(paper.chunks),
        "filename": paper.filename,
        "multi_version": bool(versions),
        "versions": versions,
    }
    version_text = (
        f"多版本（{len(versions)} 个相关版本）" if versions else "单版本"
    )
    summary = (
        f"《{paper.title}》 id={paper.id} 年份={data['year'] or '未知'} "
        f"页数={paper.pages} 块数={len(paper.chunks)} 版本情况={version_text}"
    )
    if versions:
        summary += "\n相关版本：" + "、".join(
            f"{v['paper_id']}({v['relation']}, {v['similarity']})" for v in versions
        )
    return ToolOutcome(ok=True, summary=summary, data=data)


def _answer_question(ctx: ToolContext, args: AnswerQuestionArgs) -> ToolOutcome:
    paper = ctx.paper(args.paper_id)
    result = answer_question(
        args.question,
        retriever=paper.bm25,
        index=paper.text_index,
        doc_id=paper.id,
        k=args.k,
        llm_call=ctx.llm_call,
    )

    evidence = [_evidence_from_chunk(paper, hit.chunk) for hit in result.evidence]
    claims = [
        {
            "text": claim.claim.text,
            "status": claim.status,
            "reason": claim.reason,
            "pages": sorted({anchor.page for anchor in claim.anchors}),
        }
        for claim in result.claims
    ]
    stats = result.stats
    lines = [
        f"《{paper.title}》问答：留下 {len(result.kept)} 条结论，"
        f"拦截 {stats.get('unsupported', 0)} 条，证据 {len(evidence)} 块。"
    ]
    for claim in claims[:6]:
        if claim["status"] == "unsupported":
            # 被拦下的也展示：这是抗幻觉校验真的生效了，不是空白
            lines.append(f"- [拦截] {claim['text']}（{claim['reason']}）")
        else:
            pages = "".join(f"〔p.{p}〕" for p in claim["pages"])
            lines.append(f"- [{claim['status']}] {claim['text']}{pages}")
    for item in evidence[:6]:
        lines.append(f"证据：[{item.paper_id} #{item.chunk_index}] p.{item.page}")
    return ToolOutcome(
        ok=True,
        summary="\n".join(lines),
        data={
            "paper_id": paper.id,
            "question": args.question,
            "claims": claims,
            "stats": stats,
        },
        evidence=evidence,
    )


def _compare_versions(ctx: ToolContext, args: CompareVersionsArgs) -> ToolOutcome:
    paper_a = ctx.paper(args.paper_a)
    paper_b = ctx.paper(args.paper_b)
    if paper_a.id == paper_b.id:
        raise ValueError("两份是同一个文件，没什么可比的")

    # 谁新谁旧和 HTTP 接口共用同一份判断（arXiv 版本号 → 生成日期 → 上传时间）。
    # 以前这里自己写了一遍"只看日期"，扫描件在这条路上会被判成更新的那一版。
    # 局部 import：工具层不在模块加载时绑定具体注册表实现（见文件头的 TYPE_CHECKING）。
    from app.api.registry import order_versions

    old, new, order_by = order_versions(paper_a, paper_b)
    relation = _relation(old, new)
    diff = diff_versions(old.path, new.path)
    ocr_side = (
        "both" if old.scanned and new.scanned
        else "old" if old.scanned
        else "new" if new.scanned
        else ""
    )
    summary = (
        summarize_diff(diff, llm_call=ctx.llm_call, ocr_side=ocr_side)
        if args.summarize
        else DiffSummary(items=[], ok=True)
    )

    data = {
        "old": {"paper_id": old.id, "title": old.title},
        "new": {"paper_id": new.id, "title": new.title},
        "relation": relation.kind,
        "similarity": relation.similarity,
        "stats": diff.stats,
        "summary": summary.items,
        "summary_ok": summary.ok,
        "summary_note": summary.reason,
        "order_by": order_by,
        "ocr_side": ocr_side,
    }
    lines = [
        f"《{new.title}》 vs 《{old.title}》：relation={relation.kind}，"
        f"差异 {len(diff.changes)} 处（{diff.stats}）。"
    ]
    if order_by == "uploaded_at":
        lines.append("注意：两版的 arXiv 版本号与生成日期都取不到，新旧是按上传时间排的，不代表论文本身的先后。")
    if ocr_side:
        lines.append("注意：其中一侧是扫描件，逐字差异里混着 OCR 识别误差。")
    if summary.items:
        lines.append("摘要：" + "；".join(item["what"] for item in summary.items[:6]))
    elif not summary.ok:
        lines.append(f"摘要未生成：{summary.reason}")
    return ToolOutcome(ok=True, summary="\n".join(lines), data=data)


def _search_arxiv(ctx: ToolContext, args: SearchArxivArgs) -> ToolOutcome:
    result = search_arxiv_tool(args.query, limit=args.limit, fetch=ctx.arxiv_fetch)
    entries = [
        {
            "arxiv_id": entry.arxiv_id,
            "title": entry.title,
            "authors": list(entry.authors),
            "published": entry.published,
            "abstract": entry.abstract,
            "pdf_url": entry.pdf_url,
            "abs_url": entry.abs_url,
        }
        for entry in result.results
    ]
    lines = [f"arXiv 检索「{result.query}」：{len(entries)} 条，degraded={result.degraded}。"]
    for entry in entries:
        lines.append(f"- {entry['arxiv_id']} | {entry['title']} | {entry['published']}")
    if result.note:
        lines.append(f"说明：{result.note}")
    # 只读检索，不下载、不入库；断网也只是 data.degraded=True，不当作工具失败
    return ToolOutcome(
        ok=True,
        summary="\n".join(lines),
        data={
            "query": result.query,
            "results": entries,
            "degraded": result.degraded,
            "note": result.note,
        },
    )


def _paper_rank_row(node) -> dict[str, Any]:
    return {
        "id": node.id,
        "label": node.label,
        "year": node.year,
        "paper_ids": list(node.paper_ids),
        "versions": node.versions,
        "cites": node.cites,
        "cited_by": node.cited_by,
        "rank": node.rank,
    }


def _citation_graph(ctx: ToolContext, args: CitationGraphArgs) -> ToolOutcome:
    graph = ctx.graph()
    papers = sorted(
        (node for node in graph.nodes if node.kind == "paper"),
        key=lambda node: (-node.rank, node.id),
    )
    picked = papers[: args.max_nodes]
    data = {
        "stats": dict(graph.stats),
        "notes": list(graph.notes),
        "nodes": [_paper_rank_row(node) for node in picked],
    }
    lines = [
        f"引用图谱：{graph.stats.get('papers', 0)} 篇论文、"
        f"{graph.stats.get('nodes', 0)} 个节点、{graph.stats.get('edges', 0)} 条边。"
    ]
    for i, node in enumerate(picked, start=1):
        lines.append(
            f"{i}. {node.label}（rank={node.rank}, 被引 {node.cited_by} 次, "
            f"ids={'/'.join(node.paper_ids)}）"
        )
    return ToolOutcome(ok=True, summary="\n".join(lines), data=data)


def _find_main_papers(ctx: ToolContext, args: FindMainPapersArgs) -> ToolOutcome:
    graph = ctx.graph()
    papers = sorted(
        (node for node in graph.nodes if node.kind == "paper"),
        key=lambda node: (-node.rank, node.id),
    )[: args.top]
    data = {"papers": [_paper_rank_row(node) for node in papers]}
    if not papers:
        summary = "引用图谱里没有库内论文。"
    else:
        summary = "基石论文（按库内 PageRank）：\n" + "\n".join(
            f"{i}. {node.label}（rank={node.rank}, 被引 {node.cited_by} 次）"
            for i, node in enumerate(papers, start=1)
        )
    return ToolOutcome(ok=True, summary=summary, data=data)


def _list_papers(ctx: ToolContext, args: ListPapersArgs) -> ToolOutcome:
    papers = sorted(ctx.papers, key=lambda paper: paper.id)
    rows = [
        {
            "paper_id": paper.id,
            "title": paper.title,
            "year": _paper_year(paper),
            "pages": paper.pages,
            "chunks": len(paper.chunks),
            "filename": paper.filename,
        }
        for paper in papers
    ]
    lines = [f"库内共 {len(rows)} 篇："]
    for row in rows:
        lines.append(
            f"- {row['paper_id']} | {row['title']} | {row['year'] or '年份未知'} | "
            f"{row['pages']} 页 / {row['chunks']} 块"
        )
    return ToolOutcome(ok=True, summary="\n".join(lines), data={"papers": rows})


def _scan_pdf(ctx: ToolContext, args: ScanPdfArgs) -> ToolOutcome:
    from dataclasses import asdict

    from app.parsing.scan import analyze

    paper = ctx.paper(args.paper_id)
    report = analyze(str(paper.path))
    data = asdict(report)
    scanned = data.get("scanned_pages") or []
    summary = (
        f"《{paper.title}》文字层体检：{data.get('readable_pages', 0)}/"
        f"{data.get('pages', 0)} 页可读，扫描页 {len(scanned)} 页，"
        f"OCR 可用={'是' if data.get('ocr_available') else '否'}。{data.get('note', '')}"
    )
    return ToolOutcome(ok=True, summary=summary, data=data)


MAX_TABLE_TOOL_ROWS = 40
MAX_TABLE_TOOL_TABLES = 5


def _row_line(row) -> str:
    """格内换行（booktabs 表体挤在一格）在一行摘要里用 ` / ` 顶掉。"""
    return " | ".join(cell.replace("\n", " / ") for cell in row)


def _find_tables(ctx: ToolContext, args: FindTablesArgs) -> ToolOutcome:
    """读一篇论文里**行×列还原过**的表格。

    这一步的意义不在"多一个工具"：切块器读到的表格是被拆散的行 ——
    `BLEU` 在表头、`41.8` 在第七行，两者之间没有任何关系。这里给的是
    还原后的格子，数字和列名在同一行里，模型不用猜。
    扫描件提不出来（没有文字层），`report.note` 里会写明原因，不编。
    """
    paper = ctx.paper(args.paper_id)
    report = paper.visuals
    needle = args.keyword.strip().lower()

    picked: list[tuple[object, list]] = []
    for table in report.tables:
        if needle:
            body = [row for row in table.rows if any(needle in cell.lower() for cell in row)]
            if not body:
                continue
            # 表头必须跟着一起给：只给 `GPT-3 175B | 175.0B | 96`，谁也看不出 96 是层数
            header = tuple(table.rows[:1])
            picked.append((table, list(header) + [row for row in body if row not in header]))
        else:
            picked.append((table, list(table.rows)))

    total = len(report.tables)
    if not picked:
        scope = f"关键词「{args.keyword}」没有匹配到" if needle else "没有可提取的"
        return ToolOutcome(
            ok=True,
            summary=f"《{paper.title}》{scope}表格。{report.note}",
            data={"tables": [], "count": total, "matched": 0, "note": report.note},
        )

    lines = [f"《{paper.title}》共 {total} 张表格，命中 {len(picked)} 张（已按行×列还原）。"]
    budget = MAX_TABLE_TOOL_ROWS
    for index, (table, rows) in enumerate(picked[:MAX_TABLE_TOOL_TABLES]):
        head = f"- {table.label or '（无表题）'} · p.{table.page} · {table.n_rows}×{table.n_cols}"
        if table.caption:
            head += f" · {truncate(table.caption, 80)}"
        lines.append(head)
        for row in rows[:budget]:
            lines.append("  " + _row_line(row))
        if len(rows) > budget:
            lines.append(f"  …另有 {len(rows) - budget} 行")
        budget = max(0, budget - len(rows))
        if budget == 0:
            remaining = len(picked[:MAX_TABLE_TOOL_TABLES]) - index - 1
            if remaining:
                lines.append(f"（限于篇幅，另有 {remaining} 张表格未列出）")
            break

    data = {
        "tables": [
            {
                "label": table.label,
                "page": table.page,
                "caption": table.caption,
                "rows": [list(row) for row in rows],
            }
            for table, rows in picked[:MAX_TABLE_TOOL_TABLES]
        ],
        "count": total,
        "matched": len(picked),
        "note": report.note,
    }
    return ToolOutcome(ok=True, summary="\n".join(lines), data=data)


# ---------------------------------------------------------------- 装配


def build_tool_registry() -> ToolRegistry:
    """按展示顺序装配工具。

    顺序固定，`/api/agent/tools` 与提示词里的工具目录因此完全一致，
    模型看到的能力清单不会因为字典遍历顺序变化而漂移。
    """
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="search_library",
            description="在全部已上传论文里做 BM25 关键词检索，返回带 paper_id 与 chunk_index 的证据块。",
            args_model=SearchLibraryArgs,
            handler=_search_library,
        )
    )
    registry.register(
        ToolSpec(
            name="get_paper",
            description="查看一篇论文的元信息：标题、年份、页数、块数，以及库内是否存在该论文的其他版本。",
            args_model=GetPaperArgs,
            handler=_get_paper,
        )
    )
    registry.register(
        ToolSpec(
            name="answer_question",
            description="针对某一篇论文做带抗幻觉校验的问答，结论必须能逐字回溯到原文。",
            args_model=AnswerQuestionArgs,
            handler=_answer_question,
        )
    )
    registry.register(
        ToolSpec(
            name="compare_versions",
            description="对比同一篇论文的两个版本，给出句级差异统计和可选的中文摘要。",
            args_model=CompareVersionsArgs,
            handler=_compare_versions,
        )
    )
    registry.register(
        ToolSpec(
            name="search_arxiv",
            description="只读检索 arXiv 的标题与摘要；不会下载或写入资料库。",
            args_model=SearchArxivArgs,
            handler=_search_arxiv,
        )
    )
    registry.register(
        ToolSpec(
            name="citation_graph",
            description="计算库内引用图谱与 PageRank，返回最核心的几篇论文及其影响力排名。",
            args_model=CitationGraphArgs,
            handler=_citation_graph,
        )
    )
    registry.register(
        ToolSpec(
            name="find_main_papers",
            description="按库内 PageRank 直接返回基石论文（被库内其他论文引用最多的核心工作）。",
            args_model=FindMainPapersArgs,
            handler=_find_main_papers,
        )
    )
    registry.register(
        ToolSpec(
            name="list_papers",
            description="列出库内全部论文及其 id、年份、页数与块数；用于先确认库内现有文献。",
            args_model=ListPapersArgs,
            handler=_list_papers,
        )
    )
    registry.register(
        ToolSpec(
            name="scan_pdf",
            description="诊断一篇 PDF 的文字层状况：哪些页是扫描件、本机能否 OCR。",
            args_model=ScanPdfArgs,
            handler=_scan_pdf,
        )
    )
    registry.register(
        ToolSpec(
            name="find_tables",
            description=(
                "读一篇论文里的表格（行×列还原过的单元格，不是被切块器拆散的行），"
                "可按关键词筛行并带上表头；首次解析较慢，扫描件没有文字层会如实说明。"
            ),
            args_model=FindTablesArgs,
            handler=_find_tables,
        )
    )
    return registry


TOOLS = build_tool_registry()


def tool_catalog_text(tools: ToolRegistry) -> str:
    """给模型看的工具目录。参数用 JSON schema 原样给出，避免自然语言转述失真。"""
    lines: list[str] = []
    for spec in tools.describe():
        schema = spec["parameters"]
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        params = []
        for name, info in props.items():
            params.append(
                f"{name}:{info.get('type', 'any')}"
                f"{'（必填）' if name in required else '（可选）'}"
            )
        lines.append(f"- {spec['name']}：{spec['description']}")
        lines.append(f"  参数：{', '.join(params) if params else '无'}")
    return "\n".join(lines)


def args_to_json(args: dict[str, Any]) -> str:
    """trace/提示词里展示参数用。排序键保证同样参数总是同一串文本。"""
    return json.dumps(args, ensure_ascii=False, sort_keys=True)
