"""FastAPI 服务：把解析 / 检索 / 溯源校验这条链路暴露成 HTTP。

设计上的三个取舍：

1. **问答分两种模式**。`mode=answer` 生成结论，`mode=retrieve` 只回召回证据。
   这不是为了方便调试，而是比赛演示的兜底：DeepSeek 断供时，
   界面仍然能展示"检索到了哪些块、在第几页"，而不是白屏报错。

   注意 `retrieve` 模式下**仍然会尝试**检索词扩展（中文问英文库，不扩展就是 0 召回），
   但扩展失败只退回原始问题，不会让请求失败 —— 这才是"断网还能演示"的真实含义。
   说成"完全不碰大模型"是不准确的，别那样写。
2. **被拦下的结论照样返回**（status=unsupported）。抗幻觉的证据不是"答案都对"，
   而是"拦住了哪几条、为什么拦"——把它藏起来反而浪费了这个模块的说服力。
3. **上传用内容哈希做 id**。同一份 PDF 传两次得到同一个 id，
   重复上传不会在库里堆两个副本（也是版本谱系的地基）。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.deps import (
    STORE_DIR,
    WEB_DIR,
    anchor_out,
    evidence_out,
    get_paper,
    order_versions,
    paper_out,
    registry,
)
from app.api.ratelimit import RateLimitMiddleware, limiter_from_env
from app.api.registry import Paper, PaperBusy, PaperNotFound
from app.config import load_dotenv
from app.api.schemas import (
    AnchorOut,
    AnswerOut,
    BatchItemOut,
    BilingualOut,
    BilingualParagraphOut,
    ChunkOut,
    ClaimOut,
    CompareOut,
    CopilotOutlineOut,
    CopilotOutlineRequest,
    CopilotPointOut,
    CopilotReferenceOut,
    CopilotSectionOut,
    DiffChangeOut,
    DiffSummaryItem,
    EvidenceOut,
    GraphEdgeOut,
    GraphNodeOut,
    GraphOut,
    PaperOut,
    QARequest,
    ReferenceOut,
    StatsOut,
)
from app.rag.answer import answer_question
from app.bibtex import to_bibtex
from app.rag.bilingual import translate_chunk
from app.rag.copilot import build_outline
from app.rag.query import expand_query, retrieve
from app.retrieval.bm25 import Hit
from app.versioning import Relation, classify, diff_versions, jaccard
from app.versioning_summary import DiffSummary, change_to_dict, summarize_diff

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 541 页教材也就 20MB 上下
ROOT = Path(__file__).resolve().parents[4]

load_dotenv(ROOT / ".env")

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """起服务时把 arXiv 追踪的定时器拉起来，退出时收掉。

    放在 lifespan 里而不是模块顶层：测试用 `TestClient(app)` 且不进上下文时
    根本不会触发它，单测也就不会因为有人建了一条订阅就去连 arXiv。
    `PAPER_TRACKING_SCHEDULER=0` 可以再关一道。
    """
    from app import tracking
    from app.api.routers import tracking as tracking_routes

    task = tracking.start_scheduler(tracking_routes.STORE)
    try:
        yield
    finally:
        if task is not None:
            tracking.stop_scheduler()


app = FastAPI(
    title="科研助手 Agent API",
    version="0.1.0",
    description="每句话都能点回原文精确位置的版本感知科研 Agent",
    lifespan=_lifespan,
)

# 前端开发时跑在 3000/5173，生产由本服务直接托管静态文件，同源无需 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 限流紧挨着 CORS 装上：排在业务中间件之前，被拦下的请求连解析都不做。
# 用纯 ASGI 中间件而不是 BaseHTTPMiddleware —— SSE 流式响应不该被它缓冲一层。
_limiters = limiter_from_env(os.environ)
if _limiters is not None:
    app.add_middleware(RateLimitMiddleware, limiter=_limiters)


# ------------------------------------------------------------------ 工具
#
# 这三个组装函数与 _get 已经搬进 deps.py（新增的路由模块也要用它们）。
# 这里保留旧名字，是为了让本文件几十处调用点不用一起改 —— 改名是无谓的 churn。


_paper_out = paper_out
_anchor_out = anchor_out
_evidence_out = evidence_out
_get = get_paper


# ------------------------------------------------------------------ 路由


def _rss_mb() -> float:
    """本进程常驻内存（MB）。

    读 `/proc/self/status` 而不是引 psutil：容器里就是 Linux，一个文件读完，
    不必为一个数字给镜像加依赖。非 Linux 平台返回 0 —— 这个数字是给人看的，
    拿不到时宁可空着，也不要编一个。
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return 0.0


@app.get("/api/health")
def health() -> dict:
    papers = registry.all()
    return {
        "ok": True,
        "papers": len(papers),
        # 规模与内存：长文档用例（541 页）要看的"工程鲁棒性"，给的是可核对的数字
        "pages": sum(p.pages for p in papers),
        "chunks": sum(len(p.chunks) for p in papers),
        "memory_mb": _rss_mb(),
        "llm_configured": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }


@app.get("/api/papers", response_model=list[PaperOut])
def list_papers() -> list[PaperOut]:
    return [_paper_out(p) for p in registry.all()]


@app.get("/api/graph", response_model=GraphOut)
def citation_graph(max_external: int = 24) -> GraphOut:
    """引用图谱：谁引用了谁 + 库内 PageRank 影响力排名。

    节点、连线、PageRank、坐标全在后端算好 —— 前端只管画。
    放前端算的话，同一份数据换个浏览器打开布局就变，演示时最毁信任。
    """
    g = registry.graph(max_external=max(0, min(max_external, 60)))
    return GraphOut(
        nodes=[
            GraphNodeOut(
                id=n.id,
                label=n.label,
                kind=n.kind,
                year=n.year,
                paper_ids=list(n.paper_ids),
                versions=n.versions,
                cites=n.cites,
                cited_by=n.cited_by,
                rank=n.rank,
                x=n.x,
                y=n.y,
            )
            for n in g.nodes
        ],
        edges=[GraphEdgeOut(source=e.source, target=e.target, via=e.via, score=e.score) for e in g.edges],
        stats=g.stats,
        notes=g.notes,
    )


# on_duplicate 的三种策略。默认 warn 是**不破坏性**的那个：
# 两个版本都留在库里，让用户自己决定 —— 自动覆盖一旦判错，删掉的是对方的原始文件。
DUPLICATE_POLICIES = {"warn", "keep", "replace"}
MAX_BATCH_FILES = 20


def _check_policy(on_duplicate: str) -> str:
    policy = (on_duplicate or "warn").strip().lower()
    if policy not in DUPLICATE_POLICIES:
        raise HTTPException(
            400, f"on_duplicate 只能是 {sorted(DUPLICATE_POLICIES)} 之一，收到 {on_duplicate!r}"
        )
    return policy


def _ingest(data: bytes, filename: str) -> tuple[Paper, bool]:
    """校验 + 落盘 + 切块。HTTP 状态码在这里定，批量上传复用同一套判定。

    第二个返回值是"这次上传的字节库里本来就有"。这个区分很关键：
    重传同一个文件时 `registry.add` 返回的是**库里原来那个对象**，
    后续若照常执行 on_duplicate=keep 的 `remove(paper.id)`，
    删掉的就是用户早就传好的那篇。
    """
    if not data:
        raise HTTPException(400, "文件内容为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限")
    # 只认 PDF 魔数：扩展名可以随便改，文件头不能
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise HTTPException(400, "该文件不是 PDF")

    identical = registry.lookup(data) is not None
    try:
        paper = registry.add(data, filename)
    except Exception as exc:  # noqa: BLE001 —— 解析器对坏 PDF 的异常类型不可枚举
        raise HTTPException(422, f"PDF 解析失败：{exc}") from exc

    if not paper.chunks:
        # 走到这里说明解析**已经跑完了**，扫描页该 OCR 的也认过了。
        # 所以不能照抄 `analyze()` 的 note —— 那句写的是"解析时会自动识别"，
        # 而解析刚刚结束、仍然什么都没抽到，用户会被这句假承诺带着反复重传。
        from app.parsing.scan import analyze

        report = analyze(str(paper.path))
        registry.remove(paper.id)  # 空文档没必要留在库里（三处一起删）
        raise HTTPException(422, _empty_body_reason(report))
    return paper, identical


def _empty_body_reason(report) -> str:
    """「什么都没抽到」要说清是哪一种空，否则用户不知道该不该重试。"""
    scanned = len(report.scanned_pages)
    if not scanned:
        # 有文字层却切不出块：这不是扫描件问题，别让人误以为是 OCR 没开
        return f"该 PDF 含文字层，但未能切分出正文段落。{report.note}"
    if report.ocr_available:
        return (
            f"共 {report.pages} 页，其中 {scanned} 页缺少文字层，已用 OCR 逐页识别过，"
            "但仍没有可用的文字（常见于空白页、整页图片或纯图表页）。"
        )
    return (
        f"共 {report.pages} 页，其中 {scanned} 页缺少文字层，本机没有可用的 OCR 引擎，"
        "这些页面无法识别。执行一次 `uv sync` 可连同 RapidOCR 一并安装，安装完成后重新上传即可。"
    )


@app.post("/api/papers", response_model=PaperOut, status_code=201)
async def upload_paper(
    file: UploadFile = File(...), on_duplicate: str = "warn"
) -> PaperOut:
    """上传单篇。`on_duplicate` 决定撞上旧版本时怎么办。

    - warn（默认）：两版都入库，响应里带上 duplicate_of / relation / recommendation；
    - keep：不入库，409 把库里的那篇原样返回，让前端问用户"要不要覆盖"；
    - replace：删掉库里那一版再入库 —— 这是**破坏性**的，所以必须前端先确认。

    解析放到线程池：541 页的教材要几十秒，同步跑会把这个单进程服务整个卡死，
    期间连 /api/health 都返回不了（演示现场最容易翻车的地方）。
    """
    policy = _check_policy(on_duplicate)
    data = await file.read()
    paper, identical = await run_in_threadpool(
        _ingest, data, file.filename or "untitled.pdf"
    )
    if identical:
        # 同一个文件又传了一次：什么都没变，就别拿"发现相似论文"去吓用户
        return _paper_out(paper)

    dup = registry.find_similar(paper)
    if dup is not None and policy == "keep":
        registry.remove(paper.id)
        raise HTTPException(
            409,
            {
                "message": "库里已有内容高度重合的版本，按 on_duplicate=keep 未入库",
                "existing": _paper_out(dup[0]).model_dump(),
                "uploaded": _paper_out(paper, dup).model_dump(),
            },
        )
    if dup is not None and policy == "replace":
        registry.remove(dup[0].id)
        return _paper_out(paper, None)
    # 判定依据是正文语义，不是文件名 —— 官方用例 3.1 明确要求这一点
    return _paper_out(paper, dup)


@app.post("/api/papers/batch", response_model=list[BatchItemOut], status_code=201)
async def upload_batch(
    files: list[UploadFile] = File(...), on_duplicate: str = "warn"
) -> list[BatchItemOut]:
    """批量上传：逐个处理，**一个失败只影响它自己那一行**。

    为什么不做"全有或全无"：用户一次拖进来 10 个 PDF，其中一个坏了，
    把另外 9 个已经解析好的结果回滚掉是更糟的体验（重传又要等一遍）。
    部分成功 + 逐条原因，才是用户真正要的语义。
    """
    policy = _check_policy(on_duplicate)
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(413, f"一次最多 {MAX_BATCH_FILES} 个文件")

    items: list[BatchItemOut] = []
    for upload in files:
        name = upload.filename or "untitled.pdf"
        try:
            data = await upload.read()
            paper, identical = await run_in_threadpool(_ingest, data, name)
        except HTTPException as exc:
            items.append(BatchItemOut(filename=name, ok=False, error=str(exc.detail)))
            continue
        except Exception as exc:  # noqa: BLE001 —— 单条失败绝不能拖垮整批
            items.append(BatchItemOut(filename=name, ok=False, error=f"上传失败：{exc}"))
            continue

        if identical:
            items.append(
                BatchItemOut(filename=name, ok=True, skipped=True, paper=_paper_out(paper))
            )
            continue

        dup = registry.find_similar(paper)
        if dup is not None and policy == "keep":
            registry.remove(paper.id)
            items.append(
                BatchItemOut(
                    filename=name,
                    ok=False,
                    skipped=True,
                    error=f"库里已有高度重合的版本：{dup[0].title}",
                )
            )
            continue
        if dup is not None and policy == "replace":
            registry.remove(dup[0].id)
            dup = None
        items.append(BatchItemOut(filename=name, ok=True, paper=_paper_out(paper, dup)))
    return items


@app.get("/api/papers/{paper_id}", response_model=PaperOut)
def get_paper(paper_id: str) -> PaperOut:
    return _paper_out(_get(paper_id))


@app.delete("/api/papers/{paper_id}", status_code=204)
def delete_paper(paper_id: str) -> Response:
    """删掉一篇文献：内存、sidecar、落盘副本一起删。

    204 而不是回一份 JSON：前端删完只关心"没了"，多余的回包只会让人以为
    "接口还有别的副作用"。真正不可逆的是磁盘上的那份副本，所以前端必须
    先让用户确认（界面上是两步：点删除 → 弹确认）。
    """
    try:
        registry.remove(paper_id)
    except PaperNotFound as exc:
        raise HTTPException(404, f"库内不存在该文献：{paper_id}") from exc
    except PaperBusy as exc:
        # 409 而不是 500：这是"稍后再试"（关掉正在看这份 PDF 的阅读器/预览器），
        # 不是服务器坏了。库里的状态没有被改动，用户可以直接重试。
        raise HTTPException(
            409, "该 PDF 正被其他程序占用（预览器、杀毒软件扫描均可能导致），请稍后重试删除"
        ) from exc
    return Response(status_code=204)


@app.get("/api/papers/{paper_id}/chunks", response_model=list[ChunkOut])
def list_chunks(paper_id: str) -> list[ChunkOut]:
    """前端目录树与"整段高亮"都用这个。"""
    paper = _get(paper_id)
    return [
        ChunkOut(
            index=c.index,
            section=c.section,
            page_start=c.page_start,
            page_end=c.page_end,
            n_tokens=c.n_tokens,
            text=c.text,
            preview=c.preview(160),
            rects=[[float(page), *[round(v, 1) for v in rect]] for page, rect in c.rects],
        )
        for c in paper.chunks
    ]


@app.get("/api/papers/{paper_id}/scan")
def scan_report(paper_id: str) -> dict:
    """这份 PDF 的文字层状况：几页是扫描页、本机能不能 OCR。

    单独一个端点而不是塞进 PaperOut：它要逐页 get_text，541 页的书要一两秒，
    不该让每次列表请求都付这个钱。前端打开文献时按需拉一次。
    """
    from dataclasses import asdict

    from app.parsing.scan import analyze

    paper = _get(paper_id)
    return asdict(analyze(str(paper.path)))


@app.get("/api/papers/{paper_id}/references", response_model=list[ReferenceOut])
def list_references(paper_id: str) -> list[ReferenceOut]:
    """参考文献条目，供正文里的 [11] 标记做双向跳转。

    解析不出来的条目**照样返回**（`parsed=false`）：藏起来会让人以为这篇没参考文献，
    而真相是"我们只认出了标记、没解出标题" —— 前端把 raw 原样显示即可。
    """
    paper = _get(paper_id)
    return [ReferenceOut.of(r) for r in paper.references]


@app.get("/api/papers/{paper_id}/bibtex")
def paper_bibtex(paper_id: str) -> Response:
    """把这篇的参考文献导出成 .bib —— 拿到就能丢进 Zotero / Overleaf。

    解析不出标题的条目照样导出（`note` 里写明字段不全）：丢条目比给残缺条目更难发现。
    """
    paper = _get(paper_id)
    body = to_bibtex(paper.references, source=paper.title)
    return Response(
        content=body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{paper_id}.bib"'},
    )


@app.get("/api/bibtex")
def library_bibtex() -> Response:
    """整个文献库的 .bib：写综述时一次性把这十几篇的引用都拿到。"""
    papers = registry.all()
    refs = [ref for paper in papers for ref in paper.references]
    body = to_bibtex(refs, source=f"文献库 {len(papers)} 篇")
    return Response(
        content=body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="library.bib"'},
    )


@app.get("/api/papers/{paper_id}/bilingual", response_model=BilingualOut)
def bilingual(paper_id: str, chunk: int) -> BilingualOut:
    """中英对照。

    模型不可用时**不 500**：原文照常返回、译文留空、degraded=true ——
    断网演示时这一块仍然能打开。404 只留给"这篇/这一块不存在"。
    """
    paper = _get(paper_id)
    if chunk < 0 or chunk >= len(paper.chunks):
        raise HTTPException(404, f"这篇文献没有第 {chunk} 块")

    c = paper.chunks[chunk]
    result = translate_chunk(c, paper_id=paper.id)
    return BilingualOut(
        chunk=c.index,
        section=c.section,
        page_start=c.page_start,
        page_end=c.page_end,
        paragraphs=[BilingualParagraphOut(en=p.en, zh=p.zh) for p in result.paragraphs],
        degraded=result.degraded,
        model=result.model,
    )


@app.get("/api/papers/{paper_id}/file")
def get_file(paper_id: str):
    from fastapi.responses import FileResponse

    paper = _get(paper_id)
    return FileResponse(paper.path, media_type="application/pdf", filename=paper.filename)


@app.get("/api/papers/{paper_id}/compare/{other_id}", response_model=CompareOut)
def compare(paper_id: str, other_id: str, summarize: bool = True) -> CompareOut:
    """两版差异。谁新谁旧按 arXiv 版本号 → PDF 生成日期 → 上传时间依次判。

    早先只看 `created_at`：扫描件没有元数据日期，`created_at` 是空串，
    于是退回上传时间 —— 后传上去的 v1 扫描件被判成"更新的一版"，
    方向整个反了。同一篇有 vN 的时候，vN 才是能站的依据。
    """
    a, b = _get(paper_id), _get(other_id)
    if a.id == b.id:
        raise HTTPException(400, "两份为同一文件，无需对比")

    old, new, order_by = order_versions(a, b)
    relation = classify(jaccard(old.fingerprint, new.fingerprint))
    diff = diff_versions(old.path, new.path)
    ocr_side = (
        "both" if old.scanned and new.scanned
        else "old" if old.scanned
        else "new" if new.scanned
        else ""
    )

    summary = DiffSummary(items=[], ok=True)
    if summarize:
        summary = summarize_diff(diff, ocr_side=ocr_side)

    return CompareOut(
        old=_paper_out(old),
        new=_paper_out(new),
        relation=relation.kind,
        similarity=relation.similarity,
        stats=diff.stats,
        changes=[DiffChangeOut(**change_to_dict(c)) for c in diff.changes],
        summary=[DiffSummaryItem(**it) for it in summary.items],
        summary_ok=summary.ok,
        summary_note=summary.reason,
        order_by=order_by,
        ocr_side=ocr_side,
    )


@app.post("/api/papers/{paper_id}/qa", response_model=AnswerOut)
def ask(paper_id: str, req: QARequest) -> AnswerOut:
    paper = _get(paper_id)
    t0 = time.perf_counter()

    queries = expand_query(req.question) if req.expand else [req.question]
    trace: dict = {}
    hits = retrieve(
        req.question, retriever=paper.bm25, k=req.k, expand=req.expand,
        queries=queries, trace=trace,
    )
    candidate_chunks = trace.get("candidates", 0) if trace.get("narrowed") else 0

    if req.mode == "retrieve" or not hits:
        return AnswerOut(
            question=req.question,
            claims=[],
            evidence=_evidence_out(hits),
            stats=StatsOut(),
            took_ms=int((time.perf_counter() - t0) * 1000),
            corpus_chunks=trace.get("corpus", 0),
            candidate_chunks=candidate_chunks,
            mode="retrieve",
            scanned=paper.scanned,
        )

    from app.llm.client import LLMError

    try:
        answer = answer_question(
            req.question,
            hits=hits,
            index=paper.text_index,
            doc_id=paper.id,
        )
    except LLMError as exc:
        # 大模型不可用不该让整个问答 500：降级成"只给证据"，前端照常能演示
        raise HTTPException(503, f"大模型暂不可用：{exc}") from exc

    stats = answer.stats
    return AnswerOut(
        question=req.question,
        claims=[
            ClaimOut(
                text=c.claim.text,
                status=c.status,
                reason=c.reason,
                anchors=[_anchor_out(a) for a in c.anchors],
            )
            for c in answer.claims
        ],
        evidence=_evidence_out(answer.evidence),
        stats=StatsOut(**stats),
        took_ms=int((time.perf_counter() - t0) * 1000),
        corpus_chunks=trace.get("corpus", 0),
        candidate_chunks=candidate_chunks,
        mode="answer",
        scanned=paper.scanned,
    )


@app.post("/api/copilot/outline", response_model=CopilotOutlineOut)
def copilot_outline(req: CopilotOutlineRequest) -> CopilotOutlineOut:
    """论文框架生成。

    检索与生成都放在 rag/copilot.py，这里只划边界：库空是请求没法完成（400），
    模型不可用则降级为"只有可引用原文"（200 + degraded）。
    """
    papers = registry.all()
    if not papers:
        raise HTTPException(400, "文献库是空的，请先上传论文再生成框架")

    result = build_outline(req.idea, papers, max_points=req.max_points)
    return CopilotOutlineOut(
        idea=result.idea,
        outline=[
            CopilotSectionOut(
                title=s.title,
                points=[CopilotPointOut(text=p.text, evidence=p.evidence) for p in s.points],
            )
            for s in result.outline
        ],
        references=[
            CopilotReferenceOut(
                n=e.n,
                paper_id=e.paper_id,
                title=e.title,
                page=e.page,
                quote=e.quote,
            )
            for e in result.references
        ],
        degraded=result.degraded,
    )


# 按功能拆出去的路由：各自带 APIRouter，这里统一挂载。
# 放在静态挂载之前，否则 "/" 会先被 StaticFiles 吃掉。
from app.api.routers import agent as agent_routes  # noqa: E402
from app.api.routers import cross as cross_routes  # noqa: E402
from app.api.routers import dataviz as dataviz_routes  # noqa: E402
from app.api.routers import recommend as recommend_routes  # noqa: E402
from app.api.routers import review as review_routes  # noqa: E402
from app.api.routers import stream as stream_routes  # noqa: E402
from app.api.routers import tools as tool_routes  # noqa: E402
from app.api.routers import translate as translate_routes  # noqa: E402
from app.api.routers import visuals as visual_routes  # noqa: E402
from app.api.routers import formulas as formula_routes  # noqa: E402
from app.api.routers import tracking as tracking_routes  # noqa: E402

app.include_router(tool_routes.router)
app.include_router(recommend_routes.router)
app.include_router(review_routes.router)
app.include_router(agent_routes.router)
app.include_router(dataviz_routes.router)
app.include_router(stream_routes.router)
app.include_router(translate_routes.router)
app.include_router(visual_routes.router)
app.include_router(formula_routes.router)
app.include_router(tracking_routes.router)
# 跨文献对比：同一个问题分别去两到三篇里取证据，逐条锚回各自的原文
app.include_router(cross_routes.router)


# 静态前端放在最后挂载：/api 与 /docs 先匹配，剩下的才交给静态文件
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
