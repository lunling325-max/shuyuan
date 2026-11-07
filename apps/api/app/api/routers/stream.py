"""Token 级流式问答：`POST /api/papers/{paper_id}/qa/stream`。

## 事件协议（给前端的契约）

响应是 `text/event-stream`。每帧固定两行 `event:` + `data:` 再加一个空行，
`data` 一律是 JSON（`ensure_ascii=False`，中文不会变成 `\\uXXXX`）。
事件按下面的顺序发送：

| # | event | data | 说明 |
|---|-------|------|------|
| 1 | `meta` | `{paper_id, title, question}` | 打开正文的第一帧 |
| 2 | `queries` | `{queries: [str]}` | 实际用于检索的词（retrieve 模式就是原问题） |
| 3 | `evidence` | `{hits: [{chunk_index, page, section, score, preview}]}` | 召回证据 |
| 4 | `token` | `{text: "<delta>"}` | 逐 token，可能 0..N 个 |
| 5 | `claims` | `{claims: [{text, status, reason, anchors: [...]}], stats: {supported, weak, unsupported}}` | 校验后的最终结论 |
| 6 | `done` | `{took_ms, scanned}` | 正常结束；scanned=这份文献是扫描件 |
| 7 | `error` | `{stage, message}` | 任一阶段失败；**不发 done** |

前端据此区分两种结束方式：收到 `done` 是正常完成；只收到 `error` 是半路挂了，
此时前面已经流出的 token 不能当作最终答案。

`mode=retrieve` 时只发 `meta` / `queries` / `evidence` / `done`，完全不调用模型。

## 必须诚实面对的取舍：token 是**未校验**的模型原文

`token` 事件发送的是模型此刻吐出的原始文本（本链路的答案格式是 JSON，
所以屏幕上先滚出来的也是 JSON）。**它没有经过抗幻觉校验，不能被当成结论。**
token 一旦发出就收不回来，因此服务端的策略是：

1. 先让用户看到输出在动，而不是盯着转圈；
2. 流结束后用**同一套**证据块、system prompt、claims 解析和
   `guardrails.citation.verify_claims` 做完整校验；
3. 用 `claims` 事件把最终版本（含被拦下的条目、status、reason、锚点）推给前端，
   前端应以它覆盖/替换 token 阶段的临时展示。

换句话说：**"流式渲染"不等于"流式也经过校验"**。校验发生在流结束之后，
在此之前屏幕上的一切都只是模型草稿。把这个差别写进协议，比含糊地说
"流式答案也安全"要诚实得多。

## 模型不可用不能 500

SSE 一旦开始，HTTP 状态码就改不了了。所以缺 key 这类**可预判**的失败
在开始流之前就被识别，折叠成 `meta → queries → evidence → error{stage:"llm"}`：
用户仍然能看到检索到了哪些原文块，断网演示不至于白屏。
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.api import deps
from app.api.schemas import QARequest
from app.guardrails.citation import verify_claims
from app.rag.answer import SYSTEM_PROMPT
from app.rag.query import expand_query, retrieve
from app.rag.streaming import (
    build_answer_user,
    claims_payload,
    evidence_payload,
    parse_model_claims,
    sse,
)

router = APIRouter()


def _prepare(paper, req: QARequest):
    """检索与检索词扩展（同步、可能走网络），由路由丢进线程池执行。

    返回 `(queries, hits, failure)`。`failure` 非空时仍然会先发 meta/queries/evidence，
    再发 error —— 检索失败也不该表现成"连接被掐断"。
    """
    queries: list[str] = []
    try:
        # retrieve 模式明确不调模型：不扩展，直接用原问题查库。
        if req.mode == "retrieve" or not req.expand:
            queries = [req.question]
        else:
            queries = expand_query(req.question)
        hits = retrieve(
            req.question,
            retriever=paper.bm25,
            k=req.k,
            expand=req.expand,
            queries=queries,
        )
        return queries, hits, None
    except Exception:  # noqa: BLE001 —— 检索/词法索引的异常类型不可枚举
        # 不把异常细节发给前端：可能含内部路径或索引实现细节。
        return queries or [req.question], [], ("retrieve", "检索阶段失败，请稍后重试")


def _verify(paper, claims):
    """跑完整引用校验。同步函数，供线程池调用（TextIndex 首次构建可能较慢）。"""
    return verify_claims(claims, {paper.id: paper.text_index})


def _llm_message(exc: BaseException) -> str:
    """把客户端异常翻译成前端能读的一句话，且不泄露密钥值。"""
    text = str(exc).strip()
    if "DEEPSEEK_API_KEY" in text:
        return "大模型未配置（缺少 DEEPSEEK_API_KEY），以上为仅检索结果"
    # HTTP 错误体已经由客户端截到 400 字，这里再兜一层长度。
    return text[:300] or "大模型暂不可用"


def _took_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


async def _event_stream(
    paper,
    req: QARequest,
    request: Request,
    *,
    queries: list[str],
    hits: list,
    failure: tuple[str, str] | None,
    t0: float,
    configured: bool,
) -> AsyncIterator[str]:
    """按协议逐帧产出事件。真正的生成只在这里发生。"""
    yield sse(
        "meta",
        {"paper_id": paper.id, "title": paper.title, "question": req.question},
    )
    yield sse("queries", {"queries": queries})
    yield sse("evidence", evidence_payload(hits))

    if failure is not None:
        yield sse("error", {"stage": failure[0], "message": failure[1]})
        return

    # 没有召回等价于只读检索：模型拿不到证据，硬答只会是幻觉。
    if req.mode == "retrieve" or not hits:
        yield sse("done", {"took_ms": _took_ms(t0), "scanned": paper.scanned})
        return

    # 可预判的失败在进入 token 流之前折叠成 error，HTTP 仍然是 200。
    if not configured:
        yield sse(
            "error",
            {
                "stage": "llm",
                "message": "大模型未配置（缺少 DEEPSEEK_API_KEY），以上为仅检索结果",
            },
        )
        return

    if await request.is_disconnected():
        return

    # 延迟导入：测试可以 monkeypatch `app.llm.client.stream_chat`。
    from app.llm.client import LLMError, stream_chat

    chunks: list[str] = []
    gen = stream_chat(
        build_answer_user(req.question, hits),
        system=SYSTEM_PROMPT,
        temperature=0.1,
    )
    try:
        for token in gen:
            # 客户端走了就别再读了：每多读一个 token 都是白花的钱。
            if await request.is_disconnected():
                return
            if not token:
                continue
            chunks.append(token)
            yield sse("token", {"text": token})
    except LLMError as exc:
        yield sse("error", {"stage": "llm", "message": _llm_message(exc)})
        return
    except Exception:  # noqa: BLE001 —— 生成器内部异常不该把连接变成 500
        yield sse("error", {"stage": "llm", "message": "模型流式输出中断"})
        return
    finally:
        # 无论正常结束、异常还是客户端断开，都要关掉底层 HTTP 响应，
        # 让 urllib 停止读流。close 本身失败不能盖住真正的退出原因。
        close = getattr(gen, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001
                pass

    try:
        claims = parse_model_claims("".join(chunks), doc_id=paper.id)
    except Exception:  # noqa: BLE001
        # 模型吐了一段不是合法 JSON 的文本：校验无法进行，属于真失败。
        yield sse(
            "error",
            {
                "stage": "parse",
                "message": "模型未返回可校验的 JSON，无法生成溯源结论",
            },
        )
        return

    try:
        verified = await run_in_threadpool(_verify, paper, claims)
    except Exception:  # noqa: BLE001
        yield sse("error", {"stage": "verify", "message": "引用校验失败，请稍后重试"})
        return

    yield sse("claims", claims_payload(verified))
    # scanned 挂在 done 上而不是 claims 上：它是"这份文献本身"的属性，
    # 不是校验结果的一部分，前端拿到它才能提示"引文来自 OCR"
    yield sse("done", {"took_ms": _took_ms(t0), "scanned": paper.scanned})


@router.post(
    "/api/papers/{paper_id}/qa/stream",
    response_class=StreamingResponse,
)
async def qa_stream(
    paper_id: str,
    req: QARequest,
    request: Request,
) -> StreamingResponse:
    # 取不到文献仍然是 404：这时还没开始流，状态码还能如实表达。
    paper = deps.get_paper(paper_id)
    t0 = time.perf_counter()

    # 检索是同步的（可能还含一次模型调用），放线程池里跑，别堵事件循环。
    queries, hits, failure = await run_in_threadpool(_prepare, paper, req)

    # 在开始流之前就把"模型没配"识别掉：后面才能用事件而不是状态码表达。
    configured = bool(os.environ.get("DEEPSEEK_API_KEY"))

    return StreamingResponse(
        _event_stream(
            paper,
            req,
            request,
            queries=queries,
            hits=hits,
            failure=failure,
            t0=t0,
            configured=configured,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # 让 nginx 之类的反代不要攒够一整块再转发，否则"流式"名存实亡
            "X-Accel-Buffering": "no",
        },
    )
