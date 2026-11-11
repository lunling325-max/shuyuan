"""流式问答的纯逻辑：SSE 编码、事件负载组装、模型原文解析。

这里**不生成答案**。生成用的 system prompt、证据块拼接、claims 解析与
引用校验全部复用 `app.rag.answer` / `app.guardrails.citation`，
本模块只负责把同一条链路的结果编码成可以逐帧发送的 SSE。

为什么单独一层：路由里一旦混进字符串拼接和 JSON 解析，测试就只能靠
"发起一次假请求"来覆盖，边界条件很难单独测。把这些纯函数拿出来后，
SSE 帧格式、证据字段、claims 统计都可以直接断言。
"""

from __future__ import annotations

import json
from typing import Any

from app.guardrails.citation import VerifiedClaim, summarize
from app.rag.answer import _parse_claims, build_evidence
from app.retrieval.bm25 import Hit


def sse(event: str, data: dict[str, Any]) -> str:
    """编码成一个 SSE 帧：`event:` + `data:` + 空行。

    `ensure_ascii=False` 是硬要求：中文若被转义成 `\\uXXXX`，
    前端拿到的 token 会和模型原文对不上，屏幕上还会出现一串反斜杠。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def evidence_payload(hits: list[Hit]) -> dict[str, Any]:
    """召回证据。只给预览，不把整块原文塞进流里。"""
    return {
        "hits": [
            {
                "chunk_index": h.chunk.index,
                "page": h.chunk.page_start,
                "section": h.chunk.section,
                "score": h.score,
                "preview": h.chunk.preview(160),
            }
            for h in hits
        ]
    }


def _anchor_payload(anchor) -> dict[str, Any]:
    """与 `schemas.AnchorOut` 保持同一组字段，前端不需要两套解析。"""
    return {
        "page": anchor.page,
        "bbox": [round(v, 1) for v in anchor.rect],
        "source_text": anchor.source_text,
        "exact": anchor.exact,
        "similarity": anchor.similarity,
        "truncated": getattr(anchor, "truncated", False),
        # 单篇问答里恒等于本篇 id；跨文献对比里前端靠它把锚点分到 A/B 两栏
        "doc": getattr(anchor, "doc", ""),
    }


def claims_payload(verified: list[VerifiedClaim]) -> dict[str, Any]:
    """校验后的结论。

    **unsupported 的结论也返回**，不带任何锚点。把它们藏起来会让人误以为
    模型没说过；展示成"这条被拦下了 + 原因"才是抗幻觉模块的价值。
    """
    return {
        "claims": [
            {
                "text": item.claim.text,
                "status": item.status,
                "reason": item.reason,
                "anchors": [_anchor_payload(a) for a in item.anchors],
            }
            for item in verified
        ],
        "stats": summarize(verified),
    }


def _strip_code_fence(text: str) -> str:
    """模型常把 JSON 包在 ```json ... ``` 里，解析前先剥掉。

    只处理"整段被围栏包住"这一种情况；夹在解释文字里的 JSON 不猜，
    解析失败就让上层发 error —— 猜错了比报错更危险（会把错位的内容当证据）。
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    newline = s.find("\n")
    if newline < 0:
        return s.strip("`").strip()
    s = s[newline + 1 :]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s.strip()


def parse_model_claims(raw: str, *, doc_id: str) -> list:
    """把流式累积的模型原文解析成待校验的 Claim 列表。

    复用 `answer._parse_claims`，保证流式与非流式对同一份模型输出
    的解析规则完全一致（比如空 text、非字符串 quotes 的处理）。
    """
    data = json.loads(_strip_code_fence(raw))
    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return _parse_claims(data, doc_id)


def build_answer_user(question: str, hits: list[Hit]) -> str:
    """构造生成用的 user prompt。

    模板必须与 `answer_question` 内的完全一致 —— 否则流式版只是"看起来像"
    复用了链路，模型看到的问题和证据其实换了形态。
    """
    return f"问题：{question}\n\n证据块：\n\n{build_evidence(hits)}"
