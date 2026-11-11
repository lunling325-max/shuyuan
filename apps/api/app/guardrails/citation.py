"""引用校验守卫（Citation Guardrail）。

原则：**LLM 输出的每一条结论，都必须附带能在原文里逐字找到的引用；否则不予采信。**

这是整个系统对抗幻觉的最后一道闸门，也是"溯源锚点"的产出处：
校验通过的同时就拿到了 (页码, bbox)，前端据此高亮原文。

三级判定：
- supported   : 至少一条引用逐字命中
- weak        : 只有模糊命中（≥阈值），原文可用但需标注不确定性
- unsupported : 一条都对不上 —— 判定为幻觉，整条丢弃
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from app.parsing.text_index import Anchor, TextIndex


@dataclass(slots=True)
class Quote:
    """LLM 声称的原文引用。doc 用于多文档场景（如 v1 / v7 对比）。"""

    doc: str
    text: str


@dataclass(slots=True)
class Claim:
    """LLM 输出的一条结论 + 它自带的证据。"""

    text: str
    quotes: list[Quote] = field(default_factory=list)


@dataclass(slots=True)
class VerifiedClaim:
    claim: Claim
    status: str  # supported | weak | unsupported
    anchors: list[Anchor] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "supported"


def verify_claims(
    claims: list[Claim],
    indexes: dict[str, TextIndex],
    *,
    min_similarity: float = 0.90,
) -> list[VerifiedClaim]:
    """逐条校验结论。不改写、不补全，只做判定。"""
    results: list[VerifiedClaim] = []

    for claim in claims:
        if not claim.quotes:
            results.append(
                VerifiedClaim(claim=claim, status="unsupported", reason="未提供任何引用")
            )
            continue

        anchors: list[Anchor] = []
        missing_docs: list[str] = []

        for quote in claim.quotes:
            index = indexes.get(quote.doc)
            if index is None:
                missing_docs.append(quote.doc)
                continue
            anchor = index.find(quote.text, min_similarity=min_similarity)
            if anchor is not None:
                # 记下这句话出自哪个文档。单篇问答里它恒等于本篇 id，
                # 跨文献对比里才是关键信息：同一条结论的两条引用可能来自不同论文，
                # 界面上必须能分开显示（E-038）。
                # Anchor 是 frozen 的，只能换一个新对象 —— 直接赋值会抛
                # FrozenInstanceError，而调用方通常把异常收进"降级"分支，
                # 结果是所有引用都被判成校验失败（实测 E-038 红过一轮）。
                anchors.append(replace(anchor, doc=quote.doc))

        if not anchors:
            reason = "所有引用均未在原文中命中"
            if missing_docs:
                reason += f"（未知文档: {', '.join(missing_docs)}）"
            results.append(VerifiedClaim(claim=claim, status="unsupported", reason=reason))
        elif all(a.exact for a in anchors):
            reason = "逐字命中"
            if any(a.truncated for a in anchors):
                reason += "（引用尾部被截断，已按最长可验证前缀对齐）"
            results.append(
                VerifiedClaim(claim=claim, status="supported", anchors=anchors, reason=reason)
            )
        else:
            worst = _format_similarity(min(a.similarity for a in anchors))
            results.append(
                VerifiedClaim(
                    claim=claim,
                    status="weak",
                    anchors=anchors,
                    reason=f"仅模糊命中（最低相似度 {worst}），原文需人工确认",
                )
            )

    return results


def _format_similarity(value: float) -> str:
    """相似度印到四位、去掉尾零。

    两位小数会把 0.9996 印成「1.00」，界面上就出现「仅模糊命中，相似度 1.00」
    这种自相矛盾的话 —— 用户看到 1.00 会以为这条通过了。
    """
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def summarize(results: list[VerifiedClaim]) -> dict[str, int]:
    """给日志/前端用的统计。"""
    counts = {"supported": 0, "weak": 0, "unsupported": 0}
    for r in results:
        counts[r.status] += 1
    return counts
