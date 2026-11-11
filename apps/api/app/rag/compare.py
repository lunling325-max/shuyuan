"""跨文献对比：同一个问题分别去两（三）篇里找证据，每条结论锚回各自的原文。

和单篇问答只有三处不同，但每一处都关系到"能不能核对"：

1. 证据块按论文分组编号（`A#1`、`B#2`），模型必须说明每条引用来自哪一篇；
2. 校验时把 A/B/C 映射回各自 `paper_id` 的 TextIndex —— 一条引用的出处是**论文**，
   不是段落；映射错了就等于把 B 的话算到 A 头上，比没有出处更糟；
3. 允许"两篇都有"的结论，所以每条引用的标签要逐条给出，不能一条结论只贴一个。

**不把两篇拼成一个超长上下文直接问**，正是为了防止这件事：模型分不清哪句话
出自哪篇，写出来的对比看着漂亮却没法回原文核对。分组编号 + 逐条标注，
是把"可核对"留在协议里，而不是留在模型的心情里。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.guardrails.citation import Claim, Quote, VerifiedClaim, summarize, verify_claims
from app.parsing.text_index import TextIndex
from app.retrieval.bm25 import Hit

# 三篇足够覆盖赛题里"GPT-1 到 GPT-3 的演进"这类问题，再多就不是"对比"而是综述了。
LABELS = ("A", "B", "C")

SYSTEM_PROMPT = """你是严谨的论文对比助手。用户会给你同一个问题、两到三篇论文的证据块，
你要给出**可核对**的对比结论。

铁律：
1. 只能使用证据块中的内容，禁止引入证据之外的知识，即使你确定它是对的。
2. 每条结论必须附 quotes，每个 quote 是一个对象：
   {"paper": "A", "quote": "从 A 的证据块里逐字复制的英文片段（20–120 词）"}
   - paper 只能是 A、B、C 之一，必须与证据块编号的前缀一致；
   - quote 必须逐字复制，不改写、不翻译、不拼接相距很远的句子、不改标点。
3. 只在某一篇里出现的结论，就只引用那一篇 —— 不要为了"看起来对称"硬凑另一篇的引用。
4. 证据不足以回答时输出 {"claims": []}，不要猜，也不要拿常识补。
5. 结论用中文写；专有名词照抄原文（Transformer、BLEU），通用词写中文。

只输出 JSON，不要任何解释文字：
{"claims": [{"text": "中文结论", "quotes": [{"paper": "A", "quote": "原文片段"}]}]}"""


@dataclass(slots=True)
class PaperRef:
    label: str
    id: str
    title: str


@dataclass(slots=True)
class Comparison:
    question: str
    papers: list[PaperRef]
    claims: list[VerifiedClaim] = field(default_factory=list)
    evidence: dict[str, list[Hit]] = field(default_factory=dict)
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, int]:
        return summarize(self.claims)


def build_evidence(evidence: dict[str, list[Hit]], titles: dict[str, str]) -> str:
    """按论文分组的证据块。前缀 `A#1` 就是模型引用时要写的 paper 标签。"""
    blocks: list[str] = []
    for label in LABELS:
        hits = evidence.get(label) or []
        if not hits:
            continue
        blocks.append(f"=== 论文 {label}：{titles.get(label, '')} ===")
        for i, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            span = (
                f"p.{chunk.page_start}"
                if chunk.page_start == chunk.page_end
                else f"p.{chunk.page_start}-{chunk.page_end}"
            )
            blocks.append(f"[{label}#{i}] {span} · {chunk.section}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def parse_claims(raw: dict, ids: dict[str, str]) -> list[Claim]:
    """把模型输出解析成待校验的 Claim；quote 必须自带 paper 标签。

    没标签的引用**直接丢掉**，不猜：猜错等于把一篇的话算到另一篇头上，
    而这张表存在的意义就是"哪句话出自哪篇"。丢掉之后这条结论若一条引用不剩，
    校验器会按既有规则判 unsupported 并写明原因。
    """
    claims: list[Claim] = []
    for item in raw.get("claims", []) or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        quotes: list[Quote] = []
        for q in item.get("quotes", []) or []:
            if not isinstance(q, dict):
                continue
            label = str(q.get("paper", "")).strip().upper()
            body = q.get("quote") or q.get("text") or ""
            if label in ids and isinstance(body, str) and body.strip():
                quotes.append(Quote(doc=ids[label], text=body.strip()))
        claims.append(Claim(text=text, quotes=quotes))
    return claims


def verify_comparison(
    claims: list[Claim],
    indexes: dict[str, TextIndex],
) -> list[VerifiedClaim]:
    """按论文分别校验。`indexes` 的键是 paper_id，与 Quote.doc 对齐。"""
    return verify_claims(claims, indexes)


def build_user(question: str, evidence: dict[str, list[Hit]], titles: dict[str, str]) -> str:
    return f"问题：{question}\n\n证据块：\n\n{build_evidence(evidence, titles)}"
