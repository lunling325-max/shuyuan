"""带溯源的问答：检索 → 生成 → 校验 → 回填页码坐标。

这里的核心不是"让模型答得好"，而是**让模型没法乱答**：

1. 证据块**不做截断**直接喂给模型 —— E-002 的教训，截断会逼模型去猜。
2. 模型必须为每条结论附上**原文逐字片段**，不是"第几条证据"这种弱引用。
   只要它想引用某句话，那句话就必须真实存在。
3. 校验不过的结论**直接丢弃**，不做"温和改写"——改写等于替幻觉背书。
4. 每条留下的结论都带上 (页码, bbox)，前端可以点过去高亮。

已知边界：守卫验的是"引用存在"，不是"引用支撑结论"。
所以这里额外要求模型给出的 quotes 必须来自本轮证据块，
把"引用一段真实但无关的原文"这条路的难度抬高一档。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.guardrails.citation import Claim, Quote, VerifiedClaim, summarize, verify_claims
from app.parsing.chunker import Chunk
from app.parsing.text_index import TextIndex
from app.retrieval.bm25 import Hit

SYSTEM_PROMPT = """你是严谨的论文精读助手，只依据证据块回答。

铁律：
1. 只能使用证据块中出现的内容，禁止引入证据之外的知识，即使你确定它是对的。
2. 每条结论必须附 quotes：从证据块中**逐字复制**的英文原文片段（25–120 词）。
   - 必须与证据块完全一致，不改写、不翻译、不拼接相距很远的句子、不改标点。
   - 抄不出逐字片段，就说明这条结论没有依据，直接不要输出。
3. 证据不足以回答时，输出 {"claims": []}，不要猜。
4. 结论用中文写，分两类词：
   - **专有名词照抄原文**（Transformer、BERT、Multi-Head Attention、BLEU）；
   - **通用词写中文**（"大模型"不写"big 模型"，"注意力头"不写"attention heads"）。
   禁止生造中英混写的词，也不写"大模型（big model）"这种自造的对照括号。

只输出 JSON，不要任何解释文字：
{"claims": [{"text": "中文结论（简洁、可核查）", "quotes": ["英文原文片段"]}]}"""


@dataclass(slots=True)
class Answer:
    question: str
    claims: list[VerifiedClaim]
    evidence: list[Hit]
    raw: dict = field(default_factory=dict)

    @property
    def kept(self) -> list[VerifiedClaim]:
        return [c for c in self.claims if c.status != "unsupported"]

    @property
    def dropped(self) -> list[VerifiedClaim]:
        return [c for c in self.claims if c.status == "unsupported"]

    @property
    def stats(self) -> dict[str, int]:
        return summarize(self.claims)

    def render(self) -> str:
        """人类可读的答案：每条结论后面挂页码。"""
        if not self.kept:
            return "证据不足，未能给出可溯源的结论。"
        lines = []
        for c in self.kept:
            pages = sorted({a.page for a in c.anchors})
            mark = "".join(f"〔p.{p}〕" for p in pages)
            flag = "" if c.status == "supported" else "（原文需人工确认）"
            lines.append(f"- {c.claim.text}{mark}{flag}")
        return "\n".join(lines)


def build_evidence(hits: list[Hit]) -> str:
    """把检索结果拼成证据块。**不截断**——截断是幻觉的第一步。"""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        chunk: Chunk = hit.chunk
        span = (
            f"p.{chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"p.{chunk.page_start}-{chunk.page_end}"
        )
        blocks.append(f"[#{i}] {span} · {chunk.section}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def _parse_claims(raw: dict, doc_id: str) -> list[Claim]:
    claims: list[Claim] = []
    for item in raw.get("claims", []) or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        quotes = []
        for q in item.get("quotes", []) or []:
            if isinstance(q, str) and q.strip():
                quotes.append(Quote(doc=doc_id, text=q.strip()))
        claims.append(Claim(text=text, quotes=quotes))
    return claims


_SECTION_HINT = re.compile(r"\b(?:page|p\.)\s*\d+")


def answer_question(
    question: str,
    *,
    retriever=None,
    hits: list[Hit] | None = None,
    index: TextIndex,
    doc_id: str = "paper",
    k: int = 6,
    llm_call=None,
) -> Answer:
    """端到端问答。`llm_call` 与 `hits` 都可注入，便于离线测试/复用召回结果。"""
    if llm_call is None:
        from app.llm.client import chat_json as llm_call  # 延迟导入，测试不需要网络

    if hits is None:
        hits = retriever.search(question, k=k)
    if not hits:
        return Answer(question=question, claims=[], evidence=[])

    user = f"问题：{question}\n\n证据块：\n\n{build_evidence(hits)}"
    raw = llm_call(user, system=SYSTEM_PROMPT)

    claims = _parse_claims(raw, doc_id)
    verified = verify_claims(claims, {doc_id: index})
    return Answer(question=question, claims=verified, evidence=hits, raw=raw)
