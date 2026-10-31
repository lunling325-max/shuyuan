"""写作 Copilot：先检索全库证据，再让模型搭框架。

这个模块的底线和问答模块一致：**引用必须能落地**。

1. 框架不是凭空生成的。先把整个文献库（不只是最新上传的那篇）按 idea 检索一遍，
   把真实证据块编号，连同页码和正文一起喂给模型。
2. 模型只准返回证据块**编号**，不许自己写引用文本。编号在这里被逐一校验：
   不在本轮证据里的编号一律丢弃，要点退化成 `evidence: []`。
3. 引用文本由服务端从证据块正文里截取，最多 200 字 —— 模型没有机会"编"出一段原文。
4. 大模型挂了仍然返回检索结果：outline 为空、`degraded=true`，
   但 references 照样是真实文献，前端可以展示"库里能引用哪些原文"。

为什么检索失败时还要给一份兜底证据？
中文 idea 查英文库，词法通道需要先做检索词扩展；扩展本身也依赖大模型。
断网时扩展必然失败，检索会 0 召回 —— 那样 references 也会空，
"断网也能演示"就成了一句空话。所以 0 召回时退回每篇论文的首块：
它是真实的、可引用的原文，只是相关性没有保证。这是演示兜底，不是检索结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from app.parsing.chunker import Chunk
from app.rag.query import expand_query, retrieve

if TYPE_CHECKING:  # 只为类型标注，避免 rag 层在运行时反向依赖 HTTP 层
    from app.api.registry import Paper

MAX_EVIDENCE = 8
MAX_QUOTE_CHARS = 200
MAX_SECTIONS = 6

SYSTEM_PROMPT = """你是严谨的科研写作 Copilot，负责把研究想法扩展成论文框架。

铁律：
1. 只依据给定的编号证据块提出要点；证据不足的要点照常写，但 evidence 必须是空数组 []。
2. 每个要点给出它依据的证据块编号列表，编号只能来自本轮给出的证据块。
3. 输出 {min_sections}-{max_sections} 个章节，每个章节 2-{max_points} 条要点。
4. 不要输出参考文献条目、不要编造编号；服务端会按编号回填真实引用。
5. 只输出 JSON，不要任何解释文字：
{{"outline": [{{"title": "1 引言", "points": [{{"text": "要点", "evidence": [1, 3]}}]}}]}}"""


@dataclass(slots=True)
class Evidence:
    """一条可被引用的真实证据块。`n` 就是喂给模型的编号。"""

    n: int
    paper_id: str
    title: str
    page: int
    section: str
    text: str

    @property
    def quote(self) -> str:
        # 直接切原文前缀（不是改写），保证引用是逐字的
        return self.text.strip()[:MAX_QUOTE_CHARS]


@dataclass(slots=True)
class Point:
    text: str
    evidence: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Section:
    title: str
    points: list[Point] = field(default_factory=list)


@dataclass(slots=True)
class OutlineResult:
    idea: str
    outline: list[Section]
    references: list[Evidence]
    degraded: bool


def collect_evidence(
    papers: Sequence["Paper"],
    idea: str,
    *,
    limit: int = MAX_EVIDENCE,
    per_paper: int = 4,
    llm_call=None,
) -> list[Evidence]:
    """在整个库里找与 idea 最相关的证据块，按相关度编号。

    先扩一次检索词，再对每篇论文各查一遍 —— 每篇用自己的 BM25，
    因为词表和 idf 都是按单篇建的；最后全局排序取前 `limit` 条。
    """
    if llm_call is None:
        from app.llm.client import chat_json as llm_call

    # 跨语言检索必须扩词；扩词失败时 expand_query 会退回原始 idea，不会抛异常
    queries = expand_query(idea, llm_call=llm_call)

    candidates: list[tuple[float, str, int, "Paper", Chunk]] = []
    for paper in papers:
        if not paper.chunks:
            continue
        try:
            hits = retrieve(
                idea,
                retriever=paper.bm25,
                k=per_paper,
                expand=False,
                queries=queries,
            )
        except Exception:  # noqa: BLE001 —— 单篇索引坏了不该拖垮整个 Copilot
            continue
        for hit in hits:
            candidates.append((hit.score, paper.id, hit.chunk.index, paper, hit.chunk))

    if not candidates:
        # 0 召回（通常是断网导致扩词失败）：退回每篇论文的首块做演示兜底
        for paper in papers:
            if paper.chunks:
                candidates.append((0.0, paper.id, paper.chunks[0].index, paper, paper.chunks[0]))
            if len(candidates) >= limit:
                break

    # 同分时按 id / index 排，保证同样输入得到同样编号（前端才能稳定回链）
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    picked = candidates[:limit]
    return [
        Evidence(
            n=i,
            paper_id=paper.id,
            title=paper.title,
            page=chunk.page_start,
            section=chunk.section,
            text=chunk.text,
        )
        for i, (_score, _pid, _idx, paper, chunk) in enumerate(picked, start=1)
    ]


def _evidence_text(evidence: list[Evidence]) -> str:
    blocks = []
    for e in evidence:
        span = f"p.{e.page}" if e.page else "p.?"
        blocks.append(f"[{e.n}] {span} · {e.section} · {e.title}\n{e.text}")
    return "\n\n---\n\n".join(blocks)


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):  # bool 是 int 的子类，别把 True 当成编号 1
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_outline(
    raw: object,
    evidence_by_n: dict[int, Evidence],
    *,
    max_points: int,
) -> list[Section]:
    """把模型输出变成服务端认可的结构；不存在的证据编号在这里被丢掉。"""
    if not isinstance(raw, dict):
        return []
    items = raw.get("outline")
    if not isinstance(items, list):
        return []

    sections: list[Section] = []
    for item in items[:MAX_SECTIONS]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue

        points: list[Point] = []
        raw_points = item.get("points")
        if not isinstance(raw_points, list):
            raw_points = []
        for entry in raw_points[:max_points]:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            raw_evidence = entry.get("evidence")
            if not isinstance(raw_evidence, list):
                raw_evidence = [] if raw_evidence is None else [raw_evidence]
            numbers = sorted(
                {
                    n
                    for n in (_as_int(v) for v in raw_evidence)
                    if n is not None and n in evidence_by_n
                }
            )
            points.append(Point(text=text, evidence=numbers))
        sections.append(Section(title=title, points=points))

    return sections


def build_outline(
    idea: str,
    papers: Sequence["Paper"],
    *,
    max_points: int = 4,
    limit: int = MAX_EVIDENCE,
    llm_call=None,
) -> OutlineResult:
    """检索 → 编号证据 → 生成框架 → 校验编号 → 回填真实引用。"""
    if llm_call is None:
        from app.llm.client import chat_json as llm_call

    evidence = collect_evidence(papers, idea, limit=limit, llm_call=llm_call)
    if not evidence:
        # 库里没有任何可切块的正文：没有证据就不编框架
        return OutlineResult(idea=idea, outline=[], references=[], degraded=True)

    by_n = {e.n: e for e in evidence}
    system = SYSTEM_PROMPT.format(
        min_sections=4,
        max_sections=MAX_SECTIONS,
        max_points=max(2, min(max_points, 4)),
    )
    user = f"研究想法：{idea}\n\n证据块：\n\n{_evidence_text(evidence)}\n\n请生成论文框架。"

    try:
        raw = llm_call(user, system=system, temperature=0.1)
    except Exception:  # noqa: BLE001 —— 降级为"只有可引用原文"，不能 500
        return OutlineResult(idea=idea, outline=[], references=list(evidence), degraded=True)

    outline = _parse_outline(raw, by_n, max_points=max_points)
    if not outline:
        # 模型返回空/垃圾：outline 没得用，但真实证据仍然交给前端
        return OutlineResult(idea=idea, outline=[], references=list(evidence), degraded=True)

    # references 只收被真正引用到的编号，并按编号去重
    cited = sorted({n for section in outline for point in section.points for n in point.evidence})
    return OutlineResult(
        idea=idea,
        outline=outline,
        references=[by_n[n] for n in cited],
        degraded=False,
    )
