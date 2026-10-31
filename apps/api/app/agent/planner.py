"""两段式规划 + 执行循环：先给一张可解释的计划，再让模型边走边决定。

为什么不直接"一次性出完整计划然后机械执行"？
真实任务里，第一步检索到什么会改变第二步该做什么。只用静态计划，Agent
看起来像在跑脚本；每一步都把观测喂回去、允许追加步骤，才符合评分表里
"Task Planning / 动态规划"的语义。

但也不能让模型无限追加 —— 这里有三道闸：
1. 初始计划最多 8 步，超了截断；
2. 每次追加都重新校验工具名和参数，坏步骤丢弃并记入 trace；
3. 总执行步数硬上限，到顶后强制收尾，模型再怎么要求 call 也不会执行。

两种模式都保留：
- `/api/agent/plan` 只跑 `plan_task`，便宜、可解释，适合演示"它是怎么想的"；
- `/api/agent/run` 先展示初始计划，再执行并在每一步后询问模型
  `{"action": "call"}` 或 `{"action": "finish"}`，trace 记录全过程。

抗幻觉沿用项目铁律：模型只能给 `(paper_id, chunk_index)` 指针，而且这些
指针必须落在**本轮工具真正观测过**的证据块里；quote 由服务端从 chunk
正文逐字截取。模型编出来的 id、类型不对的 chunk_index 一律丢弃并计数。

降级底线：模型不可用时返回 200 + `degraded=True` + 规则式计划
（直接拿任务原文做 `search_library`），并用工具观测拼出可核对的结论。
库为空由路由层提前返回 400，不在这里处理。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agent.registry import (
    TOOLS,
    ObservedEvidence,
    ToolContext,
    ToolOutcome,
    ToolRegistry,
    args_to_json,
    tool_catalog_text,
    truncate,
)

MAX_PLAN_STEPS = 8
MAX_RUN_STEPS = 12
MAX_PREVIEW_CHARS = 280
MAX_ANSWER_CHARS = 4000
MAX_TASK_CHARS = 2000
MAX_WHY_CHARS = 200

PLAN_SYSTEM_PROMPT = """你是科研助手 Agent 的规划器。只输出 JSON，不要解释文字。

规则：
1. 只能使用工具目录里出现的工具名，参数必须严格符合给出的 schema。
2. 每一步只做一件事；步骤数最多 8 步，按执行顺序排列。
3. 每一步都必须写 why：用一句中文说明"为什么需要这一步、它要回答什么"。
4. 不要编造 paper_id。初始计划可以先用 search_library / list_papers 发现文献，
   再决定是否 answer_question / get_paper / compare_versions。
5. 只读检索：search_arxiv 不会下载或入库，别把它当作获取全文的手段。

输出格式：
{"steps": [{"id": 1, "tool": "search_library", "args": {"query": "attention", "k": 5}, "why": "先定位相关证据块"}]}"""

DECIDE_SYSTEM_PROMPT = """你是科研助手 Agent 的执行器。你刚执行完一个工具步骤，现在决定下一步。

只输出 JSON，二选一：
1. 继续调用工具：
{"action": "call", "tool": "工具名", "args": {...}, "why": "为什么还要这一步"}
2. 收尾：
{"action": "finish", "answer": "中文结论，不要自己写引用原文", "evidence": [{"paper_id": "id", "chunk_index": 0}]}

铁律：
- 工具名和参数必须来自给出的工具目录。
- evidence 只能引用上面观测里真实出现过的 [paper_id #chunk_index]；
  服务端会重新校验，编造的指针会被丢弃。
- answer 里不要粘贴"原文引用"，引用统一放在 evidence 指针里。
- 信息已经足够就 finish，不要为了多调工具而继续。"""


@dataclass(slots=True)
class PlanStep:
    id: int
    tool: str
    args: dict[str, Any]
    why: str = ""


@dataclass(slots=True)
class DroppedStep:
    tool: str
    args: dict[str, Any]
    why: str
    reason: str


@dataclass(slots=True)
class Plan:
    task: str
    steps: list[PlanStep] = field(default_factory=list)
    degraded: bool = False
    notes: list[str] = field(default_factory=list)
    dropped: list[DroppedStep] = field(default_factory=list)


@dataclass(slots=True)
class TraceEntry:
    step: int
    tool: str
    args: dict[str, Any]
    why: str
    ok: bool
    took_ms: int
    observation_preview: str
    dropped_reason: str | None = None


@dataclass(slots=True)
class AgentRun:
    task: str
    plan: Plan
    trace: list[TraceEntry]
    answer: str
    evidence: list[ObservedEvidence]
    degraded: bool
    notes: list[str]
    dropped_evidence: int = 0


@dataclass(slots=True)
class _Decision:
    kind: str  # call | finish | drop | invalid
    step: PlanStep | None = None
    answer: str = ""
    evidence: list[Any] = field(default_factory=list)
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    why: str = ""
    reason: str = ""


# ---------------------------------------------------------------- 小工具


def _resolve_llm(llm_call: Callable[..., dict] | None) -> Callable[..., dict]:
    """延迟导入真实客户端。

    测试 monkeypatch `app.llm.client.chat_json` 后，下一次调用就能命中假函数；
    如果写成模块级 `from ... import chat_json`，替换发生在 import 之后就不生效。
    """
    if llm_call is not None:
        return llm_call
    from app.llm.client import chat_json

    return chat_json


def _clamp(value: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(number, high))


def _one_line(value: object) -> str:
    return truncate(str(value) or type(value).__name__, 200)


def _papers_text(papers: list) -> str:
    if not papers:
        return "（库为空）"
    lines = []
    for paper in sorted(papers, key=lambda item: item.id)[:20]:
        lines.append(
            f"- {paper.id} | 《{getattr(paper, 'title', '')}》 | "
            f"{getattr(paper, 'pages', 0)} 页 / {len(getattr(paper, 'chunks', []) or [])} 块"
        )
    if len(papers) > 20:
        lines.append(f"……另有 {len(papers) - 20} 篇，未在提示词里展开")
    return "\n".join(lines)


def fallback_plan(
    task: str, tools: ToolRegistry | None = None, *, max_steps: int = MAX_PLAN_STEPS
) -> list[PlanStep]:
    """模型不可用时的规则式计划。

    只选最稳的一步：拿任务原文去查全库 BM25。中文任务查英文库可能 0 召回，
    但这是**可以解释的 0 召回**，比"编一个答案"诚实，也比空白页有用。
    """
    tools = tools if tools is not None else TOOLS
    if max_steps < 1 or tools.get("search_library") is None:
        return []
    query = " ".join(task.split())[:200]
    return [
        PlanStep(
            id=1,
            tool="search_library",
            args={"query": query, "k": 5},
            why="模型不可用：按任务原文直接做全库关键词检索",
        )
    ]


# ---------------------------------------------------------------- 规划


def plan_task(
    task: str,
    papers: list,
    *,
    llm_call: Callable[..., dict] | None = None,
    tools: ToolRegistry | None = None,
    max_steps: int = MAX_PLAN_STEPS,
) -> Plan:
    """让模型产出初始计划，并由服务端逐条校验、截断。"""
    tools = tools if tools is not None else TOOLS
    call = _resolve_llm(llm_call)
    limit = _clamp(max_steps, 1, MAX_RUN_STEPS)

    user = (
        f"任务：{task}\n\n"
        f"资料库：\n{_papers_text(papers)}\n\n"
        f"可用工具：\n{tool_catalog_text(tools)}\n\n"
        "请给出执行计划。"
    )
    try:
        raw = call(user, system=PLAN_SYSTEM_PROMPT, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 —— 任何模型侧失败都走降级
        reason = _one_line(exc)
        return Plan(
            task=task,
            steps=fallback_plan(task, tools, max_steps=limit),
            degraded=True,
            notes=[
                f"大模型不可用（{reason}），已改用规则式降级计划（search_library）。"
            ],
        )

    return _parse_plan(task, raw, tools, limit)


def _parse_plan(
    task: str, raw: object, tools: ToolRegistry, max_steps: int
) -> Plan:
    if not isinstance(raw, dict):
        return Plan(task=task, notes=["模型输出不是 JSON 对象，计划为空。"])

    items = raw.get("steps")
    if not isinstance(items, list):
        return Plan(task=task, notes=["模型输出里没有 steps 数组，计划为空。"])

    notes: list[str] = []
    if len(items) > max_steps:
        notes.append(
            f"模型给出 {len(items)} 个步骤，超过上限 {max_steps}，已截断。"
        )
        items = items[:max_steps]

    steps: list[PlanStep] = []
    dropped: list[DroppedStep] = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            reason = "步骤不是 JSON 对象"
            notes.append(f"丢弃第 {position} 个步骤：{reason}。")
            dropped.append(DroppedStep(tool="", args={}, why="", reason=reason))
            continue

        tool = item.get("tool")
        raw_args = item.get("args", {})
        why = truncate(str(item.get("why") or "").strip(), MAX_WHY_CHARS)
        spec, parsed, reason = tools.validate(tool, raw_args)
        if reason:
            tool_name = tool if isinstance(tool, str) else "?"
            notes.append(f"丢弃第 {position} 个步骤（{tool_name}）：{reason}。")
            dropped.append(
                DroppedStep(
                    tool=tool if isinstance(tool, str) else "",
                    args=raw_args if isinstance(raw_args, dict) else {},
                    why=why,
                    reason=reason,
                )
            )
            continue

        assert spec is not None and parsed is not None
        # id 用"通过校验后的顺序"重新编号：模型给的 id 可能重复或跳号，
        # 前端按 id 渲染执行顺序时会错位。
        steps.append(
            PlanStep(
                id=len(steps) + 1,
                tool=spec.name,
                args=parsed.model_dump(),
                why=why,
            )
        )

    if not steps:
        notes.append("没有通过校验的可执行步骤。")
    return Plan(task=task, steps=steps, notes=notes, dropped=dropped)


# ---------------------------------------------------------------- 执行决策


def _decision_user(
    task: str,
    trace: list[TraceEntry],
    tools: ToolRegistry,
    remaining: int,
) -> str:
    lines = [
        f"任务：{task}",
        "",
        "可用工具：",
        tool_catalog_text(tools),
        "",
        "已执行步骤与观测（evidence 只能引用这里出现过的 [paper_id #chunk_index]）：",
    ]
    executed = [entry for entry in trace if entry.step > 0]
    if not executed:
        lines.append("（尚未执行任何工具）")
    for entry in executed:
        lines.append(
            f"[{entry.step}] {entry.tool}({args_to_json(entry.args)}) -> "
            f"{'成功' if entry.ok else '失败'}"
        )
        lines.append(entry.observation_preview or "（无观测）")
        lines.append("")

    dropped = [entry for entry in trace if entry.step <= 0]
    for entry in dropped:
        lines.append(f"[被丢弃] {entry.tool}：{entry.dropped_reason or '参数不合法'}")
    if dropped:
        lines.append("")

    lines.append(f"剩余可执行步数：{remaining}。")
    lines.append("信息足够就 finish；否则给出下一步 call。")
    return "\n".join(lines)


def _ask_decision(
    task: str,
    trace: list[TraceEntry],
    tools: ToolRegistry,
    call: Callable[..., dict],
    remaining: int,
) -> tuple[object | None, str]:
    try:
        raw = call(
            _decision_user(task, trace, tools, remaining),
            system=DECIDE_SYSTEM_PROMPT,
            temperature=0.0,
        )
        return raw, ""
    except Exception as exc:  # noqa: BLE001
        return None, _one_line(exc)


def _parse_decision(raw: object, tools: ToolRegistry) -> _Decision:
    if not isinstance(raw, dict):
        return _Decision(kind="invalid", reason="模型决策不是 JSON 对象")

    action = str(raw.get("action") or "").strip().lower()
    if action == "finish":
        answer = raw.get("answer")
        evidence = raw.get("evidence")
        return _Decision(
            kind="finish",
            answer=answer if isinstance(answer, str) else "",
            evidence=evidence if isinstance(evidence, list) else [],
        )

    if action == "call":
        tool = raw.get("tool")
        raw_args = raw.get("args", {})
        why = truncate(str(raw.get("why") or "").strip(), MAX_WHY_CHARS)
        spec, parsed, reason = tools.validate(tool, raw_args)
        if reason:
            return _Decision(
                kind="drop",
                tool=tool if isinstance(tool, str) else "",
                args=raw_args if isinstance(raw_args, dict) else {},
                why=why,
                reason=reason,
            )
        assert spec is not None and parsed is not None
        return _Decision(
            kind="call",
            step=PlanStep(
                id=0,  # 追加步骤的 id 不影响 trace；trace 用执行序号
                tool=spec.name,
                args=parsed.model_dump(),
                why=why,
            ),
        )

    return _Decision(kind="invalid", reason="action 必须是 call 或 finish")


# ---------------------------------------------------------------- 证据回填


def _resolve_evidence(
    raw: list[Any], observed: dict[tuple[str, int], ObservedEvidence]
) -> tuple[list[ObservedEvidence], int]:
    """把模型给的指针映射回真实证据块。

    只认本轮工具观测过的块，而不是"库里任意存在的块"。理由：模型只在观测里
    见过这些 chunk_index，如果它引用了没见过的块，即使那个块真实存在，
    也无法说明结论是从它看到的材料推出来的。稳妥做法是一律丢弃。
    """
    dropped = 0
    picked: dict[tuple[str, int], ObservedEvidence] = {}

    for item in raw:
        if not isinstance(item, dict):
            dropped += 1
            continue
        paper_id = item.get("paper_id")
        chunk_index = item.get("chunk_index")
        if (
            not isinstance(paper_id, str)
            or not paper_id.strip()
            or isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
        ):
            dropped += 1
            continue
        evidence = observed.get((paper_id, chunk_index))
        if evidence is None:
            dropped += 1
            continue
        # 重复指针不算"编造"，去重即可
        picked.setdefault((paper_id, chunk_index), evidence)

    ordered = [picked[key] for key in sorted(picked)]
    return ordered, dropped


def _local_answer(trace: list[TraceEntry]) -> str:
    """没有模型结论时，用工具观测拼一个可核对的收尾。

    不生成任何"论文里没说过"的判断，只把摘要摆出来。降级时可以难看，
    但不能假装是大模型给的结论。
    """
    executed = [entry for entry in trace if entry.step > 0]
    if not executed:
        return "没有执行任何工具步骤，无法给出结论。"
    lines = [
        f"共执行 {len(executed)} 个工具步骤。模型未给出最终结论，"
        "以下为可核对的工具观测摘要："
    ]
    for entry in executed[:6]:
        text = " ".join(entry.observation_preview.split())
        lines.append(f"- {entry.tool}：{text[:180]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 运行


def run_task(
    task: str,
    papers: list,
    *,
    llm_call: Callable[..., dict] | None = None,
    tools: ToolRegistry | None = None,
    max_steps: int = MAX_PLAN_STEPS,
    arxiv_fetch: Callable[[str], bytes] | None = None,
) -> AgentRun:
    """规划 → 执行 → 观测回喂 → 动态追加/收尾。"""
    tools = tools if tools is not None else TOOLS
    limit = _clamp(max_steps, 1, MAX_RUN_STEPS)
    call = _resolve_llm(llm_call)

    plan = plan_task(task, papers, llm_call=call, tools=tools, max_steps=limit)
    notes = list(plan.notes)
    degraded = plan.degraded

    context = ToolContext(papers=papers, llm_call=call, arxiv_fetch=arxiv_fetch)
    trace: list[TraceEntry] = []

    # 计划阶段丢弃的步骤也进 trace：前端要能显示"模型想调一个不存在的工具，
    # 我们拦下了"。step=0 表示它没有真正执行。
    for item in plan.dropped:
        trace.append(
            TraceEntry(
                step=0,
                tool=item.tool,
                args=item.args,
                why=item.why,
                ok=False,
                took_ms=0,
                observation_preview="",
                dropped_reason=item.reason,
            )
        )

    observed: dict[tuple[str, int], ObservedEvidence] = {}
    queue = list(plan.steps)
    executed = 0
    llm_available = not plan.degraded
    finish_answer: str | None = None
    finish_evidence: list[Any] | None = None

    while queue and executed < limit:
        step = queue.pop(0)
        started = time.perf_counter()
        outcome: ToolOutcome = tools.execute(step.tool, step.args, context)
        took_ms = max(0, int((time.perf_counter() - started) * 1000))
        executed += 1

        trace.append(
            TraceEntry(
                step=executed,
                tool=step.tool,
                args=step.args,
                why=step.why,
                ok=outcome.ok,
                took_ms=took_ms,
                observation_preview=truncate(
                    outcome.summary, MAX_PREVIEW_CHARS
                ),
                dropped_reason=outcome.dropped_reason or None,
            )
        )
        for evidence in outcome.evidence:
            observed.setdefault(evidence.key, evidence)

        # 计划本身就是降级来的，说明模型不可用，别再对每一步都打一次必然失败的请求
        if not llm_available:
            continue

        raw, error = _ask_decision(
            task, trace, tools, call, remaining=limit - executed
        )
        if error:
            degraded = True
            llm_available = False
            notes.append(
                f"第 {executed} 步后大模型决策失败（{error}），"
                "改用剩余预设计划执行。"
            )
            continue

        decision = _parse_decision(raw, tools)
        if decision.kind == "finish":
            finish_answer = decision.answer
            finish_evidence = decision.evidence
            if queue:
                notes.append(
                    f"模型在第 {executed} 步后请求收尾，跳过剩余 "
                    f"{len(queue)} 个预设计划步骤。"
                )
            break

        if decision.kind == "call":
            if executed >= limit:
                notes.append(f"达到总步数上限（{limit}），不再追加步骤。")
                continue
            assert decision.step is not None
            queue.append(decision.step)  # 追加到队尾：先跑完初始计划，再跑补充步骤
            continue

        if decision.kind == "drop":
            notes.append(
                f"模型追加的步骤被丢弃（{decision.tool or '?'}）：{decision.reason}"
            )
            trace.append(
                TraceEntry(
                    step=0,
                    tool=decision.tool,
                    args=decision.args,
                    why=decision.why,
                    ok=False,
                    took_ms=0,
                    observation_preview="",
                    dropped_reason=decision.reason,
                )
            )
            continue

        # 模型返回了无法解析的决策：不猜，按剩余预设计划继续
        notes.append(
            f"第 {executed} 步后模型未给出有效决策（{decision.reason}），"
            "按剩余预设计划继续。"
        )

    if queue and executed >= limit:
        notes.append(
            f"达到总步数上限（{limit}），已强制收尾，"
            f"不再执行剩余 {len(queue)} 步。"
        )

    has_model_answer = finish_answer is not None and bool(finish_answer.strip())
    answer = (
        truncate(finish_answer, MAX_ANSWER_CHARS)
        if has_model_answer
        else _local_answer(trace)
    )

    # 模型给指针就只信它给的；模型没给最终结论（降级/到顶/空字符串）时，
    # 自动挂上本轮排名最靠前的观测证据，至少让结论可点回原文。
    if finish_evidence is not None and (finish_evidence or has_model_answer):
        evidence, dropped_evidence = _resolve_evidence(finish_evidence, observed)
    elif not has_model_answer:
        pointers = [
            {"paper_id": item.paper_id, "chunk_index": item.chunk_index}
            for item in sorted(observed.values(), key=lambda ev: ev.key)[:5]
        ]
        evidence, dropped_evidence = _resolve_evidence(pointers, observed)
    else:
        evidence, dropped_evidence = [], 0

    if dropped_evidence:
        notes.append(
            f"剔除 {dropped_evidence} 条编造或未经本轮观测的引用。"
        )
    if degraded:
        notes.append("降级运行：结论来自工具观测，未经大模型总结。")

    return AgentRun(
        task=task,
        plan=plan,
        trace=trace,
        answer=answer,
        evidence=evidence,
        degraded=degraded,
        notes=notes,
        dropped_evidence=dropped_evidence,
    )
