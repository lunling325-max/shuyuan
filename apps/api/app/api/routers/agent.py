"""Agent 编排路由：工具清单、只规划、规划 + 执行。

边界划分与 review/tools 两个路由一致：
- 响应模型定义在本文件，不去改 `schemas.py` 的全站契约；
- registry 在请求时从 `deps` 取，测试替换后立刻生效；
- 库空是请求无法完成（400）；模型不可用由 planner 消化成 200 + degraded。

为什么 `/plan` 和 `/run` 分开：
规划只打一次模型，便宜、快，适合演示"Agent 先想再做"；执行会调用工具、
可能追加步骤，耗时和不确定性都更高。两个端点共用同一套 `plan_task`，
不会出现"演示的计划"和"真正执行的计划"是两套逻辑的尴尬。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.planner import Plan, plan_task, run_task
from app.agent.registry import TOOLS
from app.api import deps

router = APIRouter()


class AgentToolOut(BaseModel):
    name: str
    description: str = Field(description="中文说明，前端可直接展示")
    parameters: dict = Field(description="参数 JSON schema")
    read_only: bool = Field(description="True=只读，不会修改资料库")


class AgentToolsOut(BaseModel):
    tools: list[AgentToolOut]


class AgentPlanRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)


class AgentPlanStepOut(BaseModel):
    id: int
    tool: str
    args: dict
    why: str = Field(description="模型给出的、为什么要这一步")


class AgentPlanOut(BaseModel):
    task: str
    steps: list[AgentPlanStepOut]
    degraded: bool = Field(description="True=模型不可用，steps 是规则式降级计划")
    notes: list[str] = Field(default_factory=list, description="被丢弃的步骤及原因")


class AgentTraceOut(BaseModel):
    step: int = Field(description="执行序号，从 1 开始；0 表示该步骤被丢弃、未执行")
    tool: str
    args: dict
    why: str
    ok: bool
    took_ms: int
    observation_preview: str = Field(description="截断后的观测摘要，前端直接展示")
    dropped_reason: str | None = Field(
        default=None, description="非空表示步骤因工具名/参数不合法被丢弃"
    )


class AgentEvidenceOut(BaseModel):
    paper_id: str
    title: str
    chunk_index: int
    page: int
    quote: str = Field(description="服务端从原文逐字截取，单行、最多 200 字")


class AgentRunOut(BaseModel):
    task: str
    plan: AgentPlanOut = Field(description="初始计划；执行中追加的步骤只出现在 trace 里")
    trace: list[AgentTraceOut]
    answer: str
    evidence: list[AgentEvidenceOut]
    degraded: bool
    notes: list[str] = Field(default_factory=list)


def _plan_out(plan: Plan) -> AgentPlanOut:
    return AgentPlanOut(
        task=plan.task,
        steps=[
            AgentPlanStepOut(id=s.id, tool=s.tool, args=s.args, why=s.why)
            for s in plan.steps
        ],
        degraded=plan.degraded,
        notes=list(plan.notes),
    )


def _papers_or_400() -> list:
    papers = deps.registry.all()
    if not papers:
        raise HTTPException(400, "文献库是空的，请先上传论文再让 Agent 工作")
    return papers


@router.get("/api/agent/tools", response_model=AgentToolsOut)
def agent_tools() -> AgentToolsOut:
    """这个 Agent 会什么：工具名、中文描述、参数 schema。"""
    return AgentToolsOut(tools=[AgentToolOut(**item) for item in TOOLS.describe()])


@router.post("/api/agent/plan", response_model=AgentPlanOut)
def agent_plan(req: AgentPlanRequest) -> AgentPlanOut:
    """只规划不执行：便宜、可解释，用来展示 Task Planning。"""
    papers = _papers_or_400()
    return _plan_out(plan_task(req.task, papers))


@router.post("/api/agent/run", response_model=AgentRunOut)
def agent_run(req: AgentPlanRequest) -> AgentRunOut:
    """规划 + 执行 + 动态追加 + 收尾，返回完整 trace。"""
    papers = _papers_or_400()
    result = run_task(req.task, papers)
    return AgentRunOut(
        task=result.task,
        plan=_plan_out(result.plan),
        trace=[
            AgentTraceOut(
                step=entry.step,
                tool=entry.tool,
                args=entry.args,
                why=entry.why,
                ok=entry.ok,
                took_ms=entry.took_ms,
                observation_preview=entry.observation_preview,
                dropped_reason=entry.dropped_reason,
            )
            for entry in result.trace
        ],
        answer=result.answer,
        evidence=[
            AgentEvidenceOut(
                paper_id=item.paper_id,
                title=item.title,
                chunk_index=item.chunk_index,
                page=item.page,
                quote=item.quote,
            )
            for item in result.evidence
        ],
        degraded=result.degraded,
        notes=list(result.notes),
    )
