"""Agent 编排接口的测试（全程离线）。

重点不是"回答得好不好"，而是六条底线：
1. 正常规划 + 执行，trace 完整，动态追加的步骤也留下痕迹；
2. 模型编造工具名 / 参数不合法 -> 步骤被丢弃，绝不执行；
3. 编造的 paper_id / chunk_index 引用被剔除，quote 来自真实 chunk；
4. 模型不可用 -> 200 + degraded + 规则式降级计划；
5. 模型一直要求继续 -> 到总步数上限后强制收尾，不死循环；
6. 工具抛异常 -> 该步 ok=false，整个请求仍然 200。

模型全部用脚本式假函数注入，不依赖 DEEPSEEK_API_KEY，也不发任何网络包。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# 本模块会较早导入 app.api.deps（它按 import 时的 PAPER_STORE_DIR 建 registry）。
# 若不先指定目录，registry 会绑到项目根目录下真实的 data/papers（开发服务的
# 数据），后面的 test_api.py 再设 PAPER_STORE_DIR 就已经晚了，会看到别人的库。
os.environ.setdefault(
    "PAPER_STORE_DIR", tempfile.mkdtemp(prefix="papers_agent_env_")
)

ROOT = Path(__file__).resolve().parents[3]
PDF_V7 = ROOT / "pdf" / "1706.03762v7.pdf"
PDF_BERT = ROOT / "pdf" / "1810.04805_BERT.pdf"

REQUIRED_TOOLS = {
    "search_library",
    "get_paper",
    "answer_question",
    "compare_versions",
    "search_arxiv",
    "citation_graph",
    "find_main_papers",
}


class ScriptedLLM:
    """按顺序返回预设响应，并记录每次调用的提示词。

    脚本用尽时主动失败：如果实现偷偷多打了模型，测试会立刻暴露，
    而不是安静地拿到一个默认值。
    """

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def __call__(self, user: str, *, system=None, **kwargs):
        self.calls.append(user)
        if not self.responses:
            raise AssertionError("假模型调用次数超出脚本，说明执行循环多打了一次")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """两篇真实 PDF，和 test_review.py 用同一套灌库方式。"""
    from app.api.registry import PaperRegistry

    reg = PaperRegistry(tmp_path_factory.mktemp("papers_agent"))
    for path in (PDF_V7, PDF_BERT):
        reg.add(path.read_bytes(), path.name)
    return reg


@pytest.fixture()
def client(registry, monkeypatch):
    from app.api import deps
    from app.api.routers.agent import router

    monkeypatch.setattr(deps, "registry", registry)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _inject_llm(monkeypatch, script: ScriptedLLM) -> ScriptedLLM:
    # planner 在函数内部 import chat_json，所以替换模块属性对下一次调用生效
    monkeypatch.setattr("app.llm.client.chat_json", script)
    return script


def _top_search_hit(papers, query: str, k: int = 3):
    """复现 search_library 的排序，供测试构造一个"一定被观测过"的指针。"""
    scored = []
    for paper in papers:
        for hit in paper.bm25.search(query, k=k):
            scored.append((float(hit.score), paper, hit.chunk))
    scored.sort(key=lambda item: (-item[0], item[1].id, item[2].index))
    return scored[0]


def _flatten(text: str) -> str:
    return " ".join(text.split())


# ------------------------------------------------------------ 工具清单


def test_tools_endpoint_lists_real_capabilities(client):
    r = client.get("/api/agent/tools")

    assert r.status_code == 200, r.text
    tools = r.json()["tools"]
    names = {item["name"] for item in tools}
    assert REQUIRED_TOOLS <= names
    for item in tools:
        assert item["description"], "每个工具都要有中文说明给前端展示"
        assert isinstance(item["parameters"], dict)
        assert item["read_only"] is True


# ------------------------------------------------------------ 只规划


def test_plan_only_returns_validated_plan_without_executing(client, monkeypatch):
    script = _inject_llm(
        monkeypatch,
        ScriptedLLM(
            [
                {
                    "steps": [
                        {
                            "id": 99,
                            "tool": "search_library",
                            "args": {"query": "attention", "k": 2},
                            "why": "先找证据",
                        }
                    ]
                }
            ]
        ),
    )

    r = client.post("/api/agent/plan", json={"task": "总结注意力机制"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["degraded"] is False
    assert [s["tool"] for s in body["steps"]] == ["search_library"]
    # id 由服务端按通过校验后的顺序重排，不信模型给的 99
    assert body["steps"][0]["id"] == 1
    assert "trace" not in body, "只规划不应该执行任何工具"
    assert len(script.calls) == 1, "只规划只该打一次模型"


# ------------------------------------------------------------ 正常执行


def test_normal_run_executes_plan_appends_step_and_backfills_evidence(
    client, registry, monkeypatch
):
    papers = registry.all()
    _score, paper, chunk = _top_search_hit(papers, "attention", k=3)

    script = _inject_llm(
        monkeypatch,
        ScriptedLLM(
            [
                {
                    "steps": [
                        {
                            "id": 1,
                            "tool": "search_library",
                            "args": {"query": "attention", "k": 3},
                            "why": "先定位相关证据块",
                        }
                    ]
                },
                {
                    "action": "call",
                    "tool": "list_papers",
                    "args": {},
                    "why": "补充资料库清单",
                },
                {
                    "action": "finish",
                    "answer": "注意力机制是 Transformer 的核心。",
                    "evidence": [
                        {"paper_id": paper.id, "chunk_index": chunk.index},
                        {"paper_id": "ghost-paper", "chunk_index": 0},
                    ],
                },
            ]
        ),
    )

    r = client.post("/api/agent/run", json={"task": "解释 attention 的作用"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["degraded"] is False
    assert [s["tool"] for s in body["plan"]["steps"]] == ["search_library"]

    trace = body["trace"]
    assert [t["step"] for t in trace] == [1, 2]
    assert [t["tool"] for t in trace] == ["search_library", "list_papers"]
    assert all(t["ok"] for t in trace)
    assert all(t["observation_preview"] for t in trace)
    assert all(t["took_ms"] >= 0 for t in trace)
    assert all(t["dropped_reason"] is None for t in trace)
    assert trace[1]["why"] == "补充资料库清单", "动态追加的 why 也要保留"

    # 编造的 paper_id 被剔除，真实指针保留
    assert len(body["evidence"]) == 1
    ref = body["evidence"][0]
    assert ref["paper_id"] == paper.id
    assert ref["chunk_index"] == chunk.index
    assert 0 < len(ref["quote"]) <= 200
    assert "\n" not in ref["quote"], "quote 必须压成单行"
    assert ref["quote"] in _flatten(chunk.text), "quote 必须是原文逐字片段"
    assert any("剔除 1 条" in note for note in body["notes"])

    assert body["answer"] == "注意力机制是 Transformer 的核心。"
    assert len(script.calls) == 3


# ------------------------------------------------------------ 坏步骤丢弃


def test_invented_tool_and_invalid_args_are_dropped_not_executed(
    client, monkeypatch
):
    script = _inject_llm(
        monkeypatch,
        ScriptedLLM(
            [
                {
                    "steps": [
                        {
                            "id": 1,
                            "tool": "hack_database",
                            "args": {"target": "all"},
                            "why": "模型编出来的工具",
                        },
                        {
                            "id": 2,
                            "tool": "search_library",
                            "args": {"query": ""},
                            "why": "缺少有效检索词",
                        },
                        {
                            "id": 3,
                            "tool": "get_paper",
                            "args": {"paper_id": ["not", "a", "string"]},
                            "why": "参数类型错误",
                        },
                        {
                            "id": 4,
                            "tool": "list_papers",
                            "args": {},
                            "why": "唯一合法的步骤",
                        },
                    ]
                },
                {
                    "action": "finish",
                    "answer": "只执行了合法步骤。",
                    "evidence": [],
                },
            ]
        ),
    )

    r = client.post("/api/agent/run", json={"task": "演示坏步骤拦截"})

    assert r.status_code == 200, r.text
    body = r.json()
    assert [s["tool"] for s in body["plan"]["steps"]] == ["list_papers"]

    executed = [t for t in body["trace"] if t["step"] > 0]
    dropped = [t for t in body["trace"] if t["step"] == 0]
    assert [t["tool"] for t in executed] == ["list_papers"]
    assert executed[0]["ok"] is True
    assert {t["tool"] for t in dropped} == {
        "hack_database",
        "search_library",
        "get_paper",
    }
    assert all(t["ok"] is False for t in dropped)
    assert all(t["took_ms"] == 0 for t in dropped)
    assert all(t["dropped_reason"] for t in dropped)
    assert any("未知工具：hack_database" in note for note in body["notes"])
    assert any("query" in note for note in body["notes"])
    assert len(script.calls) == 2, "坏步骤不该产生额外的模型调用或执行"


# ------------------------------------------------------------ 引用抗幻觉


def test_invented_evidence_pointers_are_dropped_and_quote_comes_from_chunk(
    client, registry, monkeypatch
):
    _score, paper, chunk = _top_search_hit(registry.all(), "attention", k=3)

    _inject_llm(
        monkeypatch,
        ScriptedLLM(
            [
                {
                    "steps": [
                        {
                            "tool": "search_library",
                            "args": {"query": "attention", "k": 3},
                            "why": "找证据",
                        }
                    ]
                },
                {
                    "action": "finish",
                    "answer": "结论。",
                    "evidence": [
                        {"paper_id": paper.id, "chunk_index": chunk.index},
                        {"paper_id": paper.id, "chunk_index": 99999},
                        {"paper_id": "invented", "chunk_index": 0},
                        {"paper_id": paper.id, "chunk_index": str(chunk.index)},
                    ],
                },
            ]
        ),
    )

    body = client.post("/api/agent/run", json={"task": "attention"}).json()

    assert len(body["evidence"]) == 1
    ref = body["evidence"][0]
    assert (ref["paper_id"], ref["chunk_index"]) == (paper.id, chunk.index)
    assert ref["quote"] in _flatten(paper.chunks[chunk.index].text)
    assert any("剔除 3 条" in note for note in body["notes"])


# ------------------------------------------------------------ 降级


def test_llm_outage_returns_200_with_rule_based_plan(client, registry, monkeypatch):
    from app.llm.client import LLMError

    def down(*args, **kwargs):
        raise LLMError("缺少环境变量 DEEPSEEK_API_KEY")

    monkeypatch.setattr("app.llm.client.chat_json", down)

    r = client.post("/api/agent/run", json={"task": "attention"})

    assert r.status_code == 200, "模型不可用不能 500"
    body = r.json()
    assert body["degraded"] is True
    assert [s["tool"] for s in body["plan"]["steps"]] == ["search_library"]
    assert body["plan"]["steps"][0]["args"]["query"] == "attention"

    executed = [t for t in body["trace"] if t["step"] > 0]
    assert len(executed) == 1
    assert executed[0]["ok"] is True, "降级计划仍然要真的执行工具"
    assert body["answer"], "断网时也要给出可读的收尾"
    assert body["evidence"], "工具真实观测到的证据仍要返回"
    for ref in body["evidence"]:
        real_chunk = registry.get(ref["paper_id"]).chunks[ref["chunk_index"]]
        assert ref["quote"] in _flatten(real_chunk.text), "quote 必须来自原文"
        assert 0 < len(ref["quote"]) <= 200
    assert any("大模型不可用" in note for note in body["notes"])
    assert any("降级" in note for note in body["notes"])


# ------------------------------------------------------------ 死循环防护


def test_model_that_always_calls_is_stopped_at_step_limit(client, monkeypatch):
    state = {"calls": 0}

    def loop_llm(user, *, system=None, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return {
                "steps": [
                    {
                        "tool": "search_library",
                        "args": {"query": "attention", "k": 2},
                        "why": "先检索",
                    }
                ]
            }
        return {
            "action": "call",
            "tool": "list_papers",
            "args": {},
            "why": "继续调用，永不停",
        }

    monkeypatch.setattr("app.llm.client.chat_json", loop_llm)

    r = client.post("/api/agent/run", json={"task": "死循环测试"})

    assert r.status_code == 200, r.text
    body = r.json()
    executed = [t for t in body["trace"] if t["step"] > 0]
    assert len(executed) == 8, "总执行步数必须有硬上限"
    assert executed[0]["tool"] == "search_library"
    assert all(t["tool"] == "list_papers" for t in executed[1:])
    assert [t["step"] for t in executed] == list(range(1, 9))
    assert any("上限" in note for note in body["notes"])
    assert body["answer"], "触顶后仍要收尾，不能留空"
    # 1 次规划 + 8 次决策；触顶后那次决策不会再换来第 9 步
    assert state["calls"] == 9


# ------------------------------------------------------------ 工具异常


def test_tool_exception_becomes_failed_observation_and_request_stays_200(
    client, monkeypatch
):
    from app.agent import planner
    from app.agent.registry import ToolSpec, build_tool_registry

    tools = build_tool_registry()
    base = tools.get("search_library")
    assert base is not None

    def boom(ctx, args):
        raise RuntimeError("索引文件损坏")

    tools.register(
        ToolSpec(
            name=base.name,
            description=base.description,
            args_model=base.args_model,
            handler=boom,
            read_only=base.read_only,
        )
    )
    monkeypatch.setattr(planner, "TOOLS", tools)

    _inject_llm(
        monkeypatch,
        ScriptedLLM(
            [
                {
                    "steps": [
                        {
                            "tool": "search_library",
                            "args": {"query": "attention", "k": 3},
                            "why": "这一步会抛异常",
                        }
                    ]
                },
                {
                    "action": "finish",
                    "answer": "工具失败但已收尾。",
                    "evidence": [],
                },
            ]
        ),
    )

    r = client.post("/api/agent/run", json={"task": "工具异常测试"})

    assert r.status_code == 200, r.text
    body = r.json()
    executed = [t for t in body["trace"] if t["step"] > 0]
    assert len(executed) == 1
    assert executed[0]["ok"] is False
    assert "工具执行失败" in executed[0]["observation_preview"]
    assert body["degraded"] is False, "单个工具失败不是模型不可用"
    assert body["answer"] == "工具失败但已收尾。"


# ------------------------------------------------------------ 空库


def test_empty_library_returns_400(client, monkeypatch, tmp_path):
    from app.api import deps
    from app.api.registry import PaperRegistry

    monkeypatch.setattr(deps, "registry", PaperRegistry(tmp_path / "empty"))

    plan = client.post("/api/agent/plan", json={"task": "总结"})
    run = client.post("/api/agent/run", json={"task": "总结"})

    assert plan.status_code == 400
    assert run.status_code == 400
    assert "上传" in run.json()["detail"]
