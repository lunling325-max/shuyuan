"""模型输出的解析。

这组用例守的是**容忍度**：模型不是程序，它的输出会多一个对象、多一段解释、
多一层围栏。每一次都当成"决策失败"退回降级，用户看到的就是"Agent 干完了活
却没有结论"—— 明明工具都调成功、观测都在，就差最后一段话（E-023）。
"""

from __future__ import annotations

from app.llm.client import first_json_object


def test_two_objects_in_one_reply_take_the_first():
    """实测 DeepSeek 会连着吐两个 call，整串解析直接报错。"""
    raw = (
        '{"action": "call", "tool": "compare_versions", "args": {"paper_a": "a"}}'
        ' {"action": "call", "tool": "compare_versions", "args": {"paper_a": "b"}}'
    )
    obj = first_json_object(raw)
    assert obj is not None
    assert obj["args"]["paper_a"] == "a", "第二个对象是模型的改口，但第一个才是它的判断"


def test_trailing_prose_is_ignored():
    raw = '{"claims": []}\n\n上面这条是结论：证据不足。'
    assert first_json_object(raw) == {"claims": []}


def test_a_fenced_code_block_still_parses():
    raw = '```json\n{"sections": [{"heading": "一"}]}\n```'
    assert first_json_object(raw) == {"sections": [{"heading": "一"}]}


def test_braces_inside_strings_do_not_confuse_the_scanner():
    """先遇到的那个 `{` 可能只是一句解释里的花括号，要往后找真正的对象。"""
    raw = '注意 {这不是 JSON} 下面是结果：\n{"ok": true}'
    assert first_json_object(raw) == {"ok": True}


def test_no_json_at_all_returns_none():
    assert first_json_object("抱歉，我无法完成这个请求。") is None
    assert first_json_object('{"broken": ') is None
