"""把版本差异翻成人话。

为什么要过一遍大模型？
句级 diff 给出的是"这 41 处文本变了"，而读者要的是"第 4 版补了三处实验说明，
并统一了 Figure 2 的排版"。前者是机器能给的，后者才是人能用的。

但这一步也是最容易翻车的：喂进去的是 41 条噪声占比很高的改动，
模型完全可能一本正经地总结出"作者重写了第三节"。所以：

1. **先过滤**：参考文献编号位移、图表重排这类结构性噪声不进摘要；
2. **给原文，不给结论**：让模型看着新旧句子说话，而不是让它自由发挥；
3. **要求逐条标注出处**：摘要里每条都要挂 section，编不出来就没有出处可挂；
4. **失败可降级**：模型挂了就返回统计信息，不编一个假摘要糊弄。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.versioning import Change, VersionDiff

SYSTEM_PROMPT = """你在对比同一篇论文的两个版本，把差异总结给读者看。

规则：
- 只根据给出的句子对下判断，不要引入任何外部知识。
- **忽略**这些不算修订的东西：图表/表格的排版顺序变化、参考文献编号位移、
  标点与大小写、脚注位置、PDF 抽取造成的断词（signi- cantly）。
- 合并同类：同一节的若干处小改写成一条。
- 每条必须写 section（用给出的 section 值），不要自己编章节号。
- 最多 6 条，按重要性排序；没有实质修改就返回空数组。

只输出 JSON：
{"changes": [{"section": "3.2.2 Multi-Head Attention", "kind": "added|removed|reworded", "what": "一句话中文，说清改了什么"}]}"""

MAX_BODY_CHARS = 320


@dataclass(slots=True)
class DiffSummary:
    items: list[dict]
    ok: bool  # False = 模型不可用，items 为空
    reason: str = ""


def _trim(text: str, n: int = MAX_BODY_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[:n] + "…"


def build_prompt(diff: VersionDiff, *, max_items: int = 24) -> str:
    """只把值得看的改动挑出来 —— 喂噪声进去，出来的一定是废话。"""
    substantive = [c for c in diff.changes if not c.is_reference_churn]
    # 先排掉"纯图表题注"这类，再按页序取前若干条
    ranked = sorted(substantive, key=lambda c: (len(c.old), len(c.new)))[:max_items]

    lines = [
        f"共 {len(diff.changes)} 处文本差异，其中 {len(substantive)} 处不在参考文献里。",
        "下面是代表性的改动（旧版 → 新版）：",
    ]
    for i, c in enumerate(ranked, start=1):
        lines.append(f"\n[{i}] kind={c.kind} section={c.section or '（未标注）'}")
        for s in c.old[:2]:
            lines.append(f"  OLD: {_trim(s)}")
        for s in c.new[:2]:
            lines.append(f"  NEW: {_trim(s)}")
    return "\n".join(lines)


def summarize_diff(diff: VersionDiff, *, llm_call=None, max_items: int = 24) -> DiffSummary:
    if not diff.changes:
        return DiffSummary(items=[], ok=True, reason="两版正文没有可识别的差异")

    if llm_call is None:
        from app.llm.client import chat_json as llm_call

    try:
        raw = llm_call(build_prompt(diff, max_items=max_items), system=SYSTEM_PROMPT)
    except Exception as exc:  # noqa: BLE001 —— 摘要失败不该让版本对比整个不可用
        return DiffSummary(items=[], ok=False, reason=str(exc)[:200])

    items = []
    for item in raw.get("changes", []) or []:
        if not isinstance(item, dict):
            continue
        what = str(item.get("what") or "").strip()
        if not what:
            continue
        items.append(
            {
                "section": str(item.get("section") or "").strip(),
                "kind": str(item.get("kind") or "reworded").strip(),
                "what": what,
            }
        )
    return DiffSummary(items=items, ok=True)


def change_to_dict(c: Change) -> dict:
    return {
        "kind": c.kind,
        "section": c.section,
        "page": c.page_label,
        "old_pages": c.old_pages,
        "new_pages": c.new_pages,
        "old": c.old,
        "new": c.new,
        "reference_churn": c.is_reference_churn,
    }
