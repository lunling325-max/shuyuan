"""英译中：给"划词翻译"用的一小段英文（一个短语、一句话、最多一段）。

和中译英（`translate_zh.py`）共享同一条铁律，但场景不同：

- 中译英是"把我要投出去的稿子翻成英文"，逐段、可整篇、要学术写作腔；
- 英译中是把原文里**选中的那一截**就地解释给人看：几十个词，随取随用，
  而且要贴着原文语境（在讲 Multi-Head Attention 的那一段里，
  "head" 就是"注意力头"，不是"头"）。

不可翻译片段守恒这条照旧：原文里的 LaTeX、行内代码、`[12]` 这样的引用编号
必须在译文里原样出现。丢一个公式，读者就没法回原文核对——划词翻译的全部意义
就在于"看完还能回去"。

模型不可用时返回 degraded，不抛 500：界面上给出"这次没翻成"，原文仍在用户手里。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.rag.translate_zh import missing_fragments

# 划词是"看这一句"。1200 字符已经是一整段长文，再长就该走中英对照那条路了。
MAX_INPUT_CHARS = 1200

DEFAULT_MODEL = "deepseek-chat"

_SYSTEM_PROMPT = """你是学术论文的精读助手。把用户选中的英文片段翻成中文。

铁律：
1. 只输出 JSON：{"zh": "译文"}，不要解释，不要 Markdown 代码块。
2. 学术语体，术语用领域通用译法（attention head → 注意力头，fine-tuning → 微调）。
3. 以下内容原样保留，一个字符都不能改：
   - LaTeX 公式（$...$、$$...$$、\\frac{...}{...}）
   - 行内代码（反引号包裹的内容）
   - 方括号引用编号（如 [1]、[2,3]、[1-3]）
   - 数字与单位、模型与数据集缩写（BERT、GPT-3、BLEU、ImageNet）
4. 不补充原文没有的内容。原文是半句话，就译半句话，不要替作者把话说完。"""


@dataclass(slots=True)
class Snippet:
    """一次划词的结果。`en` 永远是原文，模型挂了也不会丢。"""

    en: str
    zh: str
    degraded: bool = False
    model: str = DEFAULT_MODEL
    notes: list[str] = field(default_factory=list)


def _prompt(text: str) -> str:
    return f'请把下面这段英文翻成中文，只返回 {{"zh": "..."}}：\n\n{text}'


def _chinese(raw: object) -> str:
    """从模型输出里取中文译文；结构不符返回空串，由调用方标降级。"""
    if isinstance(raw, dict):
        value = raw.get("zh")
        return value.strip() if isinstance(value, str) else ""
    # 模型偶尔直接返回裸字符串：能读就当译文，不因此整段失败。
    if isinstance(raw, str):
        return raw.strip()
    return ""


def translate_snippet(text: str, *, llm_call=None) -> Snippet:
    """翻一小段英文。任何模型异常都折叠成降级结果，绝不向上抛。

    `llm_call` 可注入，测试里替换掉就不会真的发网络请求。
    """
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    source = (text or "").strip()
    if not source:
        return Snippet(en="", zh="", model=model)

    if llm_call is None:
        from app.llm.client import chat_json as llm_call  # 延迟导入，测试不需要网络

    try:
        raw = llm_call(_prompt(source), system=_SYSTEM_PROMPT, temperature=0.0)
    except Exception as exc:  # noqa: BLE001 —— 翻译是增强项，不能把请求打成 500
        return Snippet(
            en=source,
            zh="",
            degraded=True,
            model=model,
            notes=[f"模型不可用：{exc.__class__.__name__}: {exc}"],
        )

    zh = _chinese(raw)
    if not zh:
        return Snippet(
            en=source,
            zh="",
            degraded=True,
            model=model,
            notes=["模型未返回有效译文"],
        )

    # 公式/引用编号被吃掉：译文仍然给出来（比什么都没有强），但要标明。
    missing = missing_fragments(source, zh)
    if missing:
        lost = "、".join(f if len(f) <= 60 else f[:60] + "..." for f in missing)
        return Snippet(
            en=source,
            zh=zh,
            degraded=True,
            model=model,
            notes=[f"译文丢失原文片段（{lost}），请对照原文确认"],
        )

    return Snippet(en=source, zh=zh, model=model)
