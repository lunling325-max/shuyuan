"""中译英：把中文摘要或整段正文逐段翻成学术英语。

为什么是"逐段"而不是"整篇一次"或"按字数切"？
1. 段落是论文里天然的语义单位。按空行/换行切，段落顺序就是原文顺序，
   前端可以稳定地做左中右对照；按字数硬切会把公式、引用标记和句子拦腰截断。
2. 一次请求翻一整篇长文，模型容易合并段落、漏段，还更容易在长上下文里
   丢掉公式。逐段翻虽然多几次调用，但每段都可独立校验、独立回退。
3. DeepSeek 端有限速，并发发请求只会换来 429；顺序翻译虽然慢一点，
   但行为确定、失败点清晰，演示时不会出现"一半段落乱序"。

最强的约束是**不可翻译片段守恒**：原文里的 LaTeX（`$...$` / `\frac{}{}`）、
行内代码、方括号引用编号（`[1]`）必须在译文中原样出现。学术翻译里丢一个
公式或把 `[1]` 吃掉，读者就无法回查依据，这比"译得不够漂亮"严重得多。
因此服务端逐段校验，缺片段就把该段退回原文并记录原因，绝不假装翻好了。

模型不可用必须是 200 + degraded，而不是 500：断网时页面还得能打开，
用户至少能看到原文和"这次没翻成"的说明。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# 20000 字：一篇中文摘要通常 300~800 字，整章正文也很少超过这个量。
# 再长就不是"一键翻译"的场景了，先拒绝比让模型截断或超时更诚实。
MAX_INPUT_CHARS = 20000

DEFAULT_MODEL = "deepseek-chat"

# 空行是首选边界；如果整段文本只有单换行（常见于从 PDF 复制出来的摘要），
# 再退到按换行切。两种切法都不按字符数硬切。
_PARA_SPLIT = re.compile(r"\n\s*\n")
_LINE_SPLIT = re.compile(r"\n+")

# LaTeX 片段：行内/独立公式，以及 \frac{}{}（可能出现在公式外）。
_LATEX_MATH = re.compile(r"\${1,2}[^$]+?\${1,2}", re.DOTALL)
_LATEX_FRAC = re.compile(r"\\frac\s*\{[^{}]*\}\s*\{[^{}]*\}")

# 行内代码与方括号引用编号。引用支持 [1]、[1,2]、[1-3] 这几种常见写法。
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_CITATION = re.compile(r"\[\s*\d+(?:\s*[,，\-–]\s*\d+)*\s*\]")

_STYLE_HINTS = {
    "abstract": (
        "这是论文摘要。使用学术摘要体：主动语态、陈述研究做了什么，"
        "例如 'We propose ...'、'This paper presents ...'，"
        "不要用 'I will ...' 这种口语化表达，也不要营销腔。"
    ),
    "section": (
        "这是论文正文的一节。保持章节体与术语前后一致，"
        "使用学术写作的客观语气，不要口语化，也不要营销腔。"
    ),
}

_SYSTEM_PROMPT = """你是严谨的学术论文翻译助手。把用户给出的中文段落翻译成英文。

铁律：
1. 只输出 JSON：{"en": "本段英文译文"}，不要输出数组、解释或 Markdown 代码块。
2. 学术语体：使用论文英文摘要/正文的写法（如 "We propose ..."），
   避免 "I will ..."、口语化表达与营销腔。
3. 以下内容必须原样保留，一个字符都不能改：
   - LaTeX 公式（$...$、$$...$$、\\frac{...}{...}）
   - 行内代码（反引号包裹的内容）
   - 方括号引用编号（如 [1]、[2,3]、[1-3]）
   - 数字与单位、专有名词缩写（如 BERT、GPT、LSTM）
4. 不要添加原文没有的内容，不要解释，不要改写引用编号的格式。"""


@dataclass(slots=True)
class Paragraph:
    zh: str
    en: str


@dataclass(slots=True)
class Translation:
    paragraphs: list[Paragraph]
    degraded: bool
    model: str
    notes: list[str] = field(default_factory=list)


def split_paragraphs(text: str) -> list[str]:
    """按空行切段；只有单换行时再按换行切。纯空白段丢弃。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    pieces = [p.strip() for p in _PARA_SPLIT.split(normalized) if p.strip()]
    if len(pieces) <= 1:
        pieces = [p.strip() for p in _LINE_SPLIT.split(normalized) if p.strip()]
    return pieces


def _compact(text: str) -> str:
    """去掉空白后再比较：模型把 `[1, 2]` 写成 `[1,2]` 不应算丢失。"""
    return re.sub(r"\s+", "", text)


def required_fragments(text: str) -> list[str]:
    """列出必须在译文中原样出现的片段，按规范去重并保持出现顺序。"""
    found: list[str] = []
    for pattern in (_LATEX_MATH, _LATEX_FRAC, _INLINE_CODE, _CITATION):
        found.extend(match.group(0) for match in pattern.finditer(text))

    unique: list[str] = []
    seen: set[str] = set()
    for fragment in found:
        key = _compact(fragment)
        if key and key not in seen:
            seen.add(key)
            unique.append(fragment)
    return unique


def missing_fragments(source: str, translation: str) -> list[str]:
    """返回译文里丢失的不可翻译片段；空列表表示全部保留。"""
    compact_translation = _compact(translation)
    return [
        fragment
        for fragment in required_fragments(source)
        if _compact(fragment) not in compact_translation
    ]


def _short(fragment: str, limit: int = 60) -> str:
    return fragment if len(fragment) <= limit else fragment[:limit] + "..."


def _system_prompt(style: str) -> str:
    return f"{_SYSTEM_PROMPT}\n\n本次文体：{_STYLE_HINTS.get(style, _STYLE_HINTS['abstract'])}"


def _prompt(paragraph: str, style: str) -> str:
    return (
        f"文体：{style}\n"
        "请把下面这一段中文翻译成学术英文，只返回 {\"en\": \"...\"}：\n\n"
        f"{paragraph}"
    )


def _english(raw: object) -> str:
    """从模型输出里取英文译文；结构不符返回空串，由调用方回退。"""
    if isinstance(raw, dict):
        value = raw.get("en")
        return value.strip() if isinstance(value, str) else ""
    # 模型偶尔直接返回裸字符串，按译文处理，不因此整段失败。
    if isinstance(raw, str):
        return raw.strip()
    return ""


def translate_text(
    text: str,
    *,
    style: str = "abstract",
    llm_call=None,
) -> Translation:
    """逐段翻译中文文本；任何模型异常都折叠成降级结果，绝不向上抛 500。

    `llm_call` 可注入，测试里替换掉就不会真的发网络请求。
    """
    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return Translation(paragraphs=[], degraded=False, model=model, notes=[])

    if llm_call is None:
        from app.llm.client import chat_json as llm_call  # 延迟导入，测试不需要网络

    translated: list[Paragraph] = []
    notes: list[str] = []
    degraded = False

    for index, zh in enumerate(paragraphs, start=1):
        try:
            raw = llm_call(
                _prompt(zh, style),
                system=_system_prompt(style),
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 —— 翻译是增强项，不能把请求打成 500
            # 模型不可用时不再继续试探：剩下的段落也只会一样失败，
            # 而且限速期间连续重试会拖长响应。整批留空，原文仍在 zh 字段里。
            return Translation(
                paragraphs=[Paragraph(zh=p, en="") for p in paragraphs],
                degraded=True,
                model=model,
                notes=[f"模型不可用，译文留空：{exc.__class__.__name__}: {exc}"],
            )

        en = _english(raw)
        if not en:
            degraded = True
            notes.append(f"第 {index} 段模型未返回有效译文，已退回原文")
            translated.append(Paragraph(zh=zh, en=zh))
            continue

        missing = missing_fragments(zh, en)
        if missing:
            degraded = True
            lost = "、".join(_short(f) for f in missing)
            notes.append(f"第 {index} 段译文丢失原文片段（{lost}），已退回原文")
            translated.append(Paragraph(zh=zh, en=zh))
            continue

        translated.append(Paragraph(zh=zh, en=en))

    return Translation(
        paragraphs=translated,
        degraded=degraded,
        model=model,
        notes=notes,
    )
