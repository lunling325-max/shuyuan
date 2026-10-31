"""中英对照：把证据块的英文正文逐段翻成中文。

为什么不做整篇翻译，也不做"一句话一句话"的流式翻译？
1. **段落是天然的对照单位**。前端要把 `en` / `zh` 并排显示，切块器已经用空行
   保留了自然段，照这个边界翻，段落数就是可校验的契约。
2. **段数必须守恒**。模型很爱把两段合成一段、或者把公式单独拆一行。
   一旦段数变了，前端并排显示就会错位，所以这里把"数量一致"当成硬校验：
   对不上就降级，绝不硬凑。
3. **降级必须是 200**。断网时演示还得继续 —— 原文照常返回，译文留空，
   `degraded=true` 让前端知道"不是没内容，是这次没翻成"。

缓存只存"确实调过模型"的结果：断网那一次不缓存，网络恢复后重试还能成功；
真正花过钱的成功响应则按 (paper_id, chunk_index) 留在进程内，重复请求不再扣费。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from app.parsing.chunker import Chunk

SYSTEM_PROMPT = """你是严谨的学术论文翻译助手。把用户给出的英文段落逐段翻译成中文。

铁律：
1. 只输出 JSON：{"zh": ["第一段译文", "第二段译文", ...]}
2. zh 数组长度必须与输入段落数**完全一致**：不许合并、不许拆分、不许跳过任何一段。
3. 公式、代码、变量名（如 softmax(QK^T/√d_k)、d_model、multi-head）保留原文，不翻译。
4. 不要添加解释、注释或原文没有的内容；术语按领域惯例翻译。"""

# 切块器用两个换行连接自然段，所以空行就是段落边界。
_PARA_SPLIT = re.compile(r"\n\s*\n")

DEFAULT_MODEL = "deepseek-chat"


@dataclass(slots=True)
class Paragraph:
    en: str
    zh: str


@dataclass(slots=True)
class Translation:
    paragraphs: list[Paragraph]
    degraded: bool
    model: str


# 一个模块级 dict 就够：进程内命中即可，没有跨进程共享的需求。
_CACHE: dict[tuple[str, int], Translation] = {}


def clear_cache() -> None:
    """测试用：不同用例的桩译文不能互相串味。"""
    _CACHE.clear()


def split_paragraphs(text: str) -> list[str]:
    """按空行切段，并丢掉纯空白段。"""
    return [piece.strip() for piece in _PARA_SPLIT.split(text) if piece.strip()]


def _prompt(paragraphs: list[str]) -> str:
    body = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs, start=1))
    return (
        f"共 {len(paragraphs)} 段。请逐段翻译，输出 {len(paragraphs)} 条译文，"
        f"顺序与编号一致：\n\n{body}"
    )


def _translations(raw: object) -> list[str]:
    """从模型输出里取出译文列表；结构不对就当一条都没拿到。

    这里**不截断**：多返回几条也要让调用方看见，好把 whole 结果标记为降级。
    """
    if not isinstance(raw, dict):
        return []
    zh = raw.get("zh")
    if isinstance(zh, str):  # 模型有时只返回一个字符串，包成单元素
        zh = [zh]
    if not isinstance(zh, list):
        return []
    return ["" if item is None else str(item).strip() for item in zh]


def translate_chunk(
    chunk: Chunk,
    *,
    paper_id: str,
    llm_call=None,
) -> Translation:
    """翻译一个块；模型失败时返回 200 可用的降级结果，而不是抛异常。

    `llm_call` 可注入，测试里替换掉就不会真的发网络请求。
    """
    key = (paper_id, chunk.index)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    paragraphs = split_paragraphs(chunk.text)
    if not paragraphs:
        return Translation(paragraphs=[], degraded=False, model=model)

    if llm_call is None:
        from app.llm.client import chat_json as llm_call  # 延迟导入，测试不需要网络

    try:
        raw = llm_call(
            _prompt(paragraphs),
            system=SYSTEM_PROMPT,
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001 —— 翻译是增强项，不能把请求打成 500
        return Translation(
            paragraphs=[Paragraph(en=p, zh="") for p in paragraphs],
            degraded=True,
            model=model,
        )

    # 段数对不上：只按 min 长度配对，多出来的原文留空译文，不编造。
    translations = _translations(raw)
    degraded = len(translations) != len(paragraphs)
    result = Translation(
        paragraphs=[
            Paragraph(en=en, zh=translations[i] if i < len(translations) else "")
            for i, en in enumerate(paragraphs)
        ],
        degraded=degraded,
        model=model,
    )
    # 这次确实调过模型（花过钱），即使段数不符也缓存，重复请求不再扣费；
    # 上面的失败分支不缓存，网络恢复后还能重试成功。
    _CACHE[key] = result
    return result
