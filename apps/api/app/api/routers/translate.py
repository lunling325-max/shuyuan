"""中译英路由：只做 HTTP 边界，逐段翻译与片段校验都在 app.rag.translate_zh。

为什么响应模型放在本文件而不是 schemas.py？
schemas.py 是全站共用契约，改它要牵动 main.py 的 import；翻译是新增模块，
自带局部模型后，挂载只需要 include 一个 router，不影响别的功能。

状态码的边界：
- 空输入 / 只有空白：400，调用方传错了东西；
- 超过 MAX_INPUT_CHARS：413，不是"服务端错误"，而是输入本身太大；
- 模型不可用：200 + degraded=true，断网时页面仍要能打开。
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.translate_en import MAX_INPUT_CHARS as MAX_EN_INPUT_CHARS
from app.rag.translate_en import translate_snippet
from app.rag.translate_zh import MAX_INPUT_CHARS, translate_text

router = APIRouter()


class TranslateRequest(BaseModel):
    text: str = Field(description="待翻译的中文摘要或正文段落")
    style: Literal["abstract", "section"] = Field(
        default="abstract", description="abstract=论文摘要；section=正文小节"
    )


class TranslateParagraphOut(BaseModel):
    zh: str = Field(description="原文中文段落，始终返回")
    en: str = Field(description="英文译文；降级或回退时可能为空或等于 zh")


class TranslateZhEnOut(BaseModel):
    paragraphs: list[TranslateParagraphOut]
    model: str
    degraded: bool = Field(
        description="True=模型不可用、或某段译文丢失公式/引用因而退回原文"
    )
    notes: list[str] = Field(default_factory=list, description="降级与回退的具体原因")


@router.post("/api/translate/zh-en", response_model=TranslateZhEnOut)
def translate_zh_en(req: TranslateRequest) -> TranslateZhEnOut:
    """中文摘要/正文 → 学术英文，逐段返回，保持原文段落顺序。"""
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "待翻译文本为空")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            413, f"文本超过 {MAX_INPUT_CHARS} 字上限，请分段后重试"
        )

    result = translate_text(text, style=req.style)
    return TranslateZhEnOut(
        paragraphs=[
            TranslateParagraphOut(zh=p.zh, en=p.en) for p in result.paragraphs
        ],
        model=result.model,
        degraded=result.degraded,
        notes=result.notes,
    )


class TranslateEnRequest(BaseModel):
    text: str = Field(description="阅读器里选中的那段英文（短语、句子或一段）")


class TranslateEnOut(BaseModel):
    en: str = Field(description="原文片段，始终返回")
    zh: str = Field(description="中文译文；降级时为空串")
    model: str
    degraded: bool = Field(
        description="True=模型不可用，或译文丢了公式/引用编号"
    )
    notes: list[str] = Field(default_factory=list, description="降级的具体原因")


@router.post("/api/translate/en-zh", response_model=TranslateEnOut)
def translate_en_zh(req: TranslateEnRequest) -> TranslateEnOut:
    """阅读器划词翻译：一小段英文 → 中文。

    上限比中译英小得多（1200 字符）：划词是"看这一句"，
    整段整篇的翻译走中英对照那条路，那条路有段落对齐与缓存。
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "待翻译文本为空")
    if len(text) > MAX_EN_INPUT_CHARS:
        raise HTTPException(
            413, f"选中文本超过 {MAX_EN_INPUT_CHARS} 字符，请分段后再试"
        )

    result = translate_snippet(text)
    return TranslateEnOut(
        en=result.en,
        zh=result.zh,
        model=result.model,
        degraded=result.degraded,
        notes=result.notes,
    )
