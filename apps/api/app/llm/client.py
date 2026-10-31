"""LLM 客户端（DeepSeek）。

刻意保持极简：只依赖标准库，方便在没装好依赖时先跑通链路。
正式接入流式响应时再换 httpx。
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


def _post(payload: dict[str, Any], *, timeout: int = 180) -> dict[str, Any]:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise LLMError("缺少环境变量 DEEPSEEK_API_KEY")

    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # noqa: UP041
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"HTTP {exc.code}: {detail}") from exc


def chat(
    user: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    model: str | None = None,
) -> str:
    """单轮对话，返回文本内容。"""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload: dict[str, Any] = {
        "model": model or os.environ.get("LLM_MODEL", "deepseek-chat"),
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = _post(payload)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"响应结构异常: {json.dumps(data, ensure_ascii=False)[:300]}") from exc


def chat_json(
    user: str,
    *,
    system: str | None = None,
    temperature: float = 0.1,
    model: str | None = None,
) -> dict[str, Any]:
    """结构化输出，返回解析后的 JSON 对象。"""
    raw = chat(user, system=system, temperature=temperature, json_mode=True, model=model)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"模型未返回合法 JSON: {raw[:300]}") from exc
