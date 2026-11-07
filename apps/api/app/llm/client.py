"""LLM 客户端（DeepSeek）。

刻意保持极简：只依赖标准库，方便在没装好依赖时先跑通链路。
正式接入流式响应时再换 httpx。
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.request
from collections.abc import Iterator
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


_DECODER = json.JSONDecoder()


def first_json_object(raw: str) -> dict[str, Any] | None:
    """从模型的原始输出里取**第一个完整的 JSON 对象**，取不到返回 None。

    为什么不直接 `json.loads(raw)`：实测 DeepSeek 会偶尔在一次回复里吐
    **两个**对象 —— 先给一个 `call`，紧跟一个措辞略不同的 `call`。
    `json.loads` 整串解析就报 `Expecting ',' delimiter`，任务 Agent 判定
    "决策失败"，退回预设计划，最后连结论都不给（E-023）。

    容忍三件事，都实测出现过：两个对象粘在一起、对象后面跟着解释文字、
    外面套着 ```json 围栏。取第一个就够了：模型想说的那件事总在最前面。
    """
    text = raw.strip()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _end = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def chat_json(
    user: str,
    *,
    system: str | None = None,
    temperature: float = 0.1,
    model: str | None = None,
) -> dict[str, Any]:
    """结构化输出，返回解析后的 JSON 对象。"""
    raw = chat(user, system=system, temperature=temperature, json_mode=True, model=model)
    parsed = first_json_object(raw)
    if parsed is None:
        raise LLMError(f"模型未返回合法 JSON: {raw[:300]}")
    return parsed


# ------------------------------------------------------------------ 流式
#
# 下面是**追加**能力：`chat` / `chat_json` 的签名与行为保持不变。
# 为什么不用 httpx：它只在 dev 依赖组里，生产镜像装不到；
# 现有链路又完全基于标准库 urllib，这里继续沿用可以保证零新增依赖。


class LLMStreamError(LLMError):
    """流式响应在传输途中断开或读超时。

    单独一个子类是为了让上层能区分两类失败：
    - 请求根本没发出去（缺 key / HTTP 4xx/5xx）—— LLMError；
    - 已经收到一部分，连接中途断了 —— LLMStreamError。
    两者都继承 LLMError，所以现有只捕获 LLMError 的调用点不受影响。
    """


def stream_chat(
    user: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    model: str | None = None,
) -> Iterator[str]:
    """流式单轮对话，逐 delta 产出 content 文本。

    协议是 DeepSeek 的 SSE：每一帧形如 `data: {json}`，最后以 `data: [DONE]` 结束。
    只 yield `choices[0].delta.content`，其余帧（role、finish_reason、心跳）一律跳过。

    注意这是一个**生成器**：函数体在第一次 `next()` 时才执行，所以
    "缺 key" 也是在开始迭代时抛 LLMError，而不是调用时。调用方若需要
    "开始流之前"就知道模型不可用，应自己先检查环境变量。
    """
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise LLMError("缺少环境变量 DEEPSEEK_API_KEY")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload: dict[str, Any] = {
        "model": model or os.environ.get("LLM_MODEL", "deepseek-chat"),
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )

    # urllib 只有一个 timeout，它同时约束连接与每次读。60 秒足够覆盖
    # 两个 token 之间正常的最大间隔；关键是**必须有上限**，否则半死连接
    # 会让生成器永远挂在 read 上，SSE 连接和线程都不释放。
    try:
        timeout = float(os.environ.get("LLM_STREAM_TIMEOUT", "60"))
    except ValueError:  # 配错了也要能跑，别让一个环境变量把流式接口变成 500
        timeout = 60.0

    resp = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                frame = json.loads(data)
            except json.JSONDecodeError:
                # SSE 偶尔会有心跳或半截帧。跳过单帧，不要把整条流打断。
                continue
            try:
                delta = frame["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, AttributeError, TypeError):
                # role 帧、finish_reason 帧没有 content，属于正常情况。
                continue
            if delta:
                yield delta
    except urllib.error.HTTPError as exc:  # noqa: UP041
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
        # 走到这里说明请求已经发出、正在读流；典型是连接被重置或 read 超时。
        raise LLMStreamError(f"流式响应中断: {exc}") from exc
    finally:
        # 单独兜住 close 的异常：否则它会在 GeneratorExit 传播途中
        # 被上面的 except OSError 捕获，把干净退出变成 "generator ignored GeneratorExit"。
        if resp is not None:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
