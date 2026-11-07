"""库变大之后跑一遍全站视图，量延迟与载荷大小。

P1 要回答的问题是："7 篇的时候一切都好，30+ 篇的时候呢？"
所以这里不测单个接口对不对，只测**全站十个视图各自还能不能在可接受的时间里返回**，
并记下每个响应的字节数 —— 库一大，返回体跟着变胖，前端的渲染开销也就跟着变。

用法（项目根目录）：
    uv run --directory apps/api python ../../build/measure_scale.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("EVAL_BASE_URL") or "http://127.0.0.1:8017"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "scale.json"


def call(method: str, path: str, payload: dict | None = None) -> tuple[int, bytes, int]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    return _send(req)


def call_multipart(path: str, fields: dict[str, str], filename: str, content: bytes):
    """数据可视化走的是 multipart 上传（用户传一份 CSV），不能当 JSON 发。"""
    boundary = "----scale" + Path(filename).stem
    parts = []
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\nContent-Type: text/csv\r\n\r\n'
    )
    head = "".join(parts).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=head + content + tail,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return _send(req)


def _send(req: urllib.request.Request) -> tuple[int, bytes, int]:
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
    return status, body, int((time.time() - started) * 1000)


def main() -> None:
    papers = json.loads(call("GET", "/api/papers")[1])
    print(f"库内 {len(papers)} 篇")
    # 挑一篇正文最长的当对照：页数越多，检索与切块的压力越真实
    longest = max(papers, key=lambda p: p.get("pages") or 0)
    print(f"对照论文：{longest['title'][:50]}（{longest['pages']} 页）\n")

    probes: list[tuple[str, str, str, dict | None]] = [
        ("文献库列表", "GET", "/api/papers", None),
        ("单篇详情", "GET", f"/api/papers/{longest['id']}", None),
        ("切块列表", "GET", f"/api/papers/{longest['id']}/chunks", None),
        ("参考文献", "GET", f"/api/papers/{longest['id']}/references", None),
        ("引用网络", "GET", "/api/graph", None),
        ("每日推荐", "GET", "/api/recommend", None),
        (
            "单篇问答",
            "POST",
            f"/api/papers/{longest['id']}/qa",
            {"question": "这篇论文要解决什么问题？", "mode": "answer"},
        ),
        (
            "综述",
            "POST",
            "/api/review",
            {"topic": "高效注意力与长上下文建模", "max_papers": 6},
        ),
        (
            "Copilot 提纲",
            "POST",
            "/api/copilot/outline",
            {"idea": "长上下文高效训练", "max_points": 4},
        ),
        (
            "Agent 规划",
            "POST",
            "/api/agent/plan",
            {"task": "梳理库里关于稀疏注意力的工作"},
        ),
    ]

    rows = []
    for name, method, path, payload in probes:
        status, body, ms = call(method, path, payload)
        note = ""
        if status >= 400:
            try:
                note = json.loads(body).get("detail", "")[:80]
            except Exception:  # noqa: BLE001
                note = body[:80].decode("utf-8", "replace")
        rows.append(
            {"name": name, "status": status, "ms": ms, "bytes": len(body), "note": note}
        )
        mark = "✓" if status < 400 else "✗"
        print(
            f"{mark} {name:<10} {ms:>6} ms  {len(body) / 1024:>8.1f} KB  {status}  {note}"
        )

    # 数据可视化：库里没有 CSV，自带一份小表 —— 它测的是"生成图表"这条模型链路，
    # 与库大小关系不大，但一站一站跑全了才算真的"十个视图都过了一遍"
    csv = "序号,吞吐提升,显存占用\n1,1.0,100\n2,1.8,86\n3,2.4,72\n4,3.5,61\n".encode()
    status, body, ms = call_multipart(
        "/api/dataviz/chart",
        {"kind": "bar", "x": "序号", "y": "吞吐提升"},
        "scale.csv",
        csv,
    )
    note = ""
    if status >= 400:
        try:
            note = json.loads(body).get("detail", "")[:80]
        except Exception:  # noqa: BLE001
            note = body[:80].decode("utf-8", "replace")
    rows.append(
        {
            "name": "数据可视化",
            "status": status,
            "ms": ms,
            "bytes": len(body),
            "note": note,
        }
    )
    print(
        f"{'✓' if status < 400 else '✗'} {'数据可视化':<10} {ms:>6} ms  {len(body) / 1024:>8.1f} KB  {status}  {note}"
    )

    ok = [r for r in rows if r["status"] < 400]
    if ok:
        times = sorted(r["ms"] for r in ok)
        print(
            f"\n可用 {len(ok)} / {len(rows)}；延迟 中位 {int(statistics.median(times))} ms、"
            f"最长 {times[-1]} ms；总载荷 {sum(r['bytes'] for r in rows) / 1024:.0f} KB"
        )
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
