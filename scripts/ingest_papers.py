"""把 build/ingest-list.json 里的论文逐篇灌进线上库，记下每篇的耗时。

用服务端的 /api/tools/arxiv/ingest，本机不下载任何文件：PDF 由服务器自己拉，
既省得在本机落盘，也让"下载"发生在它本来就跑的链路里。
每篇之间留一点间隔，避免把 arXiv 惹毛（它限流时会返回 503）。

用法：
    python ingest_papers.py --limit 2          # 先试两篇
    python ingest_papers.py --limit 50         # 正式灌
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("EVAL_BASE_URL") or "http://127.0.0.1:8017"
ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "build" / "ingest-list.json"
LOG = ROOT / "build" / "ingest-log.json"


def ingest(arxiv_id: str) -> tuple[bool, str, int]:
    body = json.dumps({"arxiv_id": arxiv_id}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/tools/arxiv/ingest",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:160]
        return False, f"HTTP {exc.code} {detail}", int((time.time() - started) * 1000)
    except Exception as exc:  # noqa: BLE001
        return (
            False,
            f"{type(exc).__name__}: {exc}",
            int((time.time() - started) * 1000),
        )
    took = int((time.time() - started) * 1000)
    # 已入库的情况接口也会返回 200，只是带 duplicate 信息 —— 那不是失败，
    # 但也不该算进"新入一篇"的耗时统计里。
    if data.get("duplicate_of"):
        return True, f"库内已有 {data['duplicate_of']}", took
    return True, data.get("title", "")[:60], took


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--gap", type=float, default=2.0, help="两篇之间的间隔秒数")
    args = parser.parse_args()

    items = json.loads(LIST.read_text(encoding="utf-8"))[
        args.skip : args.skip + args.limit
    ]
    rows = []
    for i, it in enumerate(items, start=1):
        ok, note, ms = ingest(it["arxiv_id"])
        rows.append({"arxiv_id": it["arxiv_id"], "ok": ok, "note": note, "took_ms": ms})
        mark = "✓" if ok else "✗"
        print(
            f"[{args.skip + i:>2}] {mark} arXiv:{it['arxiv_id']} {ms:>6} ms  {note}",
            flush=True,
        )
        if i < len(items):
            time.sleep(args.gap)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    LOG.write_text(
        json.dumps(old + rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    good = [r for r in rows if r["ok"] and "库内已有" not in r["note"]]
    if good:
        times = sorted(r["took_ms"] for r in good)
        print(
            f"\n新入库 {len(good)} / {len(rows)}；单篇耗时 中位 {times[len(times) // 2]} ms、"
            f"最长 {times[-1]} ms"
        )
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
