"""从 app 自己的 arXiv 检索结果里挑同主题论文，攒成一份入库清单。

为什么不用手写 arXiv 号：凭记忆写号必然出现"号对文错"，而且下一轮谁也没法复现
这份清单是怎么来的。全部走线上检索接口，清单落盘，可复查。
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

BASE = os.environ.get("EVAL_BASE_URL") or "http://127.0.0.1:8017"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "ingest-list.json"

QUERIES = [
    "long context attention efficient transformer",
    "sparse attention kernel GPU",
    "linear attention language model",
    "flash attention IO aware exact",
    "distributed training sequence parallel",
    "KV cache compression inference",
    "state space model language modeling",
    "mixture of experts efficient training",
    "long context language model benchmark",
]

# 标题里必须命中其中一个才算同主题。检索接口按相关度排序，会把
# "格点计数""项目调度""医学影像"一起捞上来，不筛一遍库里就成了杂货铺。
WANT = (
    "attention",
    "transformer",
    "kv cache",
    "long-context",
    "long context",
    "mixture of experts",
    "moe",
    "sequence parallel",
    "mamba",
    "state space",
    "flash",
    "context length",
)


def search(q: str, limit: int = 30) -> list[dict]:
    url = f"{BASE}/api/tools/arxiv/search?q={urllib.parse.quote(q)}&limit={limit}"
    body = json.loads(urllib.request.urlopen(url, timeout=120).read())
    items = body.get("results") if isinstance(body, dict) else body
    return items or []


def main() -> None:
    have = {
        p.get("arxiv_id")
        for p in json.loads(
            urllib.request.urlopen(f"{BASE}/api/papers", timeout=60).read()
        )
    }
    picked: dict[str, dict] = {}
    for q in QUERIES:
        try:
            items = search(q)
        except Exception as exc:  # noqa: BLE001 —— 单个检索式失败不该中断挑题
            print(f"[跳过] {q}：{type(exc).__name__}")
            continue
        kept = 0
        for it in items:
            aid = (it.get("arxiv_id") or it.get("id") or "").replace(
                "http://arxiv.org/abs/", ""
            )
            title = (it.get("title") or "").strip()
            if not aid or not title or aid in have or aid in picked:
                continue
            if not any(w in title.lower() for w in WANT):
                continue
            picked[aid] = {"arxiv_id": aid, "title": title, "via": q}
            kept += 1
        print(f"[{q}] 收 {kept} 条")

    OUT.write_text(
        json.dumps(list(picked.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n去重后同主题候选 {len(picked)} 篇，已写入 {OUT}")


if __name__ == "__main__":
    main()
