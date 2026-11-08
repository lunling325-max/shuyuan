"""量「正文按编号提到图表」的覆盖率 —— 图文互链到底连上了多少张卡片。

判据和前端 `buildMentions()` 完全同一套（apps/web/index.html）：

1. 卡片以**图注里认出来的编号**为准（`Table 2.1` → `table:2.1`），
   正文里出现的编号如果没有对应卡片，不算；
2. **图注自己那一块不算**：图注里必然写着它自己的编号，那是定义不是引用；
   判据是"这一块文本包含该卡片图注的前 40 字"；
3. 同一块里提了三次也只算一处。

用法:
    python scripts/measure_crosslinks.py                    # 量线上
    python scripts/measure_crosslinks.py --base http://127.0.0.1:8017
    python scripts/measure_crosslinks.py --verbose          # 逐张卡片打出来
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

# 与前端 FIG_RE 一致：`Tab.` 必须带点（不带点会撞 "tab" 的其它含义）
FIG_RE = re.compile(r"(figure|fig\.?|table|tab\.|图|表)\s*(\d+(?:\.\d+)*)", re.I)
# 补 `Tab.` 之前的写法，用来看这一刀到底多连上了多少张（E-031 的对照）
FIG_RE_OLD = re.compile(r"(figure|fig\.?|table|图|表)\s*(\d+(?:\.\d+)*)", re.I)
LABEL_RE = re.compile(r"(\d+(?:\.\d+)*)\s*[a-z]?$", re.I)
TABLE_WORDS = {"table", "tab.", "表"}


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def fig_kind(word: str) -> str:
    return "table" if word.lower() in TABLE_WORDS else "figure"


def key_of_label(kind: str, label: str) -> str:
    m = LABEL_RE.search(label or "")
    return f"{kind}:{m.group(1).lower()}" if m else ""


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=180) as r:
        return json.load(r)


def measure_paper(base: str, paper: dict, verbose: bool, pattern: re.Pattern[str]) -> dict:
    pid = paper["id"]
    chunks = get(base, f"/api/papers/{pid}/chunks")
    tables = get(base, f"/api/papers/{pid}/tables").get("tables", [])
    figures = get(base, f"/api/papers/{pid}/figures").get("figures", [])

    captions: dict[str, str] = {}
    cards: dict[str, dict] = {}
    for kind, items in (("table", tables), ("figure", figures)):
        for item in items:
            key = key_of_label(kind, item.get("label", ""))
            if not key:
                continue
            cards[key] = {"kind": kind, "label": item.get("label"), "page": item.get("page")}
            cap = squash(item.get("caption"))[:40]
            if cap:
                captions[key] = cap

    mentions: dict[str, set[int]] = {key: set() for key in cards}
    for chunk in chunks:
        flat = squash(chunk.get("text"))
        seen: set[str] = set()
        for m in pattern.finditer(flat):
            key = f"{fig_kind(m.group(1))}:{m.group(2).lower()}"
            if key in seen or key not in cards:
                continue
            cap = captions.get(key)
            if cap and cap in flat:
                continue                      # 这一块就是它自己的图注
            seen.add(key)
            mentions[key].add(chunk.get("page_start"))

    hit = [key for key in cards if mentions[key]]
    if verbose:
        for key, card in sorted(cards.items()):
            pages = sorted(mentions[key])
            where = "正文 " + " ".join(f"p.{p}" for p in pages) if pages else "仅图注"
            print(f"    {card['label']:<14} {where}")
    return {
        "file": paper.get("filename"),
        "tables": len(tables),
        "figures": len(figures),
        "cards": len(cards),
        "hit": len(hit),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # 控制台是 GBK，正文里有 ℓ 之类的字符
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://101.37.190.214:8017")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--regex", choices=("new", "old"), default="new",
                    help="old = 补 `Tab.` 之前的写法（对照用）")
    a = ap.parse_args()

    pattern = FIG_RE if a.regex == "new" else FIG_RE_OLD
    papers = get(a.base, "/api/papers")
    print(f"{'文件':<24}{'表':>4}{'图':>4}{'编号':>6}{'有正文引用':>10}")
    total_cards = total_hit = 0
    for paper in papers:
        try:
            row = measure_paper(a.base, paper, a.verbose, pattern)
        except Exception as exc:  # noqa: BLE001 —— 单篇失败不该拖垮整轮测量
            print(f"{paper.get('filename', paper['id']):<24}  失败：{exc}")
            continue
        total_cards += row["cards"]
        total_hit += row["hit"]
        print(f"{str(row['file'])[:23]:<24}{row['tables']:>4}{row['figures']:>4}"
              f"{row['cards']:>6}{row['hit']:>10}")
    pct = (100.0 * total_hit / total_cards) if total_cards else 0.0
    print(f"\n合计：卡片 {total_cards} 张，正文能跳过去 {total_hit} 张（{pct:.1f}%）"
          f"　［按编号去重：同一编号多张卡片只算一次］")
    return 0


if __name__ == "__main__":
    sys.exit(main())
