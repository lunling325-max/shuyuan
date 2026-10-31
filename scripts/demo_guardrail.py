"""端到端演示：LLM 生成版本差异结论 → 引用校验守卫逐条验真。

判据很简单：模型说的每条结论都必须附一段"原文逐字片段"，
守卫去原文里找。找不到 = 幻觉，直接毙掉。

用法: python scripts/demo_guardrail.py
前置: .env 中有 DEEPSEEK_API_KEY
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.guardrails import Claim, Quote, summarize, verify_claims  # noqa: E402
from app.llm import chat_json  # noqa: E402
from app.parsing import TextIndex  # noqa: E402

DIFF_FILE = ROOT / ".cache" / "v1_v7_refined.json"
PDFS = {"v1": ROOT / "pdf" / "1706.03762v1.pdf", "v7": ROOT / "pdf" / "1706.03762v7.pdf"}

SYSTEM = """你是学术文献版本比对助手。
硬性要求：每一条结论都必须附带至少一条"原文引用"。
引用必须是该版本 PDF 正文中**逐字连续出现**的片段（建议 8-20 个词），
不得改写、不得拼接、不得跨段合并、不得凭印象杜撰措辞。
如果某条结论你找不到逐字依据，就不要输出这条结论。"""


def build_digest(n: int = 40) -> str:
    refined = json.loads(DIFF_FILE.read_text(encoding="utf-8"))
    lines = []
    for r in refined[:n]:
        # 不截断：喂给模型的证据必须完整，否则它会把截断原样抄进引用
        lines.append(f"[p{r['page']}][{r['kind']}] 旧: {r['old']} ||| 新: {r['new']}")
    return "\n".join(lines)


def main() -> None:
    indexes = {name: TextIndex(str(p)) for name, p in PDFS.items()}
    print(f"索引就绪: " + ", ".join(f"{k}={len(v)} tokens" for k, v in indexes.items()))

    digest = build_digest()
    prompt = f"""下面是一篇论文 arXiv v1 与 v7 的逐条差异（[页码][增/删/改]，旧=仅在v1，新=仅在v7）。

请归纳出 4-6 条实质性修订，每条给出结论 + 支撑它的原文引用。

输出 JSON，结构严格如下：
{{"claims": [{{"claim": "中文结论", "quotes": [{{"doc": "v1或v7", "text": "原文逐字片段"}}]}}]}}

注意：quotes.text 必须是原文里真实存在的一段连续文字，你只能从下面差异中看到的内容去回溯，
不确定就不要写。宁可少写几条，也不要编。

差异：
{digest}"""

    print("\n调用模型生成结论...")
    data = chat_json(prompt, system=SYSTEM)
    (ROOT / ".cache" / "guardrail_raw.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raw_claims = data.get("claims", [])
    print(f"模型给出 {len(raw_claims)} 条结论\n")

    claims = [
        Claim(
            text=c.get("claim", ""),
            quotes=[Quote(doc=q.get("doc", ""), text=q.get("text", "")) for q in c.get("quotes", [])],
        )
        for c in raw_claims
    ]

    results = verify_claims(claims, indexes)

    print("=" * 78)
    print(f"{'判定':<12}{'页码':<8}结论")
    print("=" * 78)
    for r in results:
        mark = {"supported": "✅ 通过", "weak": "⚠️ 存疑", "unsupported": "❌ 驳回"}[r.status]
        pages = ",".join(str(a.page) for a in r.anchors) or "-"
        print(f"{mark:<12}{pages:<8}{r.claim.text[:60]}")
        if r.status != "supported":
            print(f"{'':<20}└─ {r.reason}")
            for q in r.claim.quotes:
                print(f"{'':<22}引用原文({q.doc}): {q.text[:80]}")

    stats = summarize(results)
    total = len(results) or 1
    print("=" * 78)
    print(
        f"通过 {stats['supported']} / 存疑 {stats['weak']} / 驳回 {stats['unsupported']}"
        f"　—— 幻觉拦截率 {stats['unsupported'] / total:.0%}"
    )

    out = ROOT / ".cache" / "guardrail_result.json"
    out.write_text(
        json.dumps(
            [
                {
                    "claim": r.claim.text,
                    "status": r.status,
                    "reason": r.reason,
                    "anchors": [
                        {"page": a.page, "rect": a.rect, "text": a.source_text, "exact": a.exact}
                        for a in r.anchors
                    ],
                }
                for r in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n完整结果 → {out}")


if __name__ == "__main__":
    main()
