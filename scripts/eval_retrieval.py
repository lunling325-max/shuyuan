"""检索质量评测：单路 BM25 vs 检索词扩展 + RRF。

评测集很小（人工标的 18 题），但每道题都能在论文里找到唯一正确的小节，
所以 Hit@1 是硬指标，不做"模糊算对"。

用法（需在 apps/api 下跑，用项目自带的 venv）：
    uv run python ../../scripts/eval_retrieval.py
    ... --no-expand          # 单路 BM25，零扩展
    ... --build-cache        # 冻结 18 题的检索式扩展（对比实验的前提）
    ... --dense              # 只走向量通道（消融）
    ... --hybrid [--dense-weight 0.4]
    ... --weighted           # 旧对照：首条检索式加权

E-003 的教训：链路里有 LLM，评测就必须先冻结它的输出。
本脚本默认读 data/eval_expansions.json，不重新调模型。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from ask import load_dotenv  # 复用 .env 读取

# (问题, 正确答案所在小节的关键字)
CASES: list[tuple[str, str]] = [
    ("为什么要除以 sqrt(dk)？", "3.2.1"),
    ("位置编码为什么用正弦函数？", "3.5"),
    ("label smoothing 的 epsilon 是多少？", "5.4"),
    ("多头注意力用了几个头？", "3.2.2"),
    ("WMT 2014 英德翻译 BLEU 拿了多少？", "6.1"),
    ("warmup 训练步数是多少？", "5.3"),
    ("dropout 概率 Pdrop 是多少？", "5.4"),
    ("编码器和解码器各堆几层？", "3.1"),
    ("自注意力每层的计算复杂度怎么比较？", "4 Why Self-Attention"),
    ("Adam 优化器的两个 beta 是多少？", "5.3"),
    ("残差连接和 layer normalization 怎么接的？", "3.1"),
    ("英德翻译用的是哪个数据集？", "5.1"),
    ("改变注意力头数会影响英法翻译吗？", "6.2"),
    ("注意力可视化能看到什么现象？", "Attention Visualizations"),
    ("What is the dropout rate Pdrop?", "5.4"),
    ("How many heads are used in multi-head attention?", "3.2.2"),
    ("What BLEU score on WMT 2014 English-to-German?", "6.1"),
    ("Why is self-attention faster than recurrent layers?", "4 Why Self-Attention"),
]


CACHE = ROOT / "data" / "eval_expansions.json"


def _load_cache() -> dict[str, list[str]]:
    if not CACHE.exists():
        return {}
    import json

    return json.loads(CACHE.read_text(encoding="utf-8"))


def _save_cache(cache: dict[str, list[str]]) -> None:
    import json

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(ROOT / "pdf" / "1706.03762v7.pdf"))
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--no-expand", action="store_true")
    ap.add_argument("--build-cache", action="store_true", help="重新生成检索词扩展缓存")
    ap.add_argument("--weighted", action="store_true", help="对比实验：给首条检索式加权")
    ap.add_argument("--dense", action="store_true", help="只走向量通道")
    ap.add_argument("--hybrid", action="store_true", help="词法 + 向量并行（默认只走词法）")
    ap.add_argument("--dense-weight", type=float, default=1.0, help="融合时向量通道的权重")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")

    from app.parsing.chunker import chunk_pdf
    from app.rag.query import expand_query
    from app.retrieval import BM25, reciprocal_rank_fusion
    from app.retrieval.dense import DenseIndex
    from app.retrieval.embeddings import load_embedder

    chunks = chunk_pdf(args.pdf)
    retriever = BM25(chunks)
    expand = not args.no_expand

    dense = None
    if args.dense or args.hybrid:
        embedder = load_embedder()
        if embedder is None:
            print("找不到本地模型，无法跑向量通道。请先下载 models/bge-small-en-v1.5")
            return 2
        dense = DenseIndex.build(chunks, embedder)
        if args.dense:
            retriever = None  # 只保留向量通道，做消融

    # 检索词扩展是 LLM 调用，天然有抖动；把结果缓存下来，
    # 才能让 "等权 vs 加权" 这种对比只受融合策略影响，不受模型情绪影响。
    cache = {} if (args.build_cache or not expand) else _load_cache()
    if args.build_cache:
        for q, _ in CASES:
            cache[q] = expand_query(q)
        _save_cache(cache)
        print(f"已缓存 {len(cache)} 条检索词扩展 → {CACHE.relative_to(ROOT)}\n")
    elif expand and not cache:
        print("没有缓存，先跑 --build-cache 固定检索词；本次直接调用模型。\n")

    hit1 = hit3 = 0
    rows = []
    for q, expected in CASES:
        queries = (cache.get(q) or expand_query(q)) if expand else [q]
        searchers = [
            (s, args.dense_weight if s is dense and s is not None else 1.0)
            for s in (retriever, dense)
            if s is not None
        ]
        if len(searchers) == 1 and len(queries) == 1:
            hits = searchers[0][0].search(queries[0], k=args.k)
        else:
            pairs = [(s.search(x, k=args.k), w) for x in queries for s, w in searchers]
            pairs = [p for p in pairs if p[0]]
            if args.weighted:  # 旧的"首条检索式加权"对照
                pairs = [(r, w * (1.0 if i == 0 else 0.6)) for i, (r, w) in enumerate(pairs)]
            rankings = [r for r, _ in pairs]
            weights = [w for _, w in pairs] if any(w != 1.0 for _, w in pairs) else None
            hits = reciprocal_rank_fusion(rankings, limit=args.k, weights=weights) if rankings else []
        sections = [h.chunk.section for h in hits]
        ok1 = bool(sections) and expected in sections[0]
        ok3 = any(expected in s for s in sections[:3])
        hit1 += ok1
        hit3 += ok3
        top = sections[0] if sections else "（无召回）"
        rows.append((q, expected, top, ok1, ok3))

    print(f"{'问题':<42} {'期望':<24} {'第一名':<44} @1  @3")
    print("-" * 128)
    for q, expected, top, ok1, ok3 in rows:
        print(f"{q[:40]:<42} {expected[:22]:<24} {top[:42]:<44} "
              f"{'✓' if ok1 else '✗':<3} {'✓' if ok3 else '✗'}")

    n = len(CASES)
    print("-" * 128)
    print(f"Hit@1 = {hit1}/{n} = {hit1 / n:.1%}    Hit@3 = {hit3}/{n} = {hit3 / n:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
