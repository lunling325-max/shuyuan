"""BM25 词法检索。

为什么在向量检索之外还要留一条词法通道？
1. **专有名词**：`PageRank`、`BGE-M3`、`d_k` 这类词，向量模型经常把近义概念混在一起，
   BM25 靠精确词形反而更准。
2. **可解释**：命中哪个词、命中几次，能直接摆给用户看，答辩时不需要"模型觉得像"。
3. **零依赖**：不下载任何模型就能跑，测试不需要网络。

打分用标准 Robertson-Sparck-Jones 形式：

    score(D, Q) = Σ_{t∈Q} IDF(t) · f(t,D)·(k1+1) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))
    IDF(t)     = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.parsing.chunker import Chunk
from app.parsing.text_index import split_tokens


def stem(token: str) -> str:
    """只做复数归并 —— 词法通道最典型的失配就是单复数。

    实测：问"注意力可视化"，论文那一节的标题是 "Attention Visualizations"，
    `visualization` 与 `visualizations` 在 BM25 里是两个毫不相干的词，
    于是唯一正确的块连候选都进不去。这不是打分问题，是词形问题。

    为什么不做完整词干化（Porter 之类）？
    1. 收益不确定，过词干化（embedding→embedd）会制造假匹配；
    2. 这里只需要覆盖 -s / -es / -ies 三个后缀，规则少到可以逐条验证。

    **只用于 BM25**，绝不能改 `split_tokens`：溯源校验要求逐字比对原文，
    词干化会把"引用原文"变成"引用一个近似词"，那是给幻觉开后门。
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"  # studies -> study, queries -> query
    # 只有在 s/x/z/ch/sh 之后 "es" 才整个是后缀：processes -> process，
    # 而 cases 必须只掉一个 s（掉两个会变 "cas"，跟 "case" 对不上）
    if token.endswith("es") and token[:-2].endswith(("ss", "x", "z", "ch", "sh")):
        return token[:-2]
    # ss / us / is 结尾不是复数：class, various, analysis
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]  # visualizations -> visualization, heads -> head
    return token


@dataclass(frozen=True, slots=True)
class Hit:
    """一条检索结果。`source` 标明来自哪条通道，融合时要看。"""

    chunk: Chunk
    score: float
    source: str = "bm25"


class BM25:
    def __init__(
        self,
        chunks: list[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        index_section: bool = True,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._index_section = index_section
        self._docs: list[list[str]] = [self._tokenize(c) for c in chunks]
        self._len = [len(d) for d in self._docs]
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0

        self._tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in self._docs:
            counts: dict[str, int] = {}
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
            self._tf.append(counts)
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _tokenize(self, chunk: Chunk) -> list[str]:
        # 章节标题也算正文的一部分：论文里很多段落用 "we" 开头，
        # 光看正文分不清在讲哪一节，标题是唯一的强信号。
        text = f"{chunk.section} {chunk.text}" if self._index_section else chunk.text
        return [stem(t) for t in split_tokens(text)]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = [stem(t) for t in split_tokens(query)]
        if not terms or not self.chunks:
            return []

        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self._tf):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / (self._avgdl or 1))
            for term in terms:
                f = counts.get(term)
                if not f:
                    continue
                score += self._idf.get(term, 0.0) * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, i))

        scored.sort(key=lambda x: (-x[0], x[1]))  # 同分按 index，保证可复现
        return [Hit(chunk=self.chunks[i], score=round(s, 4)) for s, i in scored[:k]]


def reciprocal_rank_fusion(
    rankings: list[list[Hit]],
    *,
    k: int = 60,
    limit: int = 8,
    weights: list[float] | None = None,
) -> list[Hit]:
    """倒数排名融合（RRF）。

    RRF 只吃名次、不吃分数，所以不需要把 BM25 的 0.83 和余弦的 0.71
    强行拉到一个量纲上 —— 这是多路召回最省事也最稳的融合方式。

    `weights` 给每路一个票重。实测下来，**第一条检索式通常最贴题**
    （它是模型对原问题的直译），后两条是同义改写、更容易跑偏，
    所以默认让后几路降权，避免"两路跑偏的票压过一路精准的票"。
    """
    fused: dict[int, float] = {}
    first: dict[int, Hit] = {}
    for i, ranking in enumerate(rankings):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, hit in enumerate(ranking, start=1):
            idx = hit.chunk.index
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
            first.setdefault(idx, hit)

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return [
        Hit(chunk=first[idx].chunk, score=round(score, 5), source="rrf")
        for idx, score in ordered
    ]
