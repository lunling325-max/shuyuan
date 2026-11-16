"""章节级粗筛：长文档两级检索的第一级。

为什么需要它？
一级检索（块级 BM25）在 780 块的长文档上有两个毛病：
1. **打分面铺得太平**：一个只跟某一章有关的问句，会和另外七百多块一起比分数，
   块越小、词越散，真正相关的那块反而不容易冒头；
2. **没有"层级"这个概念**：论文的章节结构是天然的分级信号，块级检索把它丢掉了。

所以先做一次**章节级**打分：把每节的标题（重复计权）与本节首块的开头当作一节
的"摘要"，用同一套词法打分选出候选章节，再只在候选章节内做块级检索。

**只在大文档上启用**。25 块的论文上，粗筛省下的是一毫秒，赔进去的是召回风险 ——
所以调用方按块数设阈值（见 `rag/query.py` 的 `MIN_CHUNKS_FOR_SECTIONS`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.parsing.chunker import Chunk
from app.parsing.text_index import split_tokens
from app.retrieval.bm25 import stem

# 一节最多拿多少个块进候选：再多的章节，标题信号已经很弱了
SECTION_HEAD_CHARS = 240


@dataclass(frozen=True, slots=True)
class SectionHit:
    name: str
    score: float
    positions: tuple[int, ...]   # 该节所有块在 BM25.chunks 里的位置


class SectionIndex:
    """按章节聚合的粗筛索引。构建是 O(块数)，可以随 BM25 一起懒加载。"""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.names: list[str] = []
        self.positions: list[tuple[int, ...]] = []
        self._docs: list[list[str]] = []

        grouped: dict[str, list[int]] = {}
        for pos, chunk in enumerate(chunks):
            grouped.setdefault(chunk.section or "", []).append(pos)

        for name, positions in grouped.items():
            first = chunks[positions[0]]
            # 标题重复一次计权：标题里的词就是这一节最强的信号
            text = f"{name} {name} {first.text[:SECTION_HEAD_CHARS]}"
            self.names.append(name)
            self.positions.append(tuple(positions))
            self._docs.append([stem(t) for t in split_tokens(text)])

        self._tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in self._docs:
            counts: dict[str, int] = {}
            for tok in doc:
                counts[tok] = counts.get(tok, 0) + 1
            self._tf.append(counts)
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1
        self._len = [len(d) for d in self._docs]
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._k1, self._b = 1.5, 0.75

    def __len__(self) -> int:
        return len(self.names)

    def score(self, query: str) -> list[SectionHit]:
        terms = [stem(t) for t in split_tokens(query)]
        if not terms:
            return []
        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self._tf):
            norm = self._k1 * (1 - self._b + self._b * self._len[i] / (self._avgdl or 1))
            s = 0.0
            for term in terms:
                f = counts.get(term)
                if f:
                    s += self._idf.get(term, 0.0) * f * (self._k1 + 1) / (f + norm)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))   # 同分按序号，保证结果可复现
        return [SectionHit(self.names[i], round(s, 4), self.positions[i]) for s, i in scored]

    def candidates(self, queries: list[str], *, top_sections: int = 3) -> set[int]:
        """多条检索式各挑若干节，取并集 —— 与 RRF 的"多路召回"是同一个思路。"""
        picked: set[int] = set()
        for q in queries:
            for hit in self.score(q)[:top_sections]:
                picked.update(hit.positions)
        return picked


def build(chunks: list[Chunk]) -> SectionIndex:
    return SectionIndex(chunks)
