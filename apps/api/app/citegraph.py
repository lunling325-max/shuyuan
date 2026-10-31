"""引用图谱：从每篇论文的参考文献里认出"它引用了库里的哪几篇"，再按 PageRank 排影响力。

三件容易被做错的事，这里都显式处理了：

1. **一篇论文只算一个节点**。库里同时有 v1 和 v7 时，它们在图上是**同一个点** ——
   别人引用的是这篇论文，不是某个版本。版本用 `paper_ids` 挂在节点上。
   不合并的话，PageRank 会把同一篇的票数劈成两半，排名直接失真。

2. **认引用靠 arXiv 号优先、标题次之**。参考文献里的标题常被截断、断词、
   大小写不一，纯字符串相等会把真引用漏掉。实测这五份样本上，真引用的
   标题 Jaccard 是 **1.00**，最像的假引用是 **0.50**（GPT-2 的 "Language models
   are unsupervised multitask learners" 对上 GPT-3 的 "Few-Shot Learners"），
   隔得够开，阈值才敢定在 0.70。

3. **外部引用不是全画**。库里五篇论文合计引了三百多篇外部文献，全画出来是一团
   毛线，什么也看不出来。只画**被库内 ≥2 篇共同引用**的（Adam、ELMo、GPT-2 这些），
   它们才真正在图上起连接作用。

PageRank 与布局都是手写的，不引 networkx：算法本身二十行，
多一个依赖换不来任何鲁棒性，反而在离线装机时多一个坑。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from app.references import Reference, extract_references, normalize_title, title_similarity
from app.versioning import classify, jaccard

# 标题判定阈值。实测：真引用 1.00，最像的假引用 0.50（GPT-2 vs GPT-3），
# 中间是空的，取 0.70 两边都留了余量
TITLE_MATCH = 0.70
MIN_SHARED_CITES = 2  # 外部文献至少被库内几篇引用才上图
MAX_EXTERNAL = 20  # 外部节点上限，超出按被引次数截断（图上画不下也没人看）
DAMPING = 0.85
MAX_ITER = 200
TOL = 1e-10

# arXiv 新式编号 YYMM.NNNNN：YY 是首次投稿的年份，MM 是月份
_ARXIV_YYMM = re.compile(r"^(\d{2})(0[1-9]|1[0-2])\.")


def _paper_year(paper: PaperLike) -> str:
    """库内论文显示的年份。

    不能直接用 PDF 元数据的生成日期：Transformer 的 v7 是 2024 年重新导出的，
    按那个显示就成了"Attention Is All You Need (2024)"；BERT 的 PDF 生成于 2019，
    而它 2018 年就发表了。arXiv 号里的 YYMM 是作者首次投稿时填的，
    和引用惯例一致 —— 四篇样本实测 4/4 对。取不到才退回生成日期。
    """
    m = _ARXIV_YYMM.match(paper.arxiv_id or "")
    if m:
        yy = int(m.group(1))
        return str(2000 + yy if yy < 90 else 1900 + yy)  # arXiv 1991 年才有，>=90 是 19xx
    return (paper.created_at or "")[:4]


class PaperLike(Protocol):
    """图谱只依赖这几个字段，不绑定具体的注册表实现。"""

    id: str
    title: str
    arxiv_id: str
    created_at: str
    uploaded_at: str
    path: Path
    chunks: list
    fingerprint: set[str]


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    label: str
    kind: str  # "paper"（库内，可点开）| "external"（只在参考文献里）
    year: str
    paper_ids: tuple[str, ...]  # 库内对应的版本 id，最新版在最后一个
    versions: int
    cites: int  # 出度：它引用了图上几篇
    cited_by: int  # 入度：库内几篇引用了它
    rank: float
    x: float  # 归一化坐标 0..1，前端直接乘画布尺寸
    y: float


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str  # 引用方
    target: str  # 被引方
    via: str  # "arxiv" | "title"
    score: float

    def as_tuple(self) -> tuple[str, str]:
        return (self.source, self.target)


@dataclass(slots=True)
class CitationGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    def node(self, node_id: str) -> GraphNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)


# ---------------------------------------------------------------- 聚类


def _cluster(papers: list[PaperLike]) -> list[list[PaperLike]]:
    """把同一篇论文的不同版本并成一组（并查集）。

    v1 与 v7 各自都和中间版本像，但 v1 与 v7 未必超过阈值 ——
    所以要做**传递闭包**，不能用"两两比对"直接分组。
    """
    parent = list(range(len(papers)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            if find(i) == find(j):
                continue
            rel = classify(jaccard(papers[i].fingerprint, papers[j].fingerprint))
            if rel.kind in ("identical", "new_version"):
                parent[find(i)] = find(j)

    groups: dict[int, list[PaperLike]] = defaultdict(list)
    for i, p in enumerate(papers):
        groups[find(i)].append(p)
    return list(groups.values())


def _newest(group: list[PaperLike]) -> PaperLike:
    """同组里**最新**的那版当代表：引用关系以它为准。

    "新"按 PDF 生成日期判（版本谱系同一套规则），拿不到才退回上传时间。
    """
    return max(group, key=lambda p: (p.created_at or "", p.uploaded_at or "", p.id))


# ---------------------------------------------------------------- PageRank


def pagerank(
    node_ids: Iterable[str],
    edges: Iterable[tuple[str, str]],
    *,
    damping: float = DAMPING,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
) -> dict[str, float]:
    """标准 PageRank 幂迭代。

    悬挂节点（出度为 0，比如外部文献）不做特殊处理会让概率漏掉，
    这里把它们的权重按均匀分布重新分配 —— 否则节点一多，排名会整体漂移。
    """
    ids = sorted(set(node_ids))
    n = len(ids)
    if n == 0:
        return {}

    out_deg: dict[str, int] = {k: 0 for k in ids}
    out_links: dict[str, list[str]] = defaultdict(list)
    for s, t in sorted(set(edges)):
        if s in out_deg and t in out_deg and s != t:
            out_deg[s] += 1
            out_links[s].append(t)

    rank = {k: 1.0 / n for k in ids}
    for _ in range(max_iter):
        dangling = sum(rank[k] for k in ids if out_deg[k] == 0)
        base = (1.0 - damping) / n + damping * dangling / n
        nxt = {k: base for k in ids}
        for k in ids:
            if not out_deg[k]:
                continue
            share = damping * rank[k] / out_deg[k]
            for target in out_links[k]:  # 票流向"被引用的那一方"，方向别弄反
                nxt[target] += share
        delta = sum(abs(nxt[k] - rank[k]) for k in ids)
        rank = nxt
        if delta < tol:
            break
    return rank


def _layout(ranked: list[str], rank: dict[str, float]) -> dict[str, tuple[float, float]]:
    """同心环布局：排名最高的在圆心，其余按排名向外一圈圈铺开。

    为什么不用力导向？因为力导向的结果依赖初始位置与迭代顺序，
    **两次打开同一个库，节点位置会不一样**。演示时点开图谱发现布局变了，
    读者第一反应是"数据不稳"，信任就没了。环布局是完全确定的。

    半径按排名给，一眼能看出"谁更靠中心"，比随机散点好读得多。
    坐标是**正方形**归一化坐标（0..1），前端按短边放缩 —— 各环才是正圆；
    直接按 660×440 画布放缩会变成椭圆，环上间距一边松一边挤。
    """
    rings = [1, 6, 12, 18]
    # 第一环给得远一点：库内论文的节点和标签都最大，挤在圆心附近会互相压住
    radii = [0.0, 0.24, 0.40, 0.47]
    pos: dict[str, tuple[float, float]] = {}
    i = 0
    for ring, (cap, r) in enumerate(zip(rings, radii)):
        members = ranked[i : i + cap]
        i += len(members)
        if not members:
            break
        if ring == 0:
            pos[members[0]] = (0.5, 0.5)
            continue
        step = 2 * math.pi / len(members)
        # 第一环特意从 -60° 起铺：正上方与正下方各留空位，
        # 圆心那个节点的标签正好写在上下，不会被环上的邻居压住。
        # 外圈再错开半格 —— 两环的节点在同一条射线上时，标签会首尾撞在一起。
        offset = -math.pi / 3 if ring == 1 else -math.pi / 2 + ring * 0.35 + math.pi / len(members)
        for k, node_id in enumerate(members):
            ang = offset + k * step
            pos[node_id] = (0.5 + r * math.cos(ang), 0.5 + r * math.sin(ang))
    return pos


# ---------------------------------------------------------------- 建图


def _ref_key(title: str) -> str:
    digest = hashlib.sha1(normalize_title(title).encode("utf-8")).hexdigest()[:10]
    return f"ref:{digest}"


def _match(
    ref: Reference,
    clusters: list[tuple[str, PaperLike]],
    skip: str,
) -> tuple[str, str, float] | None:
    """一条引用指向库里的哪篇。返回 (节点 id, 匹配依据, 分数)。"""
    if ref.arxiv_id:
        for node_id, paper in clusters:
            if node_id != skip and paper.arxiv_id and paper.arxiv_id == ref.arxiv_id:
                return node_id, "arxiv", 1.0
    best: tuple[str, float] | None = None
    if ref.title:
        for node_id, paper in clusters:
            if node_id == skip:
                continue
            score = title_similarity(ref.title, paper.title)
            if score >= TITLE_MATCH and (best is None or score > best[1]):
                best = (node_id, score)
    if best is None:
        return None
    return best[0], "title", round(best[1], 3)


def build_graph(papers: list[PaperLike], *, max_external: int = MAX_EXTERNAL) -> CitationGraph:
    """建图。纯函数：给一批论文，返回节点、边、PageRank 与坐标。"""
    graph = CitationGraph()
    if not papers:
        return graph

    groups = _cluster(papers)
    reps = [_newest(g) for g in groups]
    clusters = [(f"paper:{p.id}", p) for p in reps]
    by_id = {node_id: p for node_id, p in clusters}

    refs_by_cluster: dict[str, list[Reference]] = {}
    total_refs = parsed_refs = 0
    for node_id, paper in clusters:
        try:
            refs = extract_references(paper.chunks, paper.path)
        except Exception:  # noqa: BLE001 —— 一篇抽不出来不该让整张图消失
            graph.notes.append(f"《{paper.title[:40]}》的参考文献没能解析，它在图上没有出边")
            refs = []
        refs_by_cluster[node_id] = refs
        total_refs += len(refs)
        parsed_refs += sum(1 for r in refs if r.is_parsed)

    # 库内引用：一条引用最多投一票，重复引用（参考文献里出现两次）不重复计
    edges: dict[tuple[str, str], GraphEdge] = {}
    # 外部引用按归一化标题归并，记下"谁引了它"
    external: dict[str, dict[str, object]] = {}

    for node_id, refs in refs_by_cluster.items():
        for ref in refs:
            hit = _match(ref, clusters, skip=node_id)
            if hit:
                target, via, score = hit
                key = (node_id, target)
                if key not in edges or score > edges[key].score:
                    edges[key] = GraphEdge(node_id, target, via, score)
            elif ref.is_parsed:
                rid = _ref_key(ref.title)
                slot = external.setdefault(
                    rid, {"label": ref.title, "year": ref.year, "cited_by": set()}
                )
                if len(ref.title) > len(str(slot["label"])):  # 取最完整的那条写法
                    slot["label"] = ref.title
                if not slot["year"]:
                    slot["year"] = ref.year
                slot["cited_by"].add(node_id)  # type: ignore[union-attr]

    # 外部引用的**全部**都进 PageRank，只是不全画出来。
    # 只拿"要画的那几个"去算会翻车：真实引用图上它们本来都是节点，
    # 抽掉之后剩下的全是悬挂节点，质量按均匀分布摊到每个点上，
    # 结果"被 3 篇引用"的 Transformer 排在一堆"被 2 篇引用"的外部文献后面（实测如此）。
    for rid, slot in external.items():
        for citer in sorted(slot["cited_by"]):  # type: ignore[arg-type]
            edges.setdefault((citer, rid), GraphEdge(citer, rid, "title", 1.0))

    shared = [rid for rid, slot in external.items() if len(slot["cited_by"]) >= MIN_SHARED_CITES]
    # 外部文献之间没有边（我们没解析过它们的参考文献），所以它们只能按出度稳定排序
    shared.sort(key=lambda rid: (-len(external[rid]["cited_by"]), rid))  # type: ignore[arg-type]
    shown = shared[:max_external]
    single = len(external) - len(shared)
    keep = set(shown) | {node_id for node_id, _ in clusters}

    graph.edges = sorted(
        (e for e in edges.values() if e.target in keep), key=lambda e: (e.source, e.target)
    )

    # PageRank **只算库内论文之间**的引用。为什么不把外部文献算进去：
    # 它们在图上全是叶子（我们没解析它们的参考文献），不往外传任何权重，
    # 于是"被 2 篇引用"的 GloVe 会排在"被 3 篇引用"的 Transformer 前面 ——
    # 实测就是这样（外部叶子 0.0442 vs 论文 0.0388），榜首是读者没法点开的文献，
    # 排名因此失去意义。名次留给"库里这几篇互相之间谁更核心"。
    rank = pagerank(
        [node_id for node_id, _ in clusters],
        [e.as_tuple() for e in edges.values()
         if e.target.startswith("paper:") and e.source.startswith("paper:")],
    )

    # 度数只数**库内论文之间**的边：外部文献的"被引次数"才有意义（几篇库内论文引它），
    # 论文的"引用数"也只该数它引了几篇库内论文，否则两边数出来的东西不是一回事
    in_deg: dict[str, int] = defaultdict(int)
    out_deg: dict[str, int] = defaultdict(int)
    for e in graph.edges:
        in_deg[e.target] += 1
        if e.target.startswith("paper:"):
            out_deg[e.source] += 1

    labels: dict[str, tuple[str, str, tuple[str, ...], int, str]] = {}
    for node_id, paper in clusters:
        group = next(g for g in groups if _newest(g) is paper)
        versions = sorted(group, key=lambda p: (p.created_at or "", p.uploaded_at or "", p.id))
        labels[node_id] = (
            paper.title,
            "paper",
            tuple(p.id for p in versions),
            len(versions),
            _paper_year(paper),
        )
    for rid in shown:
        slot = external[rid]
        labels[rid] = (
            str(slot["label"]),
            "external",
            (),
            1,
            str(slot["year"] or ""),
        )

    # 从中心往外：先排库内论文（按 PageRank），再排外部文献（按被几篇引用）。
    # 读者最该看见的是"库里的核心论文在中间"，不是"某篇没见过的论文在中间"。
    def order_key(nid: str) -> tuple:
        return (0 if nid.startswith("paper:") else 1, -rank.get(nid, 0.0), -in_deg[nid], nid)

    ranked = sorted(labels, key=order_key)
    pos = _layout(ranked, rank)

    graph.nodes = [
        GraphNode(
            id=nid,
            label=labels[nid][0],
            kind=labels[nid][1],
            year=labels[nid][4],
            paper_ids=labels[nid][2],
            versions=labels[nid][3],
            cites=out_deg[nid],
            cited_by=in_deg[nid],
            rank=round(rank.get(nid, 0.0), 6),
            x=round(pos.get(nid, (0.5, 0.5))[0], 4),
            y=round(pos.get(nid, (0.5, 0.5))[1], 4),
        )
        for nid in ranked
    ]

    if len(shared) > len(shown):
        graph.notes.append(f"还有 {len(shared) - len(shown)} 篇外部文献被多篇共同引用，图上放不下，省掉了")
    if single:
        graph.notes.append(f"另有 {single} 篇只被库内一篇引用，图上省略")

    graph.stats = {
        "papers": len(papers),
        "clusters": len(clusters),
        "merges": len(papers) - len(clusters),
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "references": total_refs,
        "references_parsed": parsed_refs,
        "references_matched": sum(1 for e in graph.edges if e.target.startswith("paper:")),
        "external_shown": len(shown),
        "external_omitted": len(shared) - len(shown) + single,
    }
    return graph
