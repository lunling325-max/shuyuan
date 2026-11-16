"""从 idea 生成「架构拓扑图」脚本：Graphviz DOT / Mermaid flowchart / TikZ。

为什么不让模型直接写脚本？
模型写 DOT 或 TikZ 时，语法能"看着像"却是错的（引号、分号、坐标单位都能写崩），
而评审拿到一份编译不过的脚本，比没有更糟。所以分工是：

- **模型只产出结构**：节点、边、以及每个节点属于哪一类（输入/处理/输出）；
- **代码由我们生成**：三种目标语言由同一份结构渲染，语法一定合法。

这也是"无幻觉"在绘图上的体现：模型只能在结构层面出错（会被校验拦下），
不需要它去背 DOT 的语法细节。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_NODES = 12          # 再多就不是"一张图"，而是一堵墙
MAX_EDGES = 24
KINDS = ("input", "module", "output")

TOPO_SYSTEM = """You design an architecture diagram for a research idea.

Return ONLY JSON:
{"title": "<short diagram title, Chinese ok>",
 "nodes": [{"id": "n1", "label": "<short label, <= 10 Chinese chars or <= 4 English words>", "kind": "input|module|output"}, ...],
 "edges": [{"from": "n1", "to": "n2", "label": "<optional short edge label>"}, ...]}

Rules:
- 3 to 10 nodes. First nodes are inputs, last are outputs.
- ids are unique, lowercase, no spaces.
- The diagram must describe a data/processing pipeline, not a list."""


@dataclass(frozen=True, slots=True)
class TopoNode:
    id: str
    label: str
    kind: str = "module"


@dataclass(frozen=True, slots=True)
class TopoEdge:
    src: str
    dst: str
    label: str = ""


@dataclass(frozen=True, slots=True)
class Topology:
    title: str
    nodes: tuple[TopoNode, ...]
    edges: tuple[TopoEdge, ...] = field(default=())

    @property
    def degraded(self) -> bool:
        return not self.nodes


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(text or "")).strip("_").lower()
    return s or fallback


def from_spec(raw: dict | None) -> Topology:
    """把模型给的 JSON 收成可信的结构：非法项直接丢，绝不放行半成品。

    校验只看"能不能画出来"：id 唯一、边两端都存在、节点数有上限。
    丢掉一条边不会让图变成谎言，画一条指向不存在节点的边才会。
    """
    if not isinstance(raw, dict):
        return Topology("", (), ())

    nodes: list[TopoNode] = []
    seen: set[str] = set()
    for i, item in enumerate(raw.get("nodes") or []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        nid = _slug(item.get("id"), f"n{i + 1}")
        while nid in seen:
            nid = f"{nid}_{i}"
        kind = str(item.get("kind") or "module").strip().lower()
        nodes.append(TopoNode(nid, label[:24], kind if kind in KINDS else "module"))
        seen.add(nid)
        if len(nodes) >= MAX_NODES:
            break

    edges: list[TopoEdge] = []
    for item in raw.get("edges") or []:
        if not isinstance(item, dict):
            continue
        src, dst = str(item.get("from") or ""), str(item.get("to") or "")
        if src not in seen or dst not in seen or src == dst:
            continue
        edges.append(TopoEdge(src, dst, str(item.get("label") or "")[:16]))
        if len(edges) >= MAX_EDGES:
            break

    title = str(raw.get("title") or "").strip()[:40]
    return Topology(title, tuple(nodes), tuple(edges))


def _layers(topo: Topology) -> list[list[str]]:
    """按最长路径分层：同一层的节点画在同一列，图才不会糊成一团。"""
    depth = {n.id: 0 for n in topo.nodes}
    for _ in range(len(topo.nodes)):          # 节点数有限，迭代到稳定即可
        changed = False
        for e in topo.edges:
            if depth[e.dst] < depth[e.src] + 1:
                depth[e.dst] = depth[e.src] + 1
                changed = True
        if not changed:
            break
    layers: dict[int, list[str]] = {}
    for node in topo.nodes:
        layers.setdefault(depth[node.id], []).append(node.id)
    return [layers[k] for k in sorted(layers)]


def to_dot(topo: Topology) -> str:
    """Graphviz DOT。形状按节点类别区分，输入=椭圆、处理=方框、输出=双边框。"""
    shape = {"input": "ellipse", "module": "box", "output": "box,peripheries=2"}
    lines = [
        f'digraph "{_dot_escape(topo.title or "architecture")}" {{',
        "  rankdir=LR;",
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]
    for node in topo.nodes:
        lines.append(
            f'  "{_dot_escape(node.id)}" [label="{_dot_escape(node.label)}", '
            f'shape={shape[node.kind]}];'
        )
    for edge in topo.edges:
        label = f' [label="{_dot_escape(edge.label)}"]' if edge.label else ""
        lines.append(f'  "{_dot_escape(edge.src)}" -> "{_dot_escape(edge.dst)}"{label};')
    lines.append("}")
    return "\n".join(lines)


def to_mermaid_flow(topo: Topology) -> str:
    """Mermaid flowchart（不是 xychart —— 这一份画的是结构，不是数据）。"""
    lines = ["flowchart LR"]
    for node in topo.nodes:
        text = _mm_escape(node.label)
        if node.kind == "input":
            lines.append(f'  {node.id}(["{text}"])')
        elif node.kind == "output":
            lines.append(f'  {node.id}[["{text}"]]')
        else:
            lines.append(f'  {node.id}["{text}"]')
    for edge in topo.edges:
        if edge.label:
            lines.append(f"  {edge.src} -->|{_mm_escape(edge.label)}| {edge.dst}")
        else:
            lines.append(f"  {edge.src} --> {edge.dst}")
    return "\n".join(lines)


def to_tikz(topo: Topology) -> str:
    """TikZ：按层摆列，横向箭头。坐标由分层算出来，保证不重叠也不越界。"""
    layers = _layers(topo)
    pos: dict[str, tuple[float, float]] = {}
    for col, ids in enumerate(layers):
        for row, nid in enumerate(ids):
            pos[nid] = (col * 3.4, -row * 1.8 + (len(ids) - 1) * 0.9)
    style = {"input": "draw, ellipse", "module": "draw, rectangle, rounded corners",
             "output": "draw, rectangle, double"}
    lines = [
        "% 由 idea 生成的架构拓扑图；需要 \\usepackage{tikz}",
        "\\begin{tikzpicture}[node distance=1.4cm, >=stealth, font=\\small]",
    ]
    for node in topo.nodes:
        x, y = pos[node.id]
        lines.append(
            f"  \\node[{style[node.kind]}] ({node.id}) at ({x:.2f},{y:.2f}) "
            f"{{{_tex(node.label)}}};"
        )
    for edge in topo.edges:
        label = f" node[above,font=\\scriptsize] {{{_tex(edge.label)}}}" if edge.label else ""
        lines.append(f"  \\draw[->] ({edge.src}) --{label} ({edge.dst});")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines)


def _dot_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _mm_escape(text: str) -> str:
    return str(text).replace('"', "'")


def _tex(text: str) -> str:
    table = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$",
             "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}"}
    return "".join(table.get(ch, ch) for ch in str(text))


def build(idea: str, *, llm_call=None) -> Topology:
    """让模型出结构，我们负责画。

    模型挂了或输出不合规时返回空拓扑（`degraded` 为真），**不抛异常** ——
    与库里其他降级路径一致：画不出图是能力受限，不该让整个请求失败。
    """
    if llm_call is None:
        from app.llm.client import chat_json as llm_call
    try:
        raw = llm_call(f"Idea: {idea}", system=TOPO_SYSTEM, temperature=0.2)
    except Exception:  # noqa: BLE001 —— 模型故障类型不可枚举，降级即可
        return Topology("", (), ())
    return from_spec(raw)
