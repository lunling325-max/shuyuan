"""架构拓扑图：结构校验与三种脚本渲染。

重点不在"能画出图"，而在**坏输入不会产出坏脚本**：
模型给的边指向不存在的节点、id 重复、节点过多，都必须在渲染前被挡掉 ——
否则渲染出来的 DOT/TikZ 语法看着合法，语义却是错的。
"""

from __future__ import annotations

from app.analysis.topology import (
    MAX_NODES,
    TopoEdge,
    TopoNode,
    Topology,
    build,
    from_spec,
    to_dot,
    to_mermaid_flow,
    to_tikz,
)


def test_from_spec_drops_edges_to_unknown_nodes():
    raw = {
        "title": "pipeline",
        "nodes": [{"id": "a", "label": "输入"}, {"id": "b", "label": "处理"}],
        "edges": [{"from": "a", "to": "b"}, {"from": "a", "to": "ghost"}, {"from": "b", "to": "b"}],
    }
    topo = from_spec(raw)
    assert [n.id for n in topo.nodes] == ["a", "b"]
    assert [(e.src, e.dst) for e in topo.edges] == [("a", "b")]


def test_from_spec_dedupes_ids_and_caps_count():
    raw = {"nodes": [{"id": "x", "label": f"节点{i}"} for i in range(MAX_NODES + 5)]}
    topo = from_spec(raw)
    assert len(topo.nodes) == MAX_NODES
    assert len({n.id for n in topo.nodes}) == MAX_NODES          # id 唯一
    assert [n.id for n in topo.nodes][0] == "x"
    assert [n.id for n in topo.nodes][1].startswith("x_")        # 重名自动加后缀


def test_from_spec_rejects_garbage():
    assert from_spec(None).degraded
    assert from_spec({"nodes": [{"id": "a"}]}).degraded           # 没有 label
    assert from_spec({"nodes": "not-a-list"}).degraded


def test_dot_escapes_quotes_and_backslash():
    topo = Topology("t", (TopoNode("a", 'say "hi"\\now'),), ())
    dot = to_dot(topo)
    assert '\\"hi\\"' in dot and "\\\\now" in dot
    assert dot.rstrip().endswith("}")


def test_mermaid_uses_flowchart_not_xychart():
    topo = Topology("t", (TopoNode("a", "输入", "input"), TopoNode("b", "输出", "output")),
                    (TopoEdge("a", "b", "流转"),))
    text = to_mermaid_flow(topo)
    assert text.startswith("flowchart LR")
    assert 'a(["输入"])' in text            # 输入用圆角
    assert 'b[["输出"]]' in text            # 输出用双边框
    assert "a -->|流转| b" in text


def test_tikz_layers_do_not_overlap():
    """分层摆放：同一列不重叠，不同列横向错开 —— 坐标算错就会糊成一团。"""
    nodes = tuple(TopoNode(f"n{i}", f"节点{i}") for i in range(4))
    edges = (TopoEdge("n0", "n1"), TopoEdge("n0", "n2"), TopoEdge("n1", "n3"),
             TopoEdge("n2", "n3"))
    text = to_tikz(Topology("t", nodes, edges))
    coords = [line.split("at (")[1].split(")")[0] for line in text.splitlines()
              if " at (" in line]
    assert len(coords) == 4
    # n0 与 n3 在同一列的两端，n1/n2 同列但纵坐标必须不同
    y = {}
    for nid, c in zip(["n0", "n1", "n2", "n3"], coords):
        y[nid] = c.split(",")[1]
    assert y["n1"] != y["n2"]
    assert text.count("\\draw[->]") == 4


def test_build_degrades_when_model_fails():
    def boom(*a, **kw):
        raise RuntimeError("model down")

    topo = build("任意想法", llm_call=boom)
    assert topo.degraded and topo.nodes == ()
