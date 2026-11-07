"""数据可视化与图注提炼的测试（全程离线）。

重点覆盖四类底线：

1. 解析要"坏得明白"：BOM、千分位、百分比、空行、列数不一致都要有确定行为；
2. 四种图都能产出结构合法、可被 XML 解析的自包含 SVG；
3. 不同量纲绝不使用双 Y 轴，而是 small multiples；
4. 图注里的数字必须能在统计卡里核实，模型不可用不能 500。

所有 CSV 都在 `tmp_path` 里临时写出再上传，仓库里不留测试数据文件。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.analysis.chart import build_chart_data
from app.analysis.csv_table import (
    CsvDecodeError,
    CsvEmptyError,
    CsvFormatError,
    parse_csv,
)
from app.api.routers.dataviz import build_caption, router, validate_caption


@pytest.fixture()
def client():
    """main.py 由主控挂载；测试里只装本路由，照抄 test_review.py 的写法。"""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _write(tmp_path, text: str | bytes, name: str = "data.csv") -> bytes:
    data = text.encode("utf-8") if isinstance(text, str) else text
    path = tmp_path / name
    path.write_bytes(data)
    return path.read_bytes()


def _upload(
    client,
    tmp_path,
    text: str | bytes,
    *,
    name: str = "data.csv",
    caption: bool = False,
    **form,
):
    data = _write(tmp_path, text, name)
    payload = {"caption": "true" if caption else "false"}
    payload.update({key: str(value) for key, value in form.items()})
    return client.post(
        "/api/dataviz/chart",
        files={"file": (name, data, "text/csv")},
        data=payload,
    )


def _fake_llm(payload: dict):
    def call(user, *, system=None, **kwargs):
        return payload

    return call


def _point_count(svg: str) -> int:
    return len(re.findall(r'class="point"', svg))


def _coordinate_blocks(tikz: str) -> list[int]:
    return [
        len(re.findall(r"\([^()]+\)", block))
        for block in re.findall(r"coordinates \{([^}]*)\}", tikz)
    ]


# ================================================================ ① 解析


def test_parse_bom_thousands_percent_and_blank_lines(tmp_path):
    text = (
        "name,revenue,rate\n"
        '"A, Inc","1,234",12.5%\n'
        "\n"
        "B,2000,10%\n"
        ",,8%\n"
    )
    data = _write(tmp_path, text.encode("utf-8-sig"))
    table = parse_csv(data)

    assert [column.name for column in table.columns] == ["name", "revenue", "rate"]
    assert table.columns[1].kind == "number"
    assert table.columns[2].unit == "%"
    assert table.columns[1].values[:2] == [1234.0, 2000.0]
    assert table.columns[2].values[0] == 12.5
    assert table.columns[0].missing == 1
    assert table.stats["revenue"].max == 2000.0
    assert table.stats["rate"].mean == pytest.approx(10.166667, abs=1e-6)


def test_parse_semicolon_and_tab_delimiters(tmp_path):
    for delimiter, name in ((";", "semi.csv"), ("\t", "tab.csv")):
        data = _write(
            tmp_path,
            f"name{delimiter}value\nA{delimiter}1\nB{delimiter}2\n",
            name,
        )
        table = parse_csv(data)
        assert [column.name for column in table.columns] == ["name", "value"]
        assert table.stats["value"].n == 2


def test_parse_pads_short_rows_and_records_note(tmp_path):
    data = _write(tmp_path, "a,b,c\n1,2\n3,4,5\n")
    table = parse_csv(data)

    assert table.columns[2].values == [None, 5.0]
    assert any("列数少于表头" in note for note in table.notes)


def test_parse_rejects_ragged_extra_columns(tmp_path):
    data = _write(tmp_path, "a,b\n1,2,3\n")
    with pytest.raises(CsvFormatError, match="列"):
        parse_csv(data)


def test_parse_rejects_single_row(tmp_path):
    data = _write(tmp_path, "a,b\n")
    with pytest.raises(CsvFormatError, match="至少需要表头"):
        parse_csv(data)


def test_parse_rejects_empty_table(tmp_path):
    data = _write(tmp_path, "\n  \n\n")
    with pytest.raises(CsvEmptyError):
        parse_csv(data)


def test_parse_rejects_non_utf8_bytes(tmp_path):
    data = _write(tmp_path, b"\xff\xfe\x00a,b\n1,2\n", "gbk.csv")
    with pytest.raises(CsvDecodeError, match="UTF-8"):
        parse_csv(data)


def test_parse_handles_no_header(tmp_path):
    data = _write(tmp_path, "A,1\nB,2\nC,3\n")
    table = parse_csv(data)

    assert [column.name for column in table.columns] == ["列1", "列2"]
    assert any("没有识别到表头" in note for note in table.notes)
    assert table.columns[1].kind == "number"


# ============================================================ ② 自动推断


def test_infer_line_for_monotonic_numeric_x(tmp_path):
    table = parse_csv(_write(tmp_path, "epoch,loss,acc\n1,1.0,0.50\n2,0.8,0.62\n3,0.6,0.71\n"))
    chart = build_chart_data(table)

    assert chart.kind == "line"
    assert chart.x_label == "epoch"
    assert [series.name for series in chart.series] == ["loss", "acc"]


def test_infer_bar_for_category_and_single_numeric(tmp_path):
    table = parse_csv(_write(tmp_path, "city,sales\nA,10\nB,20\nC,15\n"))
    chart = build_chart_data(table)

    assert chart.kind == "bar"
    assert chart.x_label == "city"
    assert [series.name for series in chart.series] == ["sales"]


def test_infer_scatter_for_two_numeric_columns(tmp_path):
    # 横轴刻意不单调，避免被"有序递增 -> line"规则抢走
    table = parse_csv(
        _write(tmp_path, "height,weight\n170,55\n160,64\n180,72\n175,68\n")
    )
    chart = build_chart_data(table)

    assert chart.kind == "scatter"
    assert chart.x_label == "height"
    assert chart.series[0].name == "weight"


def test_infer_line_for_date_axis_not_bar(tmp_path):
    table = parse_csv(
        _write(
            tmp_path,
            "date,sales\n2024-01-01,10\n2024-02-01,12\n2024-03-01,15\n",
        )
    )
    chart = build_chart_data(table)

    assert chart.kind == "line"
    assert chart.x_label == "date"
    assert [series.name for series in chart.series] == ["sales"]


def test_high_cardinality_category_falls_back_to_line(tmp_path):
    rows = "\n".join(f"item{index},{index}" for index in range(30))
    table = parse_csv(_write(tmp_path, "item,value\n" + rows + "\n"))
    chart = build_chart_data(table)

    assert chart.kind == "line", "30 个类别不适合柱状图"
    assert chart.x_label == "item"


def test_explicit_line_without_x_uses_category_labels(tmp_path):
    table = parse_csv(_write(tmp_path, "city,sales\nA,10\nB,20\nC,15\n"))
    chart = build_chart_data(table, kind="line")

    assert chart.kind == "line"
    assert chart.x_label == "city"


def test_infer_radar_for_category_and_three_metrics(tmp_path):
    table = parse_csv(
        _write(
            tmp_path,
            "algo,precision,recall,f1\nA,0.8,0.7,0.75\nB,0.9,0.6,0.72\nC,0.7,0.9,0.79\n",
        )
    )
    chart = build_chart_data(table)

    assert chart.kind == "radar"
    assert chart.labels == ["precision", "recall", "f1"]
    assert [series.name for series in chart.series] == ["A", "B", "C"]


def test_explicit_kind_beats_inference(tmp_path):
    table = parse_csv(_write(tmp_path, "epoch,loss\n1,1.0\n2,0.8\n3,0.6\n"))
    chart = build_chart_data(table, kind="bar")

    assert chart.kind == "bar"


def test_more_than_eight_series_merge_into_other(tmp_path):
    header = "x," + ",".join(f"s{index}" for index in range(1, 10))
    rows = "\n".join(
        f"{x}," + ",".join(str(x + index) for index in range(1, 10))
        for x in (1, 2, 3)
    )
    table = parse_csv(_write(tmp_path, header + "\n" + rows + "\n"))
    chart = build_chart_data(table)

    assert len(chart.series) == 8
    assert chart.series[-1].name == "其他"
    # 第 8、9 个序列逐行取均值
    assert chart.series[-1].values == [9.5, 10.5, 11.5]
    assert "其他" in chart.note


# ============================================================== ③ 四种图


@pytest.mark.parametrize(
    "name,text,form,expected_paths,expected_points",
    [
        (
            "line",
            "epoch,loss,acc\n1,1.0,0.50\n2,0.8,0.62\n3,0.6,0.71\n4,0.5,0.77\n",
            {},
            2,
            8,
        ),
        # 柱状图每个类别一根柱子，所以 series 路径数 = 类别数
        ("bar", "city,sales\nA,10\nB,20\nC,15\n", {}, 3, 0),
        (
            "scatter",
            "height,weight\n170,55\n160,64\n180,72\n175,68\n",
            {"kind": "scatter"},
            0,
            4,
        ),
        (
            "radar",
            "algo,precision,recall,f1\nA,0.8,0.7,0.75\nB,0.9,0.6,0.72\nC,0.7,0.9,0.79\n",
            {},
            3,
            9,
        ),
    ],
)
def test_four_kinds_produce_well_formed_svg(
    client, tmp_path, name, text, form, expected_paths, expected_points
):
    response = _upload(client, tmp_path, text, **form)
    assert response.status_code == 200, response.text
    body = response.json()
    svg = body["chart"]["svg"]

    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "viewBox" in svg
    assert "xmlns" in svg
    ET.fromstring(svg)  # 必须能被当作 XML 解析，标签里的 < > 不能破坏结构

    assert len(re.findall(r'class="series"', svg)) == expected_paths
    assert _point_count(svg) == expected_points
    assert body["chart"]["width"] > 0
    assert body["chart"]["height"] > 0


def test_svg_escapes_markup_in_labels(client, tmp_path):
    text = 'name,value\n"a<b & c",1\nB,2\n'
    response = _upload(client, tmp_path, text, kind="bar")
    assert response.status_code == 200

    svg = response.json()["chart"]["svg"]
    assert "<b>" not in svg
    assert "&lt;b" in svg
    assert "&amp;" in svg
    ET.fromstring(svg)


def test_single_series_has_no_legend_but_multi_series_does(client, tmp_path):
    single = _upload(client, tmp_path, "city,sales\nA,10\nB,20\n").json()
    multi = _upload(
        client,
        tmp_path,
        "epoch,loss,acc\n1,1.0,0.5\n2,0.8,0.6\n3,0.6,0.7\n",
        name="line.csv",
    ).json()

    assert single["chart"]["legend"] == []
    assert multi["chart"]["legend"] == ["loss", "acc"]


def test_bar_labels_every_bar_when_few_categories(client, tmp_path):
    body = _upload(
        client,
        tmp_path,
        "city,sales\nA,10\nB,20\nC,15\nD,12\nE,18\n",
    ).json()

    assert body["chart"]["kind"] == "bar"
    assert body["chart"]["svg"].count('class="value-label"') == 5


def test_bar_labels_only_extremes_when_many_categories(client, tmp_path):
    rows = "\n".join(f"C{index},{index + 1}" for index in range(10))
    body = _upload(client, tmp_path, "city,sales\n" + rows + "\n").json()

    assert body["chart"]["kind"] == "bar"
    assert body["chart"]["svg"].count('class="value-label"') == 2


def test_svg_output_is_deterministic(client, tmp_path):
    text = "epoch,loss\n1,1.0\n2,0.8\n3,0.6\n"
    first = _upload(client, tmp_path, text, name="a.csv").json()["chart"]["svg"]
    second = _upload(client, tmp_path, text, name="b.csv").json()["chart"]["svg"]

    assert first == second, "SVG 不应包含时间戳或随机 id"


def test_svg_css_is_scoped_so_inline_embedding_cannot_leak(client, tmp_path):
    # 内联 <style> 是文档级的；规则必须全部挂在根类下，否则会改坏页面样式
    svg = _upload(
        client, tmp_path, "city,sales\nA,10\nB,20\nC,15\n"
    ).json()["chart"]["svg"]
    assert 'class="dataviz-svg"' in svg
    style = svg.split("<style>", 1)[1].split("</style>", 1)[0]
    for selector in re.findall(r"([^{}]+)\{", style):
        assert ".dataviz-svg" in selector


def test_dark_mode_uses_dark_theme_not_simple_inversion(client, tmp_path):
    text = "city,sales\nA,10\nB,20\nC,15\n"
    light = _upload(client, tmp_path, text, name="light.csv").json()["chart"]["svg"]
    dark = _upload(
        client, tmp_path, text, name="dark.csv", dark=True
    ).json()["chart"]["svg"]

    assert "#ffffff" in light
    assert "#11161d" in dark
    assert light != dark
    ET.fromstring(dark)


# =========================================================== ④ 双 Y 轴


def test_mixed_units_use_small_multiples_not_dual_axis(client, tmp_path):
    text = "year,acc,loss\n2020,90%,1.2\n2021,92%,1.0\n2022,95%,0.8\n"
    body = _upload(client, tmp_path, text).json()

    note = body["chart"]["note"]
    assert "双 Y 轴" in note
    assert "small multiples" in note
    assert body["chart"]["kind"] == "line"

    # 两个分面 = 两个独立纵轴，而不是一条图上的左右双轴
    assert body["exports"]["tikz"].count(r"\begin{axis}") == 2
    assert body["exports"]["mermaid"].count("xychart-beta") == 2
    assert "SMALL_MULTIPLES = True" in body["exports"]["matplotlib"]
    assert _coordinate_blocks(body["exports"]["tikz"]) == [3, 3]


# ===================================================== ⑤ 图注数字校验


def _line_body(client, tmp_path, monkeypatch, payload):
    monkeypatch.setattr("app.llm.client.chat_json", _fake_llm(payload))
    response = _upload(
        client,
        tmp_path,
        "epoch,loss\n1,1.0\n2,0.8\n3,0.6\n",
        caption=True,
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_caption_accepts_numbers_that_exist_in_the_stats_card(client, tmp_path, monkeypatch):
    body = _line_body(
        client,
        tmp_path,
        monkeypatch,
        {
            "caption": "损失从 1.0（第 1 轮）降到 0.6（第 3 轮），均值 0.8，n=3。",
            "key_numbers": [{"value": 0.8, "meaning": "平均损失"}],
        },
    )

    assert body["degraded"] is False
    assert body["caption"]["trusted"] is True
    assert "[未核实]" not in body["caption"]["text"]
    assert body["caption"]["key_numbers"] == [
        {"value": 0.8, "meaning": "平均损失"}
    ]
    assert "数据区间：n=3" in body["caption"]["text"]


def test_caption_scrubs_invented_number(client, tmp_path, monkeypatch):
    body = _line_body(
        client,
        tmp_path,
        monkeypatch,
        {
            "caption": "损失从 1.0 降到 0.6，性能提升 42.7%。",
            "key_numbers": [{"value": 42.7, "meaning": "编造的提升幅度"}],
        },
    )

    assert body["caption"]["trusted"] is True
    assert "42.7" not in body["caption"]["text"]
    assert "[未核实]" in body["caption"]["text"]
    assert body["caption"]["key_numbers"] == [], "编造的关键数字也要剔除"
    assert any("42.7" in note for note in body["notes"])


def test_caption_falls_back_to_template_when_most_numbers_are_invented(client, tmp_path, monkeypatch):
    body = _line_body(
        client,
        tmp_path,
        monkeypatch,
        {
            "caption": "准确率提升 88.8%，损失下降 77.7%，样本量 12345。",
            "key_numbers": [],
        },
    )

    assert body["caption"]["trusted"] is False
    assert "88.8" not in body["caption"]["text"]
    assert body["caption"]["text"].startswith("图 1.")
    assert "数据区间：n=3" in body["caption"]["text"]
    assert any("模板图注" in note for note in body["notes"])


def test_caption_number_matching_handles_percent_and_rounding():
    allowed = {0.5, 12.345, 3.0}

    scrubbed, invented, total = validate_caption(
        "占比 50%，系数 12.3，样本 n=3。", allowed
    )
    assert invented == []
    assert total == 3
    assert scrubbed == "占比 50%，系数 12.3，样本 n=3。"

    scrubbed, invented, total = validate_caption("占比 60%。", allowed)
    assert invented == ["60%"]
    assert "[未核实]" in scrubbed


# ======================================================= ⑥ 模型不可用


def test_llm_outage_degrades_with_template_caption(client, tmp_path, monkeypatch):
    from app.llm.client import LLMError

    def down(*args, **kwargs):
        raise LLMError("缺少环境变量 DEEPSEEK_API_KEY")

    monkeypatch.setattr("app.llm.client.chat_json", down)

    response = _upload(
        client,
        tmp_path,
        "epoch,loss\n1,1.0\n2,0.8\n3,0.6\n",
        caption=True,
    )

    assert response.status_code == 200, "模型不可用不能 500"
    body = response.json()
    assert body["degraded"] is True
    assert body["caption"]["trusted"] is False
    assert body["caption"]["text"].startswith("图 1.")
    assert "数据区间：n=3" in body["caption"]["text"]
    assert any("模型不可用" in note for note in body["notes"])
    # 图和导出仍然完整
    assert body["chart"]["svg"].startswith("<svg")
    assert body["exports"]["tikz"]


def test_caption_disabled_skips_model_entirely(client, tmp_path, monkeypatch):
    called = {"count": 0}

    def explode(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("caption=false 时不应调用模型")

    monkeypatch.setattr("app.llm.client.chat_json", explode)
    body = _upload(client, tmp_path, "city,sales\nA,1\nB,2\n").json()

    assert body["caption"] is None
    assert body["degraded"] is False
    assert called["count"] == 0


# ================================================= ⑦ 导出的点数一致性


def test_line_exports_have_same_point_count_as_svg(client, tmp_path):
    rows = "\n".join(f"{index},{1.0 - index * 0.05:.2f}" for index in range(1, 8))
    text = "epoch,loss\n" + rows + "\n"
    body = _upload(client, tmp_path, text, name="seven.csv").json()

    assert _point_count(body["chart"]["svg"]) == 7
    assert _coordinate_blocks(body["exports"]["tikz"]) == [7]

    mermaid_arrays = re.findall(r"line \[([^\]]*)\]", body["exports"]["mermaid"])
    assert [len(values.split(",")) for values in mermaid_arrays] == [7]

    mpl_arrays = re.findall(r'"values": \[([^\]]*)\]', body["exports"]["matplotlib"])
    assert [len(values.split(",")) for values in mpl_arrays] == [7]


def test_bar_exports_have_same_category_count_as_svg(client, tmp_path):
    body = _upload(client, tmp_path, "city,sales\nA,10\nB,20\nC,15\n").json()

    assert len(re.findall(r'class="series"', body["chart"]["svg"])) == 3
    assert _coordinate_blocks(body["exports"]["tikz"]) == [3]
    assert [
        len(values.split(","))
        for values in re.findall(r"bar \[([^\]]*)\]", body["exports"]["mermaid"])
    ] == [3]
    assert [
        len(values.split(","))
        for values in re.findall(
            r'"values": \[([^\]]*)\]', body["exports"]["matplotlib"]
        )
    ] == [3]


def test_radar_exports_cover_every_axis_value(client, tmp_path):
    body = _upload(
        client,
        tmp_path,
        "algo,precision,recall,f1\nA,0.8,0.7,0.75\nB,0.9,0.6,0.72\nC,0.7,0.9,0.79\n",
        name="radar.csv",
    ).json()

    # 3 个类别 × 3 个指标；每个导出都必须保留 9 个数值
    assert _coordinate_blocks(body["exports"]["tikz"]) == [3, 3, 3]
    assert [
        len(values.split(","))
        for values in re.findall(r"bar \[([^\]]*)\]", body["exports"]["mermaid"])
    ] == [3, 3, 3]
    assert [
        len(values.split(","))
        for values in re.findall(
            r'"values": \[([^\]]*)\]', body["exports"]["matplotlib"]
        )
    ] == [3, 3, 3]


def test_mermaid_pie_style_is_available():
    table = parse_csv(b"city,sales\nA,10\nB,20\nC,15\n")
    chart = build_chart_data(table)
    from app.analysis.export import to_mermaid

    pie = to_mermaid(chart, style="pie")
    assert pie.startswith("pie showData")
    assert pie.count('"A"') == 1 and pie.count('"B"') == 1 and pie.count('"C"') == 1


def test_scatter_exports_and_svg_keep_the_same_points(client, tmp_path):
    body = _upload(
        client,
        tmp_path,
        "height,weight\n170,55\n160,64\n180,72\n175,68\n",
        name="scatter.csv",
        kind="scatter",
    ).json()

    assert _point_count(body["chart"]["svg"]) == 4
    assert _coordinate_blocks(body["exports"]["tikz"]) == [4]
    mermaid_points = re.findall(
        r'^\s+".*": \[', body["exports"]["mermaid"], flags=re.MULTILINE
    )
    assert len(mermaid_points) == 4, "散点图降级为 quadrantChart 后仍是 4 个点"
    assert [
        len(values.split(","))
        for values in re.findall(
            r'"values": \[([^\]]*)\]', body["exports"]["matplotlib"]
        )
    ] == [4]


def test_matplotlib_script_is_syntactically_valid_and_self_contained(client, tmp_path):
    body = _upload(
        client,
        tmp_path,
        "epoch,loss\n1,1.0\n2,0.8\n3,0.6\n",
    ).json()
    script = body["exports"]["matplotlib"]

    compile(script, "<matplotlib>", "exec")
    assert "1.0" in script and "0.6" in script
    assert "read_csv" not in script and "pandas" not in script


# ============================================================ ⑧ 错误码


def test_empty_csv_returns_400(client, tmp_path):
    """空文件是 400，不是 422。

    这条和它的兄弟要一致：二进制文件头、非 UTF-8、列数不齐都是 400 ——
    共同点是"你上传的东西根本不成其为 CSV"。422 留给"是合法 CSV，但做不了图"
    （比如只有表头、没有数值列）。同一类错误给出两种状态码，前端按码分流时就得写两遍。
    """
    response = _upload(client, tmp_path, b"")
    assert response.status_code == 400
    assert "空" in response.json()["detail"]


def test_no_numeric_column_returns_422(client, tmp_path):
    response = _upload(client, tmp_path, "name,color\nA,red\nB,blue\n")
    assert response.status_code == 422
    assert "数值" in response.json()["detail"]


def test_non_csv_magic_returns_400(client, tmp_path):
    response = _upload(client, tmp_path, b"%PDF-1.7 fake pdf", name="fake.csv")
    assert response.status_code == 400
    assert "不是 CSV" in response.json()["detail"]


def test_non_utf8_upload_returns_400(client, tmp_path):
    response = _upload(
        client, tmp_path, b"name,value\n\xff\xfe,2\n", name="gbk.csv"
    )
    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_oversized_csv_returns_413(client, tmp_path):
    payload = b"a,b\n" + b"1,2\n" * 1_600_000  # 约 6.4 MB
    response = _upload(client, tmp_path, payload, name="big.csv")
    assert response.status_code == 413


def test_preview_returns_table_without_chart(client, tmp_path):
    data = _write(tmp_path, "city,sales,rate\nA,10,10%\nB,20,20%\n")
    response = client.post(
        "/api/dataviz/preview",
        files={"file": ("data.csv", data, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "table" in body
    assert "chart" not in body
    assert body["table"]["stats"]["sales"]["max"] == 20.0
    assert body["table"]["columns"][2]["unit"] == "%"


def test_blank_optional_form_fields_fall_back_to_inference(client, tmp_path):
    # 前端常把没选的字段以空字符串提交，不能因此把空 kind 当成非法类型
    response = _upload(
        client,
        tmp_path,
        "month,sales\n1,10\n2,12\n3,15\n",
        kind="",
        x="",
        y="",
    )
    assert response.status_code == 200
    assert response.json()["chart"]["kind"] in {"line", "bar"}


def test_unknown_explicit_kind_returns_422(client, tmp_path):
    response = _upload(client, tmp_path, "city,sales\nA,1\nB,2\n", kind="pie")
    assert response.status_code == 422
    assert "不支持" in response.json()["detail"]


def test_radar_with_too_few_metrics_returns_422(client, tmp_path):
    response = _upload(
        client, tmp_path, "algo,precision\nA,0.8\nB,0.9\n", kind="radar"
    )
    assert response.status_code == 422
    assert "至少需要 3" in response.json()["detail"]
