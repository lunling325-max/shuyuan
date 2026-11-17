"""公式识别：词元还原规则，以及模型缺失时的降级。

这一层最容易出错的地方不是模型，是**还原**：模型的词元几乎每个字形都带前导 `Ġ`
（空格标记），命令名（mathrm/cdot）不带 —— 粗暴地"删掉所有空格"会把
`\\cdot warmup` 粘成不存在的命令 `\\cdotwarmup`。三条规则在这里钉住。
"""

from __future__ import annotations

import app.parsing.formula_latex as ocr

G = "Ġ"        # Ġ：模型输出里表示"这个词元前有一个空格"
BS = chr(92)        # 反斜杠：用 chr 构造，避免测试文件被多层转义改坏


def test_space_after_command_is_kept():
    """`\\cdot` 后面那一格必须留着，否则命令名会被后面的词吃掉。"""
    pieces = [G + BS, "cdot", G + "w", G + "a", G + "r", G + "m", G + "u", G + "p", G + "x"]
    assert ocr._to_latex(pieces) == BS + "cdot warmupx"


def test_spaces_inside_braces_are_squeezed():
    pieces = [G + BS, "mathrm", G + "{", G + "A", G + "t", G + "t", G + "}"]
    assert ocr._to_latex(pieces) == BS + "mathrm{Att}"


def test_letter_runs_outside_braces_are_joined():
    pieces = [G + "l", G + "r", G + "a", G + "t", G + "e", G + "=", G + "d"]
    assert ocr._to_latex(pieces) == "lrate=d"


def test_nested_braces_are_handled():
    pieces = [G + "d", G + "_", G + "{", G + BS, "mathrm", G + "{", G + "m", G + "}", G + "}"]
    assert ocr._to_latex(pieces) == "d_{" + BS + "mathrm{m}}"


def test_available_is_false_without_model(tmp_path, monkeypatch):
    """模型不在时 `available()` 必须为假 —— 调用方据此返回"不可用"而不是崩。"""
    monkeypatch.setattr(ocr, "MODEL_DIR", tmp_path / "missing")
    assert ocr.available() is False


def test_recognise_raises_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "MODEL_DIR", tmp_path / "missing")
    ocr._session.cache_clear()
    try:
        ocr.recognise(b"not-an-image")
    except ocr.OcrUnavailable:
        pass
    else:                                    # pragma: no cover - 走到这里说明降级失效
        raise AssertionError("模型缺失时应当抛 OcrUnavailable")
    finally:
        ocr._session.cache_clear()
