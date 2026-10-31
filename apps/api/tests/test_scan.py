"""扫描件识别与 OCR 降级的测试。

这里守的是三条底线，全部离线、不下载任何东西：
1. **该认的认出来**：只有图片没有文字层的 PDF 必须被判成扫描件，
   正常的文字 PDF 不能被误判 —— 误判会让好端端的文档去跑昂贵的 OCR。
2. **混合文档要精确到页**：一页文字一页图时，只能把图那页标为 scanned_pages，
   否则上层会把整份文档都当扫描件处理。
3. **没有 OCR 也不能崩**：本机没装 Tesseract 时，`ocr_available` 必须是 False，
   `ocr_pages` 必须返回空字典；损坏文件也必须返回 pages=0 的报告而不是抛异常。
   假装识别成功比明确失败更危险，因为它会把幻觉文本送进检索库。

运行: PYTHONUTF8=1 uv run pytest -q
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.parsing import scan
from app.parsing.scan import analyze, ocr_pages


def _write_text_pdf(path: Path, n_pages: int = 1) -> Path:
    """造一份有真实文字层的 PDF：每页塞足够多的正文。"""
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page()
        body = f"Page {i + 1}. The quick brown fox jumps over the lazy dog. " * 4
        page.insert_text((72, 72), body, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _make_image_only_pdf(path: Path, n_pages: int = 1) -> Path:
    """造一份只有图片、没有文字层的 PDF。

    做法是先生成带文字的页，再把它渲染成 PNG 插入新文档。
    这样新文档里只剩像素，`get_text()` 必然拿不到任何字符 ——
    这正模拟扫描仪输出的扫描件。
    """
    source = pymupdf.open()
    src_page = source.new_page()
    src_page.insert_text((72, 72), "This text must survive only as pixels. " * 8, fontsize=14)
    image = src_page.get_pixmap(dpi=120).tobytes("png")
    source.close()

    doc = pymupdf.open()
    for _ in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=image)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def text_pdf(tmp_path: Path) -> Path:
    return _write_text_pdf(tmp_path / "text.pdf")


@pytest.fixture()
def image_pdf(tmp_path: Path) -> Path:
    return _make_image_only_pdf(tmp_path / "image.pdf")


def test_image_only_pdf_is_detected_as_scanned(image_pdf: Path):
    """没有文字层的一页必须整页判成扫描件，字符数为 0。"""
    report = analyze(str(image_pdf))
    assert report.pages == 1
    assert report.scanned_pages == [1]
    assert report.readable_pages == 0
    assert report.text_chars < scan.MIN_PAGE_TEXT_CHARS
    assert "扫描件" in report.note


def test_all_image_pages_are_reported(tmp_path: Path):
    """多页扫描件必须每一页都被列出，而不是只报第一页。"""
    pdf = _make_image_only_pdf(tmp_path / "scan3.pdf", n_pages=3)
    report = analyze(str(pdf))
    assert report.pages == 3
    assert report.scanned_pages == [1, 2, 3]
    assert report.readable_pages == 0


def test_text_pdf_is_not_misjudged(text_pdf: Path):
    """正常文字 PDF 不能被当成扫描件，否则会平白触发 OCR。"""
    report = analyze(str(text_pdf))
    assert report.pages == 1
    assert report.scanned_pages == []
    assert report.readable_pages == 1
    assert report.text_chars >= scan.MIN_PAGE_TEXT_CHARS
    assert "无需 OCR" in report.note


def test_mixed_pdf_flags_only_the_image_page(tmp_path: Path):
    """一页文字一页图：只有图那页是扫描页。"""
    doc = pymupdf.open()
    text_page = doc.new_page()
    text_page.insert_text((72, 72), "This page has a real digital text layer. " * 4, fontsize=12)

    source = pymupdf.open()
    src_page = source.new_page()
    src_page.insert_text((72, 72), "Rendered to an image only. " * 8, fontsize=14)
    image = src_page.get_pixmap(dpi=120).tobytes("png")
    source.close()

    image_page = doc.new_page(width=595, height=842)
    image_page.insert_image(image_page.rect, stream=image)
    path = tmp_path / "mixed.pdf"
    doc.save(str(path))
    doc.close()

    report = analyze(str(path))
    assert report.pages == 2
    assert report.scanned_pages == [2]
    assert report.readable_pages == 1
    assert "第 2 页" in report.note


def test_ocr_available_is_false_without_tesseract(
    monkeypatch: pytest.MonkeyPatch, image_pdf: Path
):
    """PATH 上找不到 tesseract 时，必须明确报告 OCR 不可用。"""
    monkeypatch.setattr(scan.shutil, "which", lambda _name: None)
    report = analyze(str(image_pdf))
    assert report.ocr_available is False
    assert "Tesseract" in report.note


def test_ocr_pages_returns_empty_dict_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, image_pdf: Path
):
    """OCR 不可用时返回空字典且不抛异常 —— 调用方据此降级，而不是收到 500。"""
    monkeypatch.setattr(scan.shutil, "which", lambda _name: None)
    assert ocr_pages(str(image_pdf)) == {}
    # 显式指定页码也必须一样安全
    assert ocr_pages(str(image_pdf), [1]) == {}


def test_corrupt_pdf_returns_empty_report(tmp_path: Path):
    """损坏文件必须退化成 pages=0 的报告，绝不能把异常抛给 API 层。"""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a valid pdf body\n")

    report = analyze(str(broken))
    assert report.pages == 0
    assert report.readable_pages == 0
    assert report.scanned_pages == []
    assert report.text_chars == 0
    assert report.note  # 必须给人一句可读的解释
    assert "无法打开" in report.note
