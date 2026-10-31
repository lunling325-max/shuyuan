"""扫描件识别与 OCR 降级：先判断「有没有文字层」，再决定能不能识别。

为什么需要这个模块？
比赛要求用一张扫描版论文（1706.03762v1_img.pdf）验证「OCR 识别与结构化还原」。
扫描件里每个字都是像素，`page.get_text()` 只能拿到空串，原来的解析链会
一路走到空结果然后 422 —— 用户看到的是「解析失败」，却不知道失败的原因是
「这份 PDF 没有文字层」，也不知道本机压根没装 OCR 引擎。

本模块把这件事拆成两步，绝不假装成功：
1. `analyze()`：逐页统计文字层字符数，明确指出哪些页是扫描件；
   即使文件损坏也只返回报告，不把异常抛给 API 层。
2. `ocr_pages()`：OCR 引擎真的可用时才动手；单页失败就跳过继续，
   扫描件常有坏页，不能因为一页让整份文档失败；引擎不可用时返回空字典，
   由调用方把「无法识别」原样告诉用户。

判定口径：`get_text("text")` 去掉所有空白后不足 30 个字符的页算扫描页。
30 是经验阈值：真实文字页的正文远大于它，而扫描页通常只剩页码或几个
看不见的控制字符。
"""

from __future__ import annotations

import functools
import shutil
from dataclasses import dataclass

import pymupdf

# 一页文字层去掉空白后少于这么多字符，就认为它没有可用的文字层。
MIN_PAGE_TEXT_CHARS = 30
# OCR 的分辨率：200 dpi 在清晰度与耗时之间比较平衡，正文小字也能认出来。
DEFAULT_DPI = 200


@dataclass(slots=True)
class ScanReport:
    """一份 PDF 的文字层体检报告。"""

    pages: int                 # 总页数
    readable_pages: int        # 有可用文字层的页数
    scanned_pages: list[int]   # 判定为扫描件的页码（1 起）
    text_chars: int            # 文字层总字符数（去空白后）
    ocr_available: bool        # 本机能不能真的 OCR
    note: str                  # 给人看的一句中文说明


def _visible_chars(text: str) -> int:
    """文字层去掉空白后的字符数。

    只看非空白字符，因为扫描件里偶尔也会有一两个空格或换行符；
    按原始长度统计会把「空页」误判成「有文字」。
    """
    return len("".join(text.split()))


def _ocr_available() -> bool:
    """本机能否真的执行 OCR。

    PyMuPDF 自己不做识别，它只是把页面交给外部 Tesseract 可执行文件。
    所以两个条件缺一不可：
    1. PATH 上能找到 `tesseract`；
    2. PyMuPDF 能找到 Tesseract 的语言数据（tessdata）。

    先做一次 `shutil.which` 快速否定：没有可执行文件就不必启动子进程。
    只有 which 命中时才让 PyMuPDF 探测一次 tessdata —— 这比每次分析都
    真的跑一遍 OCR 便宜得多，同时也能挡住「有二进制但没有语言包」的机器。
    """
    if shutil.which("tesseract") is None:
        return False
    try:
        pymupdf.get_tessdata()
    except Exception:
        return False
    return True


def _short(error: Exception) -> str:
    """把异常压成一行，避免把多行堆栈塞进给人看的 note。"""
    return " ".join(str(error).split())[:160]


def _page_list(pages: list[int]) -> str:
    """把页码压成区间，如 `1、3-5、9`，避免长文档的说明过长。"""
    if not pages:
        return ""
    parts: list[str] = []
    start = prev = pages[0]
    for page in pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = page
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return "、".join(parts)


def _make_note(
    pages: int,
    readable_pages: int,
    scanned_pages: list[int],
    text_chars: int,
    ocr_available: bool,
) -> str:
    """生成一句中文说明，讲清「是什么情况」和「能不能 OCR」。"""
    if pages == 0:
        return "PDF 没有任何页面，无法判断文字层。"
    if not scanned_pages:
        return f"共 {pages} 页，全部有文字层（{text_chars} 个字符），无需 OCR。"
    if len(scanned_pages) == pages:
        head = f"共 {pages} 页，全部缺少文字层，判定为纯扫描件"
    else:
        head = (
            f"共 {pages} 页，其中 {len(scanned_pages)} 页缺少文字层"
            f"（第 {_page_list(scanned_pages)} 页），按扫描件处理"
        )
    if ocr_available:
        tail = "本机 OCR 可用，可调用 ocr_pages 识别。"
    else:
        tail = "本机未安装 Tesseract，无法 OCR，只能标记为扫描件并说明原因。"
    return f"{head}；{tail}"


def analyze(path: str) -> ScanReport:
    """逐页判断文字层，返回扫描件体检报告。

    这里刻意**不抛异常**：API 层拿到的应该是一份可解释的报告，
    而不是 500。打不开、加密、坏页都退化成「没有可用文字层」的描述。
    """
    ocr_ok = _ocr_available()
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        # 文件不存在、不是 PDF、头部损坏都会走到这里。
        return ScanReport(
            pages=0,
            readable_pages=0,
            scanned_pages=[],
            text_chars=0,
            ocr_available=ocr_ok,
            note=f"无法打开文件，按非 PDF 或已损坏处理：{_short(exc)}",
        )

    try:
        pages = doc.page_count
        scanned: list[int] = []
        text_chars = 0
        for page_no in range(1, pages + 1):
            try:
                text = doc[page_no - 1].get_text("text")
            except Exception:
                # 单页抽不出文字（坏页、加密）比整份失败好：按扫描页记，
                # 后续 OCR 也失败时上层还能看到「这一页没有文字层」。
                text = ""
            chars = _visible_chars(text)
            text_chars += chars
            if chars < MIN_PAGE_TEXT_CHARS:
                scanned.append(page_no)
        readable_pages = pages - len(scanned)
    finally:
        doc.close()

    return ScanReport(
        pages=pages,
        readable_pages=readable_pages,
        scanned_pages=scanned,
        text_chars=text_chars,
        ocr_available=ocr_ok,
        note=_make_note(pages, readable_pages, scanned, text_chars, ocr_ok),
    )


def ocr_pages(
    path: str,
    pages: list[int] | None = None,
    *,
    dpi: int = DEFAULT_DPI,
) -> dict[int, str]:
    """对指定页做 OCR，返回 {页码: 识别文本}。

    `pages=None` 表示整份文档。OCR 引擎不可用、文件打不开、单页识别失败，
    都只让结果字典少几项，**绝不抛异常** —— 调用方可以据此明确告诉用户
    「能识别哪几页、哪几页没识别出来」，而不是假装整份成功。

    注意 PyMuPDF 2.0 的签名是
    `page.get_textpage_ocr(flags=0, language="eng", dpi=72, full=False, tessdata=None)`，
    这里按需传 `dpi`，再用 `page.get_text(textpage=...)` 取文本。
    """
    if not _ocr_available():
        return {}

    try:
        doc = pymupdf.open(path)
    except Exception:
        return {}

    try:
        total = doc.page_count
        targets = list(range(1, total + 1)) if pages is None else pages
        result: dict[int, str] = {}
        for page_no in targets:
            if not (1 <= page_no <= total):
                continue  # 越界页码直接跳过，不为一页废掉整批
            try:
                page = doc[page_no - 1]
                textpage = page.get_textpage_ocr(dpi=dpi)
                result[page_no] = page.get_text(textpage=textpage)
            except Exception:
                # 扫描件里混着坏页很常见；跳过它，继续下一页。
                continue
        return result
    finally:
        doc.close()


@functools.lru_cache(maxsize=1)
def ocr_enabled() -> bool:
    """`_ocr_available` 的缓存版。切块时会按页问，541 页不能每页都探一次子进程。"""
    return _ocr_available()


def ocr_blocks_for_page(page: "pymupdf.Page", *, dpi: int = DEFAULT_DPI) -> dict | None:
    """把一页交给 OCR，返回 `get_text("dict")` 那种结构（带 bbox 与字号）。

    为什么不直接返回字符串：切块器要靠 `size` / `bold` 认标题，靠 `bbox` 画高亮。
    返回纯文本等于把扫描件降级成"一整页一大块"，目录树和溯源框全没了。
    `get_textpage_ocr` 出来的 textpage 恰好支持 `get_text("dict", textpage=…)`。

    返回 None 表示"这一页 OCR 不了"（引擎缺失 / 单页失败），
    调用方应当退回"没有正文"的语义，而不是塞进空字符串。
    """
    if not ocr_enabled():
        return None
    try:
        tp = page.get_textpage_ocr(dpi=dpi)
        return page.get_text("dict", textpage=tp)
    except Exception:  # noqa: BLE001 —— 扫描页千奇百怪，单页失败不该拖垮整份文档
        return None
