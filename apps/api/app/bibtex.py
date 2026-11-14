"""把解析出来的参考文献导出成 BibTeX。

BibTeX 是科研写作里的"最小可用交付物"：拿到 `.bib` 就能丢进 Zotero / Overleaf /
EndNote，不用一条条手抄。所以我们不做到"完美还原"，只保证两件事：

1. **能编译进文档**：字段与转义都合法，`bibtex` 不会因为它报错；
2. **缺什么说什么**：解析不出标题的条目**照样导出**，但在 `note` 里写明
   "本条目由 PDF 版面解析而来，字段可能不全"，而不是悄悄丢掉 —— 一份
   "少了 12 条"的 .bib 比一份"有 12 条不完整"的更难发现。

条目类型是**猜的**，规则写在 `_entry_type()` 里：会议关键词 → `@inproceedings`，
期刊关键词 → `@article`，其余 `@misc`。猜错的代价很小（BibTeX 只按类型选样式），
但我们要在注释里说清"类型是猜的"，免得有人拿它当权威。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# BibTeX 里有特殊含义的字符。花括号不转义，反而会被当成"保护大小写"的语法用，
# 所以正文里出现时也要转义掉。
_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

_CONFERENCE = re.compile(
    r"conference|proceedings|workshop|symposium|congress|"
    r"neurips|nips|icml|iclr|acl|emnlp|naacl|cvpr|iccv|eccv|aaai|ijcai|kdd|sigir",
    re.IGNORECASE,
)
_JOURNAL = re.compile(
    r"journal|transactions|letters|review|nature|science|cell|arxiv preprint",
    re.IGNORECASE,
)


def escape(text: str) -> str:
    """转义 BibTeX 特殊字符。顺序要紧：反斜杠必须先处理，否则会二次转义。"""
    out = []
    for char in text or "":
        out.append(_ESCAPE.get(char, char))
    return "".join(out)


def _ascii(text: str) -> str:
    """取姓氏/词根用的 ASCII 化：`Vaswani` 保留，`Çelik` → `Celik`，中文原样留下。"""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _surname(authors: str) -> str:
    """取第一个作者的姓。

    两种排版都要认，判据是**第一个逗号之前有几个词**：
    - `Ashish Vaswani, Noam Shazeer`（名在前）→ 逗号前是两个词 → 姓是最后一个词；
    - `Vaswani, Ashish; Shazeer, Noam`（姓在前）→ 逗号前只有一个词 → 姓就是它；
    - 完全没有逗号（`Tom B. Brown and others`）→ 取最后一个词。

    先按 ` and ` / `;` 切出第一个作者，否则"名在前"的写法整串都是逗号，
    会被误判成"姓在前"。猜错只影响引用键，不影响字段内容。
    """
    text = authors or ""
    first = re.split(r"\s+and\s+|;", text)[0]
    head = first.split(",")[0]
    words = [w for w in re.findall(r"[^\W\d_]+", _ascii(head)) if len(w) > 1]
    if not words:
        return ""
    return words[0] if len(words) == 1 else words[-1]


def _key(ref, used: set[str]) -> str:
    """生成引用键：`姓 + 年 + 标题第一个实词`，重复时加 a/b/c。"""
    parts = [_surname(ref.authors), re.sub(r"\D", "", ref.year or "")]
    title_words = [
        w for w in re.findall(r"[^\W\d_]+", _ascii(ref.title or "")) if len(w) > 3
    ]
    if title_words:
        parts.append(title_words[0])
    base = "".join(p for p in parts if p).lower() or f"ref{ref.key}"
    base = re.sub(r"[^a-z0-9]", "", base) or f"ref{ref.key}"

    candidate = base
    suffix = 0
    while candidate in used:
        suffix += 1
        candidate = f"{base}{chr(ord('a') + (suffix - 1) % 26)}" if suffix < 27 else f"{base}{suffix}"
    used.add(candidate)
    return candidate


def _entry_type(ref) -> str:
    """猜条目类型。会议 → inproceedings，期刊 → article，其余 misc。"""
    venue = ref.venue or ""
    if _CONFERENCE.search(venue):
        return "inproceedings"
    if _JOURNAL.search(venue):
        return "article"
    return "misc"


def _fields(ref) -> list[tuple[str, str]]:
    """按类型挑字段名：期刊写 `journal`，会议写 `booktitle`，其余写 `howpublished`。"""
    kind = _entry_type(ref)
    fields: list[tuple[str, str]] = []
    if ref.authors:
        fields.append(("author", escape(ref.authors)))
    if ref.title:
        fields.append(("title", escape(ref.title)))
    if ref.year:
        fields.append(("year", escape(ref.year)))
    if ref.venue:
        name = {"article": "journal", "inproceedings": "booktitle"}.get(kind, "howpublished")
        fields.append((name, escape(ref.venue)))
    if ref.arxiv_id:
        fields.append(("eprint", escape(ref.arxiv_id)))
        fields.append(("archiveprefix", "arXiv"))
    if not ref.is_parsed:
        # 没解析出标题的条目照样导出：丢掉它，这份 .bib 就少了一条而没人知道
        note = "由 PDF 版面解析而来，字段可能不完整；原文：" + (ref.raw or "")[:200]
        fields.append(("note", escape(note)))
    return fields


def to_bibtex(references: Iterable, *, source: str = "") -> str:
    """把一组参考文献转成 .bib 文本。

    `source` 会写进文件头注释（哪个 PDF 来的），方便几个人各自导出后合并。
    """
    refs = list(references)
    lines = ["% 由「溯源 · 科研助手」从 PDF 版面解析生成。", "% 条目类型与引用键是程序猜的，字段以原文为准。"]
    if source:
        lines.append(f"% 来源：{source}")
    lines.append("")

    used: set[str] = set()
    rendered = 0
    for ref in refs:
        fields = _fields(ref)
        if not fields:
            continue
        lines.append(f"@{_entry_type(ref)}{{{_key(ref, used)},")
        for name, value in fields:
            lines.append(f"  {name} = {{{value}}},")
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
        lines.append("}")
        lines.append("")
        rendered += 1

    if rendered == 0:
        lines.append("% 这份 PDF 里没有解析出可导出的参考文献条目。")
    else:
        lines.append(f"% 共 {rendered} 条（原文解析出 {len(refs)} 条，缺字段的已用 note 标出）。")
    return "\n".join(lines) + "\n"
