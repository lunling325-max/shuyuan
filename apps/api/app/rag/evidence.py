"""什么样的块不能当证据。

两条都是"能引用、但引了反而不可信"的块：

- **引文列表**：块里是 `[12] Vaswani et al., Attention is all you need.` 这种
  参考文献行。拿它当"作者原话"，结论的证据其实来自**另一篇**论文 ——
  实测写作 Copilot 就引到过 FSA 第 12 页的一条书目，正文写的是别人的工作。
- **前置页**：标题、作者、版权声明。它确实是这篇论文的原文，但作为论据没有
  任何信息量，列十条等于什么都没说。

放在独立模块里：综述和写作 Copilot 是两个检索入口，过滤规则必须完全一致 ——
以前只有综述过滤引文列表，Copilot 没过滤，同一句话在两边一个能引一个不能引。
"""

from __future__ import annotations

import re

REF_LIST_MARKERS = ("references", "bibliography", "参考文献", "reference list")


def is_reference_list(chunk) -> bool:
    """参考文献 / 附录引文串：能引用，但作为论据没有任何信息量。

    有章节名就认名字；没有章节名时看形态 —— 一块里塞了 6 个以上的 `[12]`，
    基本只可能是引文列表。判据放宽一点没关系：漏掉一块引文只是浪费一次召回，
    而把引文列表当证据，会让整篇综述的可信度崩塌。
    """
    section = (getattr(chunk, "section", "") or "").lower()
    if any(marker in section for marker in REF_LIST_MARKERS):
        return True
    return len(re.findall(r"\[\d{1,3}\]", getattr(chunk, "text", "") or "")) >= 6


def is_front_matter(chunk) -> bool:
    """第 1 页的标题/作者/版权块：能引用，但作为证据毫无信息量。

    判据刻意保守 —— 只要不是"整块都是人名和版权声明"就放行：
    正文块通常有句号、有较多 token；前置块往往以作者名开头、几乎不含句号。
    """
    text = chunk.text.strip()
    if getattr(chunk, "n_tokens", 0) >= 120:
        return False
    if text.count(".") >= 3:
        return False
    low = text.lower()
    return getattr(chunk, "page_start", 1) == 1 or "copyright" in low or "@" in text


def body_chunks(chunks, limit: int) -> list:
    """第一篇"像正文"的块：顺序不变，跳过引文列表和前置页。"""
    body = [c for c in chunks if not is_front_matter(c) and not is_reference_list(c)]
    return (body or list(chunks))[:limit]
