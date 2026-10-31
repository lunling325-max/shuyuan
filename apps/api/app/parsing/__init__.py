"""文档解析层：PDF → 结构化文本与坐标。"""

from app.parsing.text_index import Anchor, TextIndex, normalize, split_tokens

__all__ = ["TextIndex", "Anchor", "normalize", "split_tokens"]
