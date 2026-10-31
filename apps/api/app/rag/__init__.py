"""RAG 问答层：检索 → 生成 → 引用校验 → 回填坐标。"""

from app.rag.answer import Answer, answer_question, build_evidence

__all__ = ["Answer", "answer_question", "build_evidence"]
