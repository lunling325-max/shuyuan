"""HTTP 层。业务逻辑一律留在 parsing / retrieval / rag，这里只做协议转换。"""

from app.api.main import app

__all__ = ["app"]
