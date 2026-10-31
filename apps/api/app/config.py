"""配置加载。

极简 .env 读取，不引第三方依赖：正式部署时环境变量由 Docker Compose 注入，
本地开发才需要这个文件。已存在的环境变量**永不覆盖** —— 否则
`DEEPSEEK_API_KEY=xxx uvicorn ...` 这种临时覆盖会被 .env 悄悄顶掉。
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
