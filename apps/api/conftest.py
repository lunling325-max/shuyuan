"""让 `from app...` 在任意工作目录下都能导入。

pytest 只把测试文件所在目录塞进 sys.path，`app` 包在上一级，
所以这里显式补一条。放在 apps/api/ 下，对 tests/ 里的所有用例生效。
"""

import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 测试里关掉限流：几百个用例共用同一个 TestClient 来源，开着的话跑到一半就开始
# 429，测的就不是业务了。限流本身由 tests/test_ratelimit.py 单独用独立 app 测。
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "0")
