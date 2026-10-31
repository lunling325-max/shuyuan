"""让 `from app...` 在任意工作目录下都能导入。

pytest 只把测试文件所在目录塞进 sys.path，`app` 包在上一级，
所以这里显式补一条。放在 apps/api/ 下，对 tests/ 里的所有用例生效。
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
