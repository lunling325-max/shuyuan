"""按功能拆分的路由模块。

规矩只有一条：**每个路由模块自带 `router = APIRouter()`，由 main.py 统一 include。**
注册表、404 包装、响应模型组装一律从 `app.api.deps` 取，不要 import main ——
那会绕成环（main 要 include 这些模块）。也不要在路由模块里新建 registry：
第二个实例意味着第二份解析缓存，同一篇文献会被解析两遍。
"""
