"""数据分析子包：CSV 解析统计、服务端 SVG 图表、图表源码导出。

为什么单独成包而不是塞进 `app.api`：
解析与统计不依赖 HTTP，也不依赖任何模型，可以直接单元测试。路由层只做
"字节 → Table → ChartData → 响应"的编排，出错时转换成合适的 HTTP 状态码。
这样同一份 `Table` 可以同时喂给 SVG、TikZ 和 Matplotlib 导出，保证三者
用的是同一批清洗后的数据。
"""

from __future__ import annotations
