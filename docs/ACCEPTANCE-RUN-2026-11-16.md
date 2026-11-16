# 线上验收记录 · 2026-11-16

**服务**：http://101.37.190.214:8017/ · 15 篇文献 · 806 页 · 1317 块

> 按 `docs/ACCEPTANCE.md` 在**真实部署**上跑一遍，而不是在本地。
> 本地验过的路径线上可能不同（E-063：本地四项齐全，线上连 541 页教材都没有）。

## 0 素材覆盖

`python scripts/check_testcases.py` → **四项全覆盖**：v1 矢量 · v7 · 扫描件 · 541 页教材（780 块）。

## 1–9 主流程

| 步骤 | 关键证据 |
|---|---|
| 0 打开页面 | 视图数=11 · 文献行数=15 · 首屏标题=15 篇文献 · 共 806 页 · 1317 块索引常驻 · 进程 701 M |
| 1 问答溯源 | 状态行=检索 6 块 · 2183ms · 通过 2 · 待确认 0 · 拦截 0 · 引用按钮数=2 · 带页码=True · 英文引文=True |
| 1b 跳转高亮 | 高亮存在=True |
| 2 图表公式 | 表格数=4 · 正文长度=3598 · 提到公式=True |
| 3 版本谱系 | 面板摘要=Attention Is All You Need / v1 · 2017-06-13 · 15 页 / → / Attention Is All You Need / v7 · 2024-04-10 · 15 页 / 同一篇的不同版本 · · 有改动统计=True |
| 4 跨文献对比 | 可选项数=15 |
| 5 检索 | 含金量按钮数=4 |
| 6 推荐追踪 | 推荐条数=12 · 订阅区为空（**探针跑在订阅创建之前**；创建并抓取后的结果见下节） |
| 7 数据出图 | 图表节点=3484 · 图注=图1为雷达图，展示 layer、bleu_en_de 与 bleu_en_fr 三项指标在 base 与 big 两个模 |
| 9 不该出现的 | 出现「0 被引」且无说明=False · 出现「解析失败」=False |

## 新增三项（本次上线）

- **规模与内存**：15 篇文献 · 共 806 页 · 1317 块索引常驻 · 进程 741 MB
- **摘要块**：可见=True，开头「Transformers are quickly becoming one of the」，展开可用=True
- **架构拓扑图**：稀疏注意力长上下文推理架构 · 5 个节点 / 5 条边；脚本由结构渲染，不依赖模型写代码。；导出=['Graphviz (.dot)', 'Mermaid (.mmd)', 'TikZ (.tex)']

## 订阅追踪（第 6 步）

- query=diffusion model
- 已见 3 条 · 本次新增 0 · 抓回 3
- 状态: ok · 说明: 没有新增（本轮拿回 40 条，都已见过或与关键词无关）
- 下次到点: 2026-11-17T10:59:43.927928+00:00
- 前两条: Restoring Diversity in Distilled I | JVT: Just Video Transformer makes 

## 复现方式

```
python scripts/check_testcases.py                        # 素材覆盖
python X:/tmp/prod_accept.py                             # 1–7 主流程 + 四不该
python X:/tmp/topo_ui.py http://101.37.190.214:8017/      # 架构拓扑图
python X:/tmp/abstract_ui.py http://101.37.190.214:8017/  # 摘要
python X:/tmp/header_probe.py http://101.37.190.214:8017/ # 规模与内存
```

## 已知局限

- 2–3 篇的摘要开头仍带作者/机构串（上标数字与邮箱混排），剥离规则只认已知事实；
- 扫描件的 OCR 文本无标点，作者串与摘要之间没有可靠分界；
- 环境里没有 `SEMANTIC_SCHOLAR_API_KEY` 时，文献检索会明说该源被限流，不静默少一路。
