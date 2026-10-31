# 项目计划 · 科研助手 Agent（智领科研·意图驱动）

> 省赛作品提交截止：**2026-11-18**　|　决策记录日期：2026-10-31　|　剩余：18 天

## 1. 一句话定位

> 不是又一个文献 Chatbot，而是**每句话都能点回原文精确位置的版本感知科研 Agent**。

三个差异化抓手（全部对应评分表明码格子）：

| 抓手 | 技术实现 | 命中评分项 |
|---|---|---|
| 溯源锚点 | PyMuPDF 取词坐标 → chunk 存 (page, bbox) → 前端 PDF.js 高亮框 | 抗幻觉 8% |
| 版本谱系 | 内容哈希聚类 + LLM 差异摘要，而非简单去重 | RAG 创新 8% + 官方用例 3.1 |
| 基石挖掘 | NetworkX + PageRank/Centrality + 时间线过滤 Future Work | 图谱算法 10% |

## 2. 范围决策：深度优先 4 个模块

| 模块 | 深度 | 理由 | 目标天 |
|---|---|---|---|
| 一 · 解析与去重 | **做穿** | 解析 7% + 版本创新 8% + 官方用例 | D1-2, D6-8 |
| 二 · 抗幻觉 RAG | **做穿** | 溯源 8% + 混合检索，全项目地基 | D3-5 |
| 三 · 联动阅读翻译 | **做穿** | 交互 3% + UI 3%，视频里最炸 | D12-13 |
| 五 · 引用图谱综述 | **做穿** | 图谱 10%，全表单格最大 | D9-11 |
| 四 · 检索追踪推荐 | **最小可演示** | 无对应高分格 | D14 |
| 六 · 写作与可视化 | **最小可演示** | 只做 Mermaid + Matplotlib，放弃 TikZ | D14-15 |

**明确不做**：多用户推荐算法、原位排版 PDF 翻译（改为 Markdown 双语对照）、TikZ 渲染。

## 3. 技术架构

```
Next.js 14 (TS) ── 前端：PDF.js 阅读器 / 图谱 / Copilot
      │  REST + SSE
      ▼
FastAPI ── LangGraph 编排 ──┬── 解析层：PyMuPDF + PaddleOCR
                            ├── 检索层：BGE-M3(dense+sparse) → Qdrant → bge-reranker
                            ├── 图论层：NetworkX
                            └── 生成层：DeepSeek-V3
      │
PostgreSQL（元数据 + 任务队列 + 版本谱系）  Qdrant（向量）
```

- 前端：Next.js 14 App Router · TypeScript · Tailwind · shadcn/ui · PDF.js · Cytoscape.js · KaTeX
- 后端：FastAPI · LangGraph · Qdrant · PostgreSQL · PyMuPDF · PaddleOCR
- 模型：DeepSeek-V3（生成）· SiliconFlow 托管 BGE-M3 + bge-reranker-v2-m3（向量/重排）
- 部署：单机 4C8G 轻量云 + Docker Compose，IP 直连（规避域名备案）

## 4. 核心数据模型

| 实体 | 关键字段 |
|---|---|
| `Paper` | id, arxiv_id, version, title, authors, abstract, sha256, content_hash, version_group_id, parse_status |
| `Page` | paper_id, page_no, width, height, layout_json, text |
| `Chunk` | paper_id, page_no, **bbox**, section_path, text, token_count |
| `Citation` | source_paper_id, target_ref_raw, target_paper_id?, context, is_filler |
| `VersionEdge` | from_version, to_version, diff_summary |
| `Answer` | question, text, citations[] |
| `Job` | type, status, progress, error（异步解析/OCR） |

## 5. 里程碑

| 阶段 | 天 | 交付 | 验收 |
|---|---|---|---|
| 骨架 | D1-2 | 仓库 + Docker Compose + 上传 + 解析 + PDF.js 阅读器 | 能传 v7 并在阅读器里翻页 |
| RAG 核心 | D3-5 | 切块 + BGE-M3 + Qdrant hybrid + 溯源高亮 + 单篇问答 | 问"β1 是多少"能答且跳出原文框 |
| 硬骨头 | D6-8 | 版本谱系 + 扫描件 OCR + 541 页长书 | 官方 3.1/3.2 三个用例全过 |
| 图谱 | D9-11 | 引用抽取 + PageRank + Cytoscape + 综述/Future Work | 20 篇文献出图并标出基石 |
| 阅读体验 | D12-13 | 中英双栏同步滚动 + 引用双向跳转 | 拖动联动误差 < 50ms |
| 补全 | D14-15 | 写作 Copilot + CSV→图表→图注 + arXiv 追踪 | 六大模块均可演示 |
| 收尾 | D16 | 部署 + 压测 + `.env.example` | 公网可访问 |
| 交付 | D17-18 | 技术文档 ≤30 页 + 5-8 分钟视频 | 提交 |

## 6. 风险登记

| 风险 | 影响 | 对策 |
|---|---|---|
| 541 页书一次性塞上下文 → OOM | 直接丢官方用例分 | 分页增量索引 + 异步 Job + 进度轮询，D6 前不写单测就得先写这个 |
| 扫描件 OCR 慢，阻塞请求 | 演示卡死 | PaddleOCR 走独立 Worker 队列，前端只轮询进度 |
| DeepSeek 调用成本与限流 | demo 时断供 | 全链路结果缓存（问题 + 文档版本为 key）+ 预置演示脚本 |
| Docker 环境缺失（本机未装） | 部署项 3% 归零 | **D1 就装 Docker Desktop**，不要拖到最后一周 |
| API key 泄漏进公开仓库 | 违规取消资格 | 第一天建 `.env` + `.env.example`，`.gitignore` 白名单化 |

## 7. 环境现状

- Node 24.16 / npm 11.13 ✓（pnpm 未装，用 corepack 或 npm）
- Python 3.11.0 / uv 0.12.2 ✓
- Git 2.54 ✓
- **Docker ✗ 未安装** —— 阻塞项
- GPU：RTX 5060 Laptop 8GB（BGE-M3 可本地跑，作为 API 成本兜底方案）

## 8. 待办（非代码）

- [ ] 获取官方测试集：`1706.03762v1.pdf`、`1706.03762v7.pdf`、`1706.03762v1_img.pdf`、`UnderstandingDeepLearning_02_09_26_C.pdf`（541 页）
- [ ] DeepSeek + SiliconFlow API Key
- [ ] 确认参赛组别（研究生组 / A / B / C）
- [ ] 选定公网部署主机
