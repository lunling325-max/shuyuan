# 科研助手 Agent

> 每句话都能点回原文精确位置的版本感知科研 Agent

一句话 → 检索 → 生成 → **逐条核对回原文**：核对得上的结论带页码与坐标框返回，
核对不上的直接拦下并说明原因。不是"让模型答得好"，而是**让模型没法乱答**。

## 快速开始

```bash
cd apps/api
uv sync                                    # 装依赖（走清华镜像）
cp ../../.env.example ../../.env           # 填 DEEPSEEK_API_KEY
uv run uvicorn app.api.main:app --reload --port 8000
```

打开 <http://127.0.0.1:8000/> —— 前端由同一个服务托管，无需另起进程。

- `/docs`　自动生成的 OpenAPI 文档
- 仓库根放一份 PDF 后，在界面点「＋ 上传 PDF」即可解析入库

## 目录

```
apps/api/app/
  parsing/     chunker.py      结构感知切块（段落不切开，标题必开新块，去页眉页脚）
               text_index.py   词级索引，给任意引用文本落回页码 + bbox
  retrieval/   bm25.py         词法检索 + 单复数归并 + RRF 融合
               embeddings.py   BGE 的 ONNX 推理（不引 torch）
               dense.py        向量检索（可选通道，见 E-004）
  rag/         query.py        检索词扩展 + 多路召回
               answer.py       证据 → 结论 → 校验 → 回填坐标
               bilingual.py    逐段中英对照（段数守恒，对不上就降级）
               copilot.py      全库检索 → 编号证据 → 框架 + 真实引用
  guardrails/  citation.py     三级匹配（逐字 / 截断容错 / 模糊），给出 supported/weak/unsupported
  versioning.py               版本谱系：关系判定 + 句级对齐 + 噪声过滤
  versioning_summary.py       把差异翻成中文（模型挂了就如实降级）
  references.py               参考文献解析：数字标记 / 字母标记 / ACL 无标记三种版面
  citegraph.py                引用图谱：版本聚类 → 标题匹配 → 手写 PageRank → 确定性布局
  api/         main.py         FastAPI 路由
               registry.py     文档注册表（内容哈希去重，重启自动恢复）
apps/web/      index.html     单文件前端，直连上面的接口
prototype/     mockup.html    设计稿（对照用，不接数据）
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 存活、文献数、是否配好大模型 |
| GET | `/api/papers` | 文献列表 |
| POST | `/api/papers` | 上传 PDF（multipart），按内容哈希去重 |
| DELETE | `/api/papers/{id}` | 从库里删掉一篇：内存、sidecar、落盘副本一起删，返回 204 |
| GET | `/api/papers/{id}/chunks` | 块列表，含全文与 bbox |
| GET | `/api/papers/{id}/file` | 原始 PDF（可 `#page=N` 直达某页） |
| POST | `/api/papers/{id}/qa` | 问答，返回结论 + 锚点 + 守卫统计 |
| GET | `/api/papers/{id}/bilingual?chunk=N` | 某个块的中英逐段对照；大模型不可用或段数不符时 `degraded=true`，原文照常返回 |
| POST | `/api/copilot/outline` | 论文框架：全库检索 → 编号证据 → 框架；引用由服务端回填，模型编的编号直接丢弃 |
| GET | `/api/papers/{id}/compare/{other}` | 两版差异（谁新谁旧按 PDF 生成日期判）；`?summarize=false` 跳过中文归纳 |
| GET | `/api/graph` | 引用图谱：节点 + 边 + 库内 PageRank + 归一化坐标；`?max_external=N` 控制库外文献数（夹在 0–60） |

问答的 `mode` 有两种：

- `answer`：走大模型，返回带锚点的结论（默认）
- `retrieve`：只返回召回的证据块，**不依赖大模型成功** —— 断网/欠费时仍能演示检索
  （会尽力做检索词扩展，失败就退回原始问题）

## 测试

```bash
cd apps/api && uv run pytest -q     # 174 项
```

覆盖：切块的边界条件、BM25 排序与单复数归并、RRF 去重、守卫的三级匹配、
Q&A 的"编造引用必须被丢弃"、HTTP 层的降级与去重、重启后资料库恢复、
版本谱系的降噪与"投稿戳不许混进正文"、对比接口的新旧顺序与大模型断供降级、
参考文献解析（三种版面 + 年份不许取自 arXiv 号）、PageRank 的方向与权威传递、
版本合并成单个节点、图谱坐标的确定性、中英对照的段数守恒与降级、
框架生成丢弃不存在的证据编号、删除接口的三处同步（内存、sidecar、落盘副本，
少删一处就会"删了又回来"），以及 **541 页教材**这条最难的路：
带页码的隔页页眉、页码引用、Bibliography、只能靠字号保住的一级标题。

## 实验记录

所有"哪个方案更好"的结论都带数字，写在 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)，
包括**被数据推翻的假设**（检索式加权、密集检索融合）。没有数字的取舍不写进去。

## 配置

| 环境变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | 必填，生成与检索词扩展 |
| `DEEPSEEK_BASE_URL` | 可选，默认 `https://api.deepseek.com` |
| `LLM_MODEL` | 可选，默认 `deepseek-chat` |
| `PAPER_STORE_DIR` | 可选，PDF 落盘目录，默认 `data/papers` |

`.env` 已在 `.gitignore` 里，密钥不进仓库。
