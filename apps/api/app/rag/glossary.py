"""离线跨语言检索：内置术语表（glossary）。

**要解决的问题**：BM25 对中文切出 0 个 token（§4.1），于是中文提问能召回什么，
完全取决于一次大模型改写（`app.rag.query.expand_query`）。大模型一挂、限流、
或者没配 key，中文提问就是 **0 召回** —— 这是整条链路里最脆的一环，
偏偏也是演示当天最可能发生的事。

**做法**：一张"中文术语 → 英文检索式"的表，在问题里**最长匹配**切出术语，
拼成英文检索式交给 BM25。不联网、不调模型、毫秒级。

**为什么是手写表，不是翻译模型**：
1. 生产镜像里没有 GPU，也没有下载模型的预算（2C4G 的机器装不下一个翻译模型）；
2. 术语是**封闭集合** —— 这个领域的词就那几百个，覆盖率可以逐条核对；
3. 这张表**错了能查**：命中了哪些词可以直接摆给用户看，大模型改写做不到这一点。

**边界（诚实声明）**：表覆盖的是论文里的名词性术语，不覆盖口语化的动词短语
（"怎么接的"、"能看出什么"）。所以它不追求"把中文翻成英文句子"，
只追求"捞出足够的英文锚点词"，剩下的交给 BM25 的词频统计。
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- 术语表
#
# 键是中文（最长匹配用），值是空格分隔的英文检索词。
# 英文侧刻意保留两种形态：单复数、连字符写法 —— BM25 侧虽然有 `stem()`
# 做单数归并，但 `multi-head` 与 `multihead` 这种写法差异归并不了。
#
# 收录原则：**只收名词性术语**。动词（影响、改变、是什么）一律不收 ——
# 它们在任何段落里都高频，收进来只会把 IDF 压低、把检索式稀释。

TERMS: dict[str, str] = {
    # ---- 模型结构
    "位置编码": "positional encoding sine sinusoidal",
    "正弦函数": "sine function sinusoidal sinusoid",
    "编码器": "encoder",
    "解码器": "decoder",
    "残差连接": "residual connection",
    "层归一化": "layer normalization layernorm",
    "归一化": "normalization normalize",
    "前馈网络": "feed-forward network feedforward",
    "前馈": "feed-forward feedforward",
    "子层": "sub-layer sublayer",
    "架构": "architecture",
    "模型架构": "model architecture",
    "参数量": "number of parameters parameter count",
    "参数": "parameters parametric",
    "隐藏层": "hidden layer",
    "隐藏维度": "hidden dimension d_model",
    "维度": "dimension dimensionality d_model",
    "词嵌入": "word embedding",
    "嵌入": "embedding embed",
    "词表大小": "vocabulary size",
    "词表": "vocabulary",
    "词元": "token subword",
    "令牌": "token",
    "分词器": "tokenizer",
    "分词": "tokenization segmentation",
    "字节对编码": "byte-pair encoding BPE",
    # 中文问"多少/几个"的短语：论文表格里的列名恰恰是 `n_layers`、
    # `number of parameters`——短语直译过去比单个名词命中率高得多。
    "多少个注意力头": "attention heads number of heads",
    "几个头": "attention heads number of heads",
    "多少层": "number of layers n_layers layers",
    "几层": "number of layers n_layers layers",
    "多少个参数": "number of parameters",
    "多少参数": "number of parameters",
    "层数": "number of layers",
    "层": "layer layers stacked",
    "堆": "stack stacked layers",
    "堆叠": "stack stacked",

    # ---- 注意力
    "自注意力": "self-attention",
    "多头注意力": "multi-head attention",
    "多头": "multi-head multihead heads",
    "注意力头数": "attention heads",
    "头数": "heads",
    "注意力机制": "attention mechanism",
    "注意力权重": "attention weights",
    "注意力可视化": "attention visualizations",
    "注意力": "attention",
    "缩放点积注意力": "scaled dot-product attention",
    "点积": "dot-product dot product",
    "加性注意力": "additive attention",
    "稀疏注意力": "sparse attention",
    "滑动窗口注意力": "sliding window attention",
    "掩码": "mask masking masked",
    "键值缓存": "KV cache key-value cache",
    "键值": "key-value",

    # ---- 训练
    "预训练任务": "pre-training objectives",
    "预训练": "pre-training pretraining",
    "微调": "fine-tuning finetuning fine-tune",
    "训练集": "training set training data",
    "训练": "training train",
    "学习率": "learning rate",
    "预热": "warmup warm-up",
    "批大小": "batch size",
    "批次": "batch",
    "轮次": "epoch",
    "梯度裁剪": "gradient clipping",
    "梯度": "gradient",
    "优化器": "optimizer",
    "损失函数": "loss function",
    "损失": "loss",
    "交叉熵": "cross-entropy",
    "正则化": "regularization",
    "丢弃": "dropout",
    "标签平滑": "label smoothing",
    "对抗训练": "adversarial training",
    "数据增强": "data augmentation",
    "蒸馏": "distillation",
    "量化": "quantization quantize",
    "剪枝": "pruning prune",
    "参数高效微调": "parameter-efficient fine-tuning PEFT",

    # ---- 效率与系统
    "显存": "GPU memory memory footprint",
    "内存": "memory",
    "吞吐": "throughput",
    "延迟": "latency",
    "加速比": "speedup",
    "加速": "speedup accelerate",
    "并行": "parallel parallelism",
    "计算复杂度": "computational complexity",
    "复杂度": "complexity",
    "矩阵乘法": "matrix multiplication matmul",
    "浮点运算": "FLOPs floating point operations",
    "带宽": "bandwidth",
    "算子": "operator kernel",
    "内核": "kernel",
    "张量": "tensor",
    "卷积": "convolution",
    "芯片": "chip GPU",
    "显卡": "GPU",

    # ---- 评测
    "数据集": "dataset corpora corpus",
    "基准": "benchmark",
    "消融实验": "ablation study",
    "消融": "ablation",
    "基线": "baseline",
    "准确率": "accuracy",
    "召回率": "recall",
    "精确率": "precision",
    "困惑度": "perplexity",
    "翻译": "translation translate",
    "英德": "English-German German",
    "英法": "English-French French",
    "英中": "English-Chinese Chinese",
    "双语": "bilingual",
    "零样本": "zero-shot",
    "少样本": "few-shot",
    "泛化": "generalization",
    "鲁棒性": "robustness",
    "超参数": "hyperparameter",
    "收敛": "convergence",

    # ---- 大模型与应用
    "大语言模型": "large language model",
    "大模型": "large language model",
    "语言模型": "language model",
    "检索增强生成": "retrieval-augmented generation RAG",
    "检索增强": "retrieval-augmented",
    "向量检索": "dense retrieval vector retrieval",
    "检索": "retrieval retrieve search",
    "嵌入模型": "embedding model",
    "相似度": "similarity",
    "幻觉": "hallucination",
    "提示词": "prompt",
    "提示": "prompt",
    "上下文长度": "context length",
    "长上下文": "long context",
    "上下文": "context",
    "推理": "inference reasoning",
    "智能体": "agent",
    "工具调用": "function calling tool call tools",
    "函数调用": "function calling",
    "多模态": "multimodal multi-modal",
    "扫描件": "scanned document OCR",
    "表格": "table tables",
    "插图": "figure figures",
    "公式": "equation formula",
    "参考文献": "references bibliography",
    "引用": "citation reference",

    # ---- 论文章节
    "摘要": "abstract",
    "引言": "introduction",
    "相关工作": "related work",
    "方法": "method approach technique",
    "实验结果": "experimental results",
    "实验": "experiment experiments experimental",
    "结论": "conclusion",
    "讨论": "discussion",
    "局限": "limitation limitations",
    "贡献": "contribution",
}

# 按长度倒序，保证"多头注意力"先于"注意力"命中。
_KEYS = sorted(TERMS, key=len, reverse=True)
_CJK = re.compile(r"[一-鿿]")
# 术语表里不会有的通用字，出现时不参与匹配统计，只用于判断"问题是不是中文"。
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


def is_chinese(text: str) -> bool:
    """问题里有没有汉字。纯英文问题不该走这张表 —— 它本来就是 BM25 的强项。"""
    return bool(_CJK.search(text))


def extract(question: str) -> list[tuple[str, str]]:
    """最长匹配切出术语，返回 `[(中文, 英文), ...]`，按在问题里出现的顺序。

    为什么是"最长匹配"而不是分词：中文没有空格，而术语表里的键本身就是
    词典条目 —— `多头注意力` 必须先于 `多头` 与 `注意力` 命中，否则
    "多头注意力用了几个头" 会切出三个互相稀释的词。
    """
    out: list[tuple[str, str]] = []
    i = 0
    n = len(question)
    while i < n:
        for key in _KEYS:
            if question.startswith(key, i):
                entry = (key, TERMS[key])
                if entry not in out:  # 同一个术语重复出现只收一次
                    out.append(entry)
                i += len(key)
                break
        else:
            i += 1
    return out


def expand_offline(question: str) -> list[str]:
    """把中文问题变成英文检索式；一个术语都没命中时返回空列表。

    返回空列表就是"这条路走不通"，由调用方决定退回原问题还是报错 ——
    这里不替调用方做决定。

    英文侧保留问题里本来就有的拉丁词（`sqrt(dk)`、`label smoothing`、
    `Pdrop`），它们是**最强的锚点**：精确词形匹配正是 BM25 优于向量检索的地方。
    """
    if not is_chinese(question):
        return []

    terms = extract(question)
    if not terms:
        return []

    english: list[str] = []
    for _, gloss in terms:
        for word in gloss.split():
            if word not in english:
                english.append(word)
    for word in _LATIN_RUN.findall(question):
        if word not in english:
            english.append(word)

    queries = [" ".join(english)]
    # 再给 1 条"只看头两个术语"的检索式：术语一多，整串会互相干扰，
    # 少而精的一条往往把正确小节顶上去，RRF 把两路合起来。
    #
    # 试过**每个术语各出一条**（最多四条），Hit@1 从 16/18 掉到 13/18 ——
    # 平行的几路互相稀释，把本来排第一的小节拱下去了。少而精才对。
    if len(terms) >= 2:
        head: list[str] = []
        for _, gloss in terms[:2]:
            for word in gloss.split():
                if word not in head:
                    head.append(word)
        query = " ".join(head)
        if query and query not in queries:
            queries.append(query)
    return queries


def describe(question: str) -> str:
    """给界面/日志用的一句话：命中了哪些术语。没有命中就说明白。"""
    terms = extract(question)
    if not terms:
        return "离线术语表未命中"
    return "离线术语表命中：" + "、".join(f"{zh}→{en.split()[0]}" for zh, en in terms)
