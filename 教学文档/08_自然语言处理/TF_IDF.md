# TF-IDF：词频-逆文档频率

> 所属模块：08_自然语言处理 | 源文件：TF_IDF.py | 核心功能：TF-IDF 加权、文本特征提取

## 概述

TF-IDF 是词袋模型的改进。TF（词频）衡量词在文档中的频率，IDF（逆文档频率）降低常见词的权重。两者相乘，让"对当前文档重要但在整个语料中不常见"的词获得高权重。

## 关键代码解释

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True)
X = tfidf.fit_transform(texts)
```

`sublinear_tf=True` 用 1+log(tf) 替代 tf，压缩高频词的影响。

## 注意事项

1. 仍然是稀疏高维向量
2. IDF 在训练集上计算，应用到测试集
3. 中文需要先分词

## 延伸思考

- **BM25**：TF-IDF 的改进版，搜索引擎的标准
- **TF-IDF + SVM**：经典的文本分类组合
- **Transformer 时代**：BERT 等预训练模型是否完全取代了 TF-IDF
﻿## 数学原理

### 1. TF-IDF 的基本思想

TF-IDF（Term Frequency - Inverse Document Frequency）衡量一个词对文档的重要程度。核心直觉：一个词在当前文档中出现多（高 TF），但在其他文档中出现少（高 IDF），则该词具有很强的区分能力。

### 2. 词频（TF）

词 $t$ 在文档 $d$ 中的词频：

$$\text{TF}(t, d) = \frac{c(t, d)}{\sum_{t' \in d} c(t', d)}$$

其中 $c(t, d)$ 是词 $t$ 在文档 $d$ 中的出现次数，分母是文档 $d$ 的总词数。

### 3. 逆文档频率（IDF）

词 $t$ 的逆文档频率：

$$\text{IDF}(t) = \log \frac{N}{|\{d \in D : t \in d\}|}$$

其中 $N$ 是文档总数，$|\{d : t \in d\}|$ 是包含词 $t$ 的文档数。

- 常见词（如"的"、"是"）：出现在几乎所有文档中，IDF $\approx 0$
- 罕见词（如"量子计算"）：只出现在少数文档中，IDF 较大

### 4. TF-IDF 权重

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

**sklearn 的平滑版本**（默认）：

$$\text{IDF}(t) = \log \frac{N + 1}{|\{d : t \in d\}| + 1} + 1$$

加 1 避免除零，加 1 避免 IDF 为零。

### 5. L2 归一化

sklearn 的 `TfidfVectorizer` 在计算 TF-IDF 后还进行 **L2 归一化**：

$$\hat{x}_{t,d} = \frac{\text{TF-IDF}(t, d)}{\sqrt{\sum_{t'} \text{TF-IDF}(t', d)^2}}$$

使得每个文档向量的 L2 范数为 1，便于余弦相似度计算。

### 6. 从 BoW 到 TF-IDF 的变换

代码中展示了两种方式：

```python
# 方式一：直接计算 TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)

# 方式二：先 BoW 再变换
pipeline = make_pipeline(CountVectorizer(), TfidfTransformer())
```

数学上等价：$\text{BoW} \xrightarrow{\text{TF}} \text{TF} \xrightarrow{\times \text{IDF}} \text{TF-IDF} \xrightarrow{\text{L2 norm}} \hat{X}$

### 7. TF-IDF 的几何解释

TF-IDF 将文档映射到 $|V|$ 维空间中的单位球面上（L2 归一化后）。文档间的相似度用**余弦相似度**：

$$\cos(\theta) = \mathbf{x}_d \cdot \mathbf{x}_{d'} = \sum_t \hat{x}_{t,d} \cdot \hat{x}_{t,d'}$$

### 8. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `TfidfVectorizer()` | 计算 TF-IDF 矩阵 $\hat{X} \in \mathbb{R}^{n \times |V|}$ |
| `fit_transform(corpus)` | 从语料库学习词汇表并计算 TF-IDF |
| `ngram_range=(1,2)` | 包含 unigram 和 bigram 特征 |
| `MultinomialNB` + TF-IDF | 朴素贝叶斯分类器，输入 TF-IDF 特征 |
| `make_pipeline(CountVectorizer(), TfidfTransformer())` | BoW → TF → IDF → L2 归一化 |
| `top_idx = row.argsort()[::-1][:3]` | 每个文档 TF-IDF 值最高的 3 个词 |
