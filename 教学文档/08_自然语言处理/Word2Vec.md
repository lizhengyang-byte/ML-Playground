# Word2Vec：词向量的革命

> 所属模块：08_自然语言处理 | 源文件：Word2Vec.py | 核心功能：Skip-gram/CBOW、词向量训练、语义相似度

## 概述

Word2Vec（2013）将每个词映射到一个低维稠密向量，语义相似的词在向量空间中距离近。经典发现：king - man + woman ≈ queen。

## 关键代码解释

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
# sg=1: Skip-gram, sg=0: CBOW
model.wv.most_similar("机器学习", topn=5)
```

## 注意事项

1. 需要大量语料训练
2. 一词多义问题无法解决（同一个词只有一个向量）
3. OOV（词表外）词没有向量

## 延伸思考

- **FastText**：子词级别的词向量，解决 OOV
- **GloVe**：基于共现矩阵的词向量
- **上下文词向量**：ELMo/BERT 每个词的向量随上下文变化
﻿## 数学原理

### 1. Word2Vec 的核心思想

Word2Vec 将每个词映射为一个 $d$ 维稠密向量（词嵌入），使得**语义相似的词在向量空间中距离相近**。

设词汇表为 $V$，词 $w$ 的嵌入向量为 $\mathbf{v}_w \in \mathbb{R}^d$，嵌入矩阵 $W \in \mathbb{R}^{|V| \times d}$。

### 2. CBOW 模型（Continuous Bag of Words）

CBOW 用上下文词预测中心词。

**输入**：中心词 $w_t$ 的上下文窗口 $\{w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c}\}$

**上下文向量**（平均）：

$$\mathbf{h} = \frac{1}{2c}\sum_{-c \leq j \leq c, j \neq 0} \mathbf{v}_{w_{t+j}}$$

**输出**：通过 softmax 计算中心词的概率：

$$P(w_t | \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{h})}$$

其中 $\mathbf{u}_w$ 是词 $w$ 的输出嵌入向量。

### 3. Skip-gram 模型

Skip-gram 用中心词预测上下文词（代码中 `sg=1` 使用此模型）。

**目标**：最大化给定中心词 $w_t$ 时上下文词 $w_{t+j}$ 的对数似然：

$$\mathcal{L} = \sum_{t=1}^{T}\sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

其中：

$$P(w_o | w_i) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_i})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_i})}$$

### 4. 负采样（Negative Sampling）

直接计算 softmax 的分母需遍历整个词汇表 $|V|$，计算量巨大。负采样将其简化为二分类问题：

$$\mathcal{L}_{neg} = \log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_i}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)}[\log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_{w_i})]$$

其中 $\sigma$ 是 sigmoid 函数，$K$ 是负样本数，$P_n(w) \propto f(w)^{3/4}$ 是噪声分布（词频的 3/4 次幂）。

### 5. 词向量的线性类比关系

训练好的词向量满足著名的类比关系：

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

这表明词向量空间中存在**线性子结构**，能捕捉语义和语法关系。

### 6. 余弦相似度

词向量间的相似度用余弦相似度衡量：

$$\text{sim}(w_i, w_j) = \cos(\theta) = \frac{\mathbf{v}_{w_i} \cdot \mathbf{v}_{w_j}}{\|\mathbf{v}_{w_i}\| \cdot \|\mathbf{v}_{w_j}\|}$$

代码中 `model.wv.most_similar("学习")` 返回余弦相似度最高的词。

### 7. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `vector_size=50` | 嵌入维度 $d=50$ |
| `window=3` | 上下文窗口 $c=3$ |
| `sg=1` | 使用 Skip-gram 模型 |
| `sg=0` | 使用 CBOW 模型 |
| `min_count=1` | 保留所有词（无低频词过滤） |
| `epochs=100` | 训练轮数 |
| `most_similar()` | $\arg\max_{w'} \cos(\mathbf{v}_w, \mathbf{v}_{w'})$ |
| `similarity(w1, w2)` | $\cos(\mathbf{v}_{w_1}, \mathbf{v}_{w_2})$ |
