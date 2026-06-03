# Transformer：注意力就是你所需要的

> 所属模块：06_深度学习 | 源文件：Transformer.py | 核心功能：自注意力机制、多头注意力、位置编码、编码器-解码器架构

## 概述

Transformer 是 2017 年 Google 论文 "Attention Is All You Need" 提出的架构，彻底改变了 NLP 领域，现在已扩展到 CV、音频等多个领域。核心创新：用**自注意力机制**替代 RNN 的循环结构，实现完全并行化。

## 关键代码解释

### 自注意力

`python
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
`

Q（查询）、K（键）、V（值）是输入的线性变换。注意力分数衡量"每个位置应该关注其他位置多少"。

### 多头注意力

`python
nn.MultiheadAttention(embed_dim=512, num_heads=8)
`

多个注意力头并行计算，每个头关注不同的子空间信息。

### 位置编码

Transformer 没有循环结构，需要位置编码告诉模型每个 token 的位置。通常用正弦/余弦函数或可学习的位置嵌入。

## 注意事项

1. **O(n^2) 复杂度**：序列长度的平方，长序列很慢
2. **位置编码很重要**：没有位置信息，模型无法区分序列顺序
3. **需要大量数据**：参数多，小数据集容易过拟合

## 延伸思考

- **Flash Attention**：IO 感知的高效注意力实现
- **Vision Transformer (ViT)**：Transformer 在图像领域的应用
- **线性注意力**：O(n) 复杂度的近似注意力
﻿## 数学原理

### 1. 自注意力机制（Scaled Dot-Product Attention）

**代码对应**：Transformer 的核心是自注意力。

输入序列 $\mathbf{X} \in \mathbb{R}^{n \times d}$，通过线性变换得到 Query、Key、Value：

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

注意力计算：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

除以 $\sqrt{d_k}$ 防止点积过大导致 softmax 梯度消失。

### 2. 多头注意力

**代码对应**：`nn.MultiheadAttention(embed_dim, num_heads)`。

将 Q、K、V 拆分为 $h$ 个头，每个头独立计算注意力，然后拼接：

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_Q^i, \mathbf{K}\mathbf{W}_K^i, \mathbf{V}\mathbf{W}_V^i)$$

多头让模型同时关注不同位置的不同表示子空间。

### 3. 位置编码

Transformer 没有递归结构，需要位置编码注入序列顺序信息：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

正弦位置编码的优势：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数，使模型能学习相对位置关系。

### 4. Feed-Forward Network

每个 Transformer 层包含一个位置-wise 的 FFN：

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

### 5. Layer Normalization

$$\text{LN}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu, \sigma^2$ 在特征维度上计算（不是 batch 维度）。

### 6. 计算复杂度

自注意力的计算复杂度为 $O(n^2 d)$（$n$ 为序列长度），内存复杂度 $O(n^2)$。这是 Transformer 处理长序列的主要瓶颈。
