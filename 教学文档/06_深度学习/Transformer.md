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