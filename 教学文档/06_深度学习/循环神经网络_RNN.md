# 循环神经网络 RNN：记住过去的信息
> 难度标签：高级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：06_深度学习 | 源文件：循环神经网络_RNN.py | 核心功能：序列建模、隐状态传递、梯度消失问题

## 概述

RNN 专门为序列数据设计。它在每个时间步接收当前输入和上一步的隐状态，输出新的隐状态。这让 RNN 有"记忆"——能记住序列中之前的信息。

核心公式：h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)

## 关键代码解释

### PyTorch RNN

```python
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, hn = rnn(x)  # output: 所有时间步输出, hn: 最后隐状态
```

### 梯度消失问题

RNN 在长序列上训练时，梯度会指数级衰减（消失）或爆炸。这导致 RNN 难以学习长距离依赖。

## 注意事项

1. **梯度消失/爆炸**：长序列上严重，需要 LSTM/GRU 或梯度裁剪
2. **不能并行**：时间步必须串行计算
3. **短序列效果好**：长序列考虑 LSTM/Transformer

## 延伸思考

- **双向 RNN**：同时从左到右和从右到左读序列
- **深层 RNN**：多层 RNN 堆叠
- **Transformer 的崛起**：自注意力完全替代 RNN 的序列建模能力
## 数学原理

### 1. RNN 的递推公式

**代码对应**：RNN 在时间步 $t$ 的隐藏状态更新：

$$\mathbf{h}_t = \tanh(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h)$$

$$\hat{\mathbf{y}}_t = \mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y$$

其中 $\mathbf{W}_{xh}$ 为输入到隐藏的权重，$\mathbf{W}_{hh}$ 为隐藏到隐藏的权重（循环连接）。

### 2. 通过时间的反向传播（BPTT）

梯度沿时间步反向传播：

$$\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T}\frac{\partial L_t}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T}\sum_{k=1}^{t}\frac{\partial L_t}{\partial \mathbf{h}_t}\left(\prod_{j=k+1}^{t}\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}\right)\frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$

其中 $\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \text{diag}(\tanh'(\mathbf{z}_j))\mathbf{W}_{hh}$。

### 3. 梯度消失/爆炸

$\prod_{j=k+1}^{t}\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} \approx \mathbf{W}_{hh}^{t-k}$（忽略激活函数导数）

- $\|\mathbf{W}_{hh}\| < 1$：梯度指数衰减（梯度消失）→ 长距离依赖无法学习
- $\|\mathbf{W}_{hh}\| > 1$：梯度指数增长（梯度爆炸）→ 训练不稳定

**梯度裁剪**（Gradient Clipping）缓解爆炸：$\mathbf{g} \leftarrow \min(1, \frac{c}{\|\mathbf{g}\|})\mathbf{g}$

这就是 LSTM 和 GRU 被发明的根本原因。
