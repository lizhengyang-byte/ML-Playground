# 循环神经网络 RNN：记住过去的信息

> 所属模块：06_深度学习 | 源文件：循环神经网络_RNN.py | 核心功能：序列建模、隐状态传递、梯度消失问题

## 概述

RNN 专门为序列数据设计。它在每个时间步接收当前输入和上一步的隐状态，输出新的隐状态。这让 RNN 有"记忆"——能记住序列中之前的信息。

核心公式：h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)

## 关键代码解释

### PyTorch RNN

`python
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, hn = rnn(x)  # output: 所有时间步输出, hn: 最后隐状态
`

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