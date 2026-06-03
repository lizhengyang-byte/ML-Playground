# LSTM：解决长期依赖的门控机制

> 所属模块：06_深度学习 | 源文件：LSTM.py | 核心功能：遗忘门/输入门/输出门、细胞状态、长距离依赖

## 概述

LSTM（Long Short-Term Memory）是 RNN 最成功的改进版本。它引入了**细胞状态**（cell state）和三个**门控机制**，解决了标准 RNN 的梯度消失问题。

三个门：遗忘门（丢弃旧信息）、输入门（写入新信息）、输出门（输出信息）。细胞状态像一条"传送带"，信息可以几乎无损地传递很远。

## 关键代码解释

### LSTM 单元

`python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, (hn, cn) = lstm(x)  # cn 是细胞状态
`

### 门控机制

`python
f_t = sigmoid(W_f @ [h_{t-1}, x_t])  # 遗忘门：丢多少旧信息
i_t = sigmoid(W_i @ [h_{t-1}, x_t])  # 输入门：加多少新信息
o_t = sigmoid(W_o @ [h_{t-1}, x_t])  # 输出门：输出什么
`

## 注意事项

1. **比 RNN 慢**：参数量是 RNN 的 4 倍
2. **仍然有梯度问题**：比 RNN 好很多但不是完全解决
3. **不能并行**：时间步仍需串行

## 延伸思考

- **GRU**：LSTM 的简化版，2 个门，参数更少
- **Peephole LSTM**：让门也能看到细胞状态
- **Transformer**：完全抛弃循环结构，用自注意力替代