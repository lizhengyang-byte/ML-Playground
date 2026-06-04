# LSTM：解决长期依赖的门控机制
> 难度标签：高级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：06_深度学习 | 源文件：LSTM.py | 核心功能：遗忘门/输入门/输出门、细胞状态、长距离依赖

## 概述

LSTM（Long Short-Term Memory）是 RNN 最成功的改进版本。它引入了**细胞状态**（cell state）和三个**门控机制**，解决了标准 RNN 的梯度消失问题。

三个门：遗忘门（丢弃旧信息）、输入门（写入新信息）、输出门（输出信息）。细胞状态像一条"传送带"，信息可以几乎无损地传递很远。

## 关键代码解释

### LSTM 单元

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, (hn, cn) = lstm(x)  # cn 是细胞状态
```

### 门控机制

```python
f_t = sigmoid(W_f @ [h_{t-1}, x_t])  # 遗忘门：丢多少旧信息
i_t = sigmoid(W_i @ [h_{t-1}, x_t])  # 输入门：加多少新信息
o_t = sigmoid(W_o @ [h_{t-1}, x_t])  # 输出门：输出什么
```

## 注意事项

1. **比 RNN 慢**：参数量是 RNN 的 4 倍
2. **仍然有梯度问题**：比 RNN 好很多但不是完全解决
3. **不能并行**：时间步仍需串行

## 延伸思考

- **GRU**：LSTM 的简化版，2 个门，参数更少
- **Peephole LSTM**：让门也能看到细胞状态
- **Transformer**：完全抛弃循环结构，用自注意力替代
## 数学原理

### 1. LSTM 的门控机制

**代码对应**：LSTM 通过三个门控制信息流，解决 RNN 的梯度消失问题。

**遗忘门**：$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$

**输入门**：$\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$

**候选记忆**：$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$

**细胞状态更新**：$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$

**输出门**：$\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$

**隐藏状态**：$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$

### 2. 为什么 LSTM 能解决梯度消失

细胞状态的梯度：

$$\frac{\partial\mathbf{c}_t}{\partial\mathbf{c}_{t-1}} = \text{diag}(\mathbf{f}_t)$$

当 $\mathbf{f}_t \approx \mathbf{1}$ 时（遗忘门打开），$\frac{\partial\mathbf{c}_t}{\partial\mathbf{c}_{t-1}} \approx \mathbf{I}$，梯度沿细胞状态**无衰减**传播。这是 LSTM 的关键创新——细胞状态是一条"梯度高速公路"。

对比 RNN：$\frac{\partial\mathbf{h}_t}{\partial\mathbf{h}_{t-1}} = \text{diag}(\tanh'(\cdot))\mathbf{W}_{hh}$，连乘后必然衰减。

### 3. LSTM 的参数量

每个门的参数：$4 \times (d_h \times d_h + d_h \times d_x + d_h)$

总参数量约为普通 RNN 的 4 倍（4 个门共享相同的输入组合）。
