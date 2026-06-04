# GRU：更简洁的门控循环单元
> 难度标签：高级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：06_深度学习 | 源文件：GRU.py | 核心功能：双门控机制、参数更少、效果接近 LSTM

## 概述

GRU（Gated Recurrent Unit）是 LSTM 的简化版，用两个门（重置门和更新门）替代三个门，合并了细胞状态和隐状态。参数量约为 LSTM 的 75%，在很多任务上效果相当。

## 关键代码解释

```python
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, hn = gru(x)  # 没有单独的细胞状态
```

更新门 z_t 控制保留多少旧信息、接收多少新信息。重置门 r_t 控制如何结合旧隐状态和新输入。

## 注意事项

1. **训练更快**：参数比 LSTM 少
2. **选择建议**：先试 GRU，不够再换 LSTM
3. **小数据集上差异不大**

## 延伸思考

- **LSTM vs GRU**：没有绝对优劣，取决于任务和数据
- **SRU（Simple Recurrent Unit）**：进一步简化，可并行化
- **State Space Models**：Mamba 等新架构正在取代 RNN
## 数学原理

### 1. GRU 的门控机制

**代码对应**：GRU 是 LSTM 的简化版本，合并了细胞状态和隐藏状态。

**重置门**：$\mathbf{r}_t = \sigma(\mathbf{W}_r[\mathbf{h}_{t-1}, \mathbf{x}_t])$

**更新门**：$\mathbf{z}_t = \sigma(\mathbf{W}_z[\mathbf{h}_{t-1}, \mathbf{x}_t])$

**候选隐藏状态**：$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h[\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$

**隐藏状态更新**：$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$

### 2. GRU vs LSTM

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数 | 3（遗忘、输入、输出） | 2（重置、更新） |
| 状态 | $\mathbf{h}_t$ 和 $\mathbf{c}_t$ 分离 | 只有 $\mathbf{h}_t$ |
| 参数量 | $4(d_h^2 + d_h d_x + d_h)$ | $3(d_h^2 + d_h d_x + d_h)$ |
| 训练速度 | 较慢 | 较快（参数少 ~25%） |
| 长序列 | 通常更好 | 类似或稍差 |

GRU 的更新门 $\mathbf{z}_t$ 同时扮演 LSTM 的遗忘门和输入门的角色：当 $\mathbf{z}_t \approx 0$ 时保留旧状态，当 $\mathbf{z}_t \approx 1$ 时用新候选替换。
