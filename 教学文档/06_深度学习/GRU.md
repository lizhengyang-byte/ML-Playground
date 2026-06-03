# GRU：更简洁的门控循环单元

> 所属模块：06_深度学习 | 源文件：GRU.py | 核心功能：双门控机制、参数更少、效果接近 LSTM

## 概述

GRU（Gated Recurrent Unit）是 LSTM 的简化版，用两个门（重置门和更新门）替代三个门，合并了细胞状态和隐状态。参数量约为 LSTM 的 75%，在很多任务上效果相当。

## 关键代码解释

`python
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
output, hn = gru(x)  # 没有单独的细胞状态
`

更新门 z_t 控制保留多少旧信息、接收多少新信息。重置门 r_t 控制如何结合旧隐状态和新输入。

## 注意事项

1. **训练更快**：参数比 LSTM 少
2. **选择建议**：先试 GRU，不够再换 LSTM
3. **小数据集上差异不大**

## 延伸思考

- **LSTM vs GRU**：没有绝对优劣，取决于任务和数据
- **SRU（Simple Recurrent Unit）**：进一步简化，可并行化
- **State Space Models**：Mamba 等新架构正在取代 RNN