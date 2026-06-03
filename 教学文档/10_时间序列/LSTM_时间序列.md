# LSTM 时间序列预测：深度学习版

> 所属模块：10_时间序列 | 源文件：LSTM_时间序列.py | 核心功能：序列窗口化、LSTM 模型、滚动预测

## 概述

用 LSTM 网络做时间序列预测。核心思想：将历史数据窗口化（如过去 30 天预测第 31 天），用 LSTM 学习时序模式。

## 关键代码解释

```python
# 窗口化：[t-29, t-28, ..., t] -> t+1
X_windows = sliding_window(data, window_size=30)
model = nn.LSTM(input_size=1, hidden_size=50, num_layers=2)
```

## 注意事项

1. 数据归一化很重要
2. 滚动预测误差会累积
3. 简单问题上 ARIMA 可能更好

## 延伸思考

- **Transformer for Time Series**：时序 Transformer
- **Temporal Fusion Transformer**：Google 的多步预测模型
- **时序特征工程**：滞后特征、滚动统计