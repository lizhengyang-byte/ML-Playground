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
﻿## 数学原理

### 1. LSTM 的门控机制

LSTM 通过三个门控制信息流，解决长期依赖问题：

**遗忘门**：决定丢弃哪些信息

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**输入门**：决定存储哪些新信息

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

**候选记忆**：

$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$$

**记忆更新**：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**输出门**：

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

**隐藏状态**：

$$h_t = o_t \odot \tanh(C_t)$$

其中 $\sigma$ 是 sigmoid 函数，$\odot$ 是逐元素乘法。

### 2. 滑动窗口构造

将时间序列转换为监督学习问题：

$$X_i = (y_{i}, y_{i+1}, \ldots, y_{i+L-1}), \quad Y_i = (y_{i+L}, \ldots, y_{i+L+H-1})$$

- $L$：look_back（输入窗口长度）
- $H$：horizon（预测步数）

代码中 `look_back=30, horizon=5`，用过去 30 步预测未来 5 步。

### 3. MinMaxScaler 归一化

$$\hat{y}_t = \frac{y_t - y_{min}}{y_{max} - y_{min}} \in [0, 1]$$

归一化的原因：
- LSTM 的 sigmoid/tanh 输出有界，输入也需要归一化
- 避免梯度爆炸/消失
- 加速收敛

### 4. LSTM 预测模型结构

$$\text{Input}(L) \to \text{Embedding} \to \text{LSTM}(H_{dim}) \to \text{FC}(H_{dim} \to H)$$

代码中：
- 输入：$(B, L)$ 的序列
- LSTM 输出：$(B, H_{dim})$ 的最后时间步隐藏状态
- FC 输出：$(B, H)$ 的预测值

### 5. 训练损失

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\|\hat{Y}_i - Y_i\|^2 = \text{MSE}$$

通过反向传播和 Adam 优化器更新参数。

### 6. 评估指标

**RMSE**（均方根误差）：$\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$

**MAE**：$\text{MAE} = \frac{1}{n}\sum|y_i - \hat{y}_i|$

反归一化后计算：$y_{real} = \hat{y} \times (y_{max} - y_{min}) + y_{min}$

### 7. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `MinMaxScaler()` | $\hat{y} = (y - y_{min}) / (y_{max} - y_{min})$ |
| `create_sequences(data, look_back, horizon)` | 滑动窗口 $(X_i, Y_i)$ |
| `nn.LSTM(embed_dim, hidden_dim, batch_first=True)` | LSTM 门控单元 |
| `nn.Linear(hidden_dim, horizon)` | 输出映射 $W \in \mathbb{R}^{H \times H_{dim}}$ |
| `MSELoss()` | $\mathcal{L} = \frac{1}{n}\|\hat{Y} - Y\|^2$ |
| `scaler.inverse_transform()` | 反归一化 $y = \hat{y}(y_{max}-y_{min})+y_{min}$ |
| `DataLoader(TensorDataset(X, y), batch_size=32)` | Mini-batch 训练 |
