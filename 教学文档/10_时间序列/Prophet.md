# Prophet：Facebook 的时间序列利器
> 难度标签：中级 | 预计时长：10-20分钟 | 前置知识：无学习经验


> 所属模块：10_时间序列 | 源文件：Prophet.py | 核心功能：趋势+季节+节假日分解、自动预测

## 概述

Prophet 是 Facebook 开发的时间序列预测工具，设计目标是让非专家也能做出高质量预测。它将时间序列分解为趋势 + 季节性 + 节假日效应。

## 关键代码解释

```python
from prophet import Prophet
m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.fit(df)  # df 有 ds (日期) 和 y (值) 列
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
```

## 注意事项

1. 需要安装 `prophet` 库
2. 对突变事件敏感
3. 不适合高频率数据（秒级/分钟级）

## 延伸思考

- **NeuralProphet**：Prophet 的深度学习版本
- **自动时间序列**：AutoTS、AutoGluon-TimeSeries
- **时序大模型**：TimesFM、Chronos 等基础模型
## 数学原理

### 1. Prophet 的分解模型

Prophet 将时间序列分解为三个成分：

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

- $g(t)$：趋势项（增长曲线）
- $s(t)$：周期性季节项
- $h(t)$：节假日/事件效应
- $\epsilon_t$：误差项（假设服从正态分布）

### 2. 趋势模型

**线性趋势**：

$$g(t) = (k + \mathbf{a}(t)^\top \delta) t + (m + \mathbf{a}(t)^\top \gamma)$$

其中 $k$ 是基础增长率，$\delta$ 是速率调整量，$m$ 是偏移量。

**逻辑增长趋势**（饱和增长）：

$$g(t) = \frac{C(t)}{1 + \exp(-(k + \mathbf{a}(t)^\top \delta)(t - (m + \mathbf{a}(t)^\top \gamma)))}$$

其中 $C(t)$ 是承载能力（可随时间变化）。

**变点检测**：Prophet 自动检测趋势变化点 $\{s_1, \ldots, s_S\}$，在每个变点处允许增长率变化。

### 3. 季节性模型（傅里叶级数）

用傅里叶级数拟合周期为 $P$ 的季节性：

$$s(t) = \sum_{n=1}^{N}\left(a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right)\right)$$

- 年季节性：$P=365.25$，$N=10$（默认）
- 周季节性：$P=7$，$N=3$（默认）

$N$ 越大，季节性曲线越灵活（可能过拟合）。

### 4. 节假日/事件效应

$$h(t) = \sum_{i=1}^{L} \kappa_i \cdot \mathbb{1}[t \in D_i]$$

其中 $D_i$ 是第 $i$ 个节假日的影响窗口（前后若干天），$\kappa_i$ 是待学习的效应大小。

### 5. 拟合方法

Prophet 使用 **Stan** 后端进行最大后验估计（MAP）：

$$\theta^* = \arg\max_\theta \left[\log P(y|\theta) + \log P(\theta)\right]$$

- 趋势参数的先验：变点数量、增长率的正则化
- 季节性参数的先验：傅里叶系数的正则化

### 6. 不确定性估计

Prophet 提供预测区间，通过：
- 趋势的不确定性：模拟未来变点的可能位置和大小
- 季节性的不确定性：通过贝叶斯后验采样

### 7. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `Prophet()` | 构建分解模型 $y = g + s + h + \epsilon$ |
| `df["ds"]` | 时间戳 $t$ |
| `df["y"]` | 观测值 $y(t)$ |
| `model.fit(train)` | MAP 估计所有参数 |
| `model.predict(future)` | 输出 $g(t), s(t), h(t)$ 及预测值 |
| `forecast["yhat"]` | $\hat{y}(t) = g(t) + s(t) + h(t)$ |
| `forecast["yhat_lower/upper"]` | 预测区间（不确定性） |
| `add_seasonality(name, period, fourier_order)` | 添加傅里叶季节性，$P$ 和 $N$ |
