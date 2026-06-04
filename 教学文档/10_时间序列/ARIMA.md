# ARIMA：时间序列预测的经典模型
> 难度标签：中级 | 预计时长：10-20分钟 | 前置知识：无学习经验


> 所属模块：10_时间序列 | 源文件：ARIMA.py | 核心功能：平稳性检验、差分、自相关分析、statsmodels 实现

## 概述

ARIMA（AutoRegressive Integrated Moving Average）是时间序列预测的经典统计模型。三个参数：p（自回归阶数）、d（差分阶数）、q（移动平均阶数）。

## 关键代码解释

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(ts, order=(p, d, q))
fitted = model.fit()
forecast = fitted.forecast(steps=10)
```

## 注意事项

1. 数据必须平稳（均值和方差不随时间变化）
2. ADF 检验判断平稳性
3. ACF/PACF 图辅助选择 p 和 q

## 延伸思考

- **SARIMA**：加入季节性
- **ARIMA 的自动选择**：`pmdarima.auto_arima`
- **Prophet**：Facebook 的时间序列工具，更易用
## 数学原理

### 1. ARIMA(p, d, q) 模型

ARIMA 由三部分组成：自回归（AR）、差分（I）、滑动平均（MA）。

**AR(p) 自回归**：

$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \epsilon_t$$

当前值 $y_t$ 是过去 $p$ 个值的线性组合加白噪声。

**MA(q) 滑动平均**：

$$y_t = \mu + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

当前值 $y_t$ 是过去 $q$ 个误差的线性组合。

**ARMA(p, q)**（平稳序列）：

$$y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \epsilon_t + \sum_{j=1}^{q}\theta_j \epsilon_{t-j}$$

### 2. 差分与平稳性

**一阶差分**：$\nabla y_t = y_t - y_{t-1}$

**$d$ 阶差分**：$\nabla^d y_t = \nabla(\nabla^{d-1} y_t)$

差分消除趋势，使序列平稳。ARIMA 对差分后的序列 $\nabla^d y_t$ 拟合 ARMA(p, q)。

### 3. ADF 平稳性检验

原假设 $H_0$：序列存在单位根（非平稳）。检验统计量：

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{j=1}^{k}\delta_j \Delta y_{t-j} + \epsilon_t$$

检验 $\gamma = 0$ 是否成立。$p < 0.05$ 时拒绝 $H_0$，认为序列平稳。

### 4. ACF 与 PACF 定阶

| 模型 | ACF | PACF |
|------|-----|------|
| AR(p) | 拖尾（指数衰减） | $p$ 阶截尾 |
| MA(q) | $q$ 阶截尾 | 拖尾 |
| ARMA(p,q) | 拖尾 | 拖尾 |

**截尾**：在某阶之后突然变为 0（统计显著范围内）。**拖尾**：逐渐衰减。

### 5. 模型选择准则

**AIC**（Akaike 信息准则）：

$$\text{AIC} = -2\ln \hat{L} + 2k$$

**BIC**（贝叶斯信息准则）：

$$\text{BIC} = -2\ln \hat{L} + k\ln n$$

其中 $\hat{L}$ 是最大似然值，$k$ 是参数个数。AIC/BIC 越小越好。

### 6. 预测

ARIMA 的 $h$ 步预测：

$$\hat{y}_{T+h|T} = c + \sum_{i=1}^{p}\phi_i \hat{y}_{T+h-i|T} + \sum_{j=1}^{q}\theta_j \hat{\epsilon}_{T+h-j}$$

其中 $\hat{y}_{T+h-i|T} = y_{T+h-i}$（当 $h-i \leq 0$），预测区间随 $h$ 增大而变宽。

### 7. Ljung-Box 检验

检验残差是否为白噪声（模型充分）：

$$Q(k) = n(n+2)\sum_{j=1}^{k}\frac{\hat{\rho}_j^2}{n-j}$$

$H_0$：残差无自相关。$p > 0.05$ 表示模型拟合充分。

### 8. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `adfuller(序列)` | ADF 检验 $\gamma=0$ |
| `ARIMA(序列, order=(p,d,q))` | ARIMA(p,d,q) 模型 |
| `acf(序列, nlags=40)` | 计算样本自相关函数 $\hat{\rho}_k$ |
| `pacf(序列, nlags=40)` | 计算样本偏自相关函数 $\hat{\phi}_{kk}$ |
| `acorr_ljungbox(resid)` | Ljung-Box 白噪声检验 |
| `mean_squared_error` | $\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
| `forecast(steps=20)` | $h$ 步预测 $\hat{y}_{T+h|T}$ |
