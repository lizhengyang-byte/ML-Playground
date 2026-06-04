# SARIMA：带季节性的 ARIMA
> 难度标签：中级 | 预计时长：10-20分钟 | 前置知识：无学习经验


> 所属模块：10_时间序列 | 源文件：SARIMA.py | 核心功能：季节性差分、季节性参数、周期性建模

## 概述

SARIMA 在 ARIMA 基础上加入季节性分量（P, D, Q, s），能建模周期性规律（如月度销售额、日温度变化）。

## 关键代码解释

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
```

`s=12` 表示季节周期为 12（月度数据的年周期）。

## 注意事项

1. 季节周期 s 需要根据数据频率设定
2. 参数多，调参困难
3. 长期预测效果差（统计模型的通病）

## 延伸思考

- **ETS 模型**：指数平滑状态空间模型
- **TBATS**：多季节性时间序列
- **机器学习方法**：XGBoost、LSTM 对时序建模
## 数学原理

### 1. SARIMA(p,d,q)(P,D,Q,s) 模型

SARIMA 在 ARIMA 基础上增加季节性分量，记为 $\text{SARIMA}(p,d,q)(P,D,Q,s)$：

$$(1 - \sum_{i=1}^p \phi_i B^i)(1 - \sum_{I=1}^P \Phi_I B^{Is})(1-B)^d(1-B^s)^D y_t$$

$$= (1 + \sum_{j=1}^q \theta_j B^j)(1 + \sum_{J=1}^Q \Theta_J B^{Js})\epsilon_t$$

其中 $B$ 是后移算子（$B y_t = y_{t-1}$），$s$ 是季节周期。

### 2. 季节性分解

加法分解模型：

$$y_t = T_t + S_t + R_t$$

- $T_t$：趋势成分（长期变化方向）
- $S_t$：季节成分（固定周期重复模式），$\sum_{j=0}^{s-1} S_{t+j} = 0$
- $R_t$：残差成分（随机波动）

乘法分解模型：$y_t = T_t \times S_t \times R_t$

### 3. 季节性差分

**一阶季节差分**（周期 $s$）：

$$\nabla_s y_t = y_t - y_{t-s}$$

消除季节性。代码中 SARIMA 的 $D$ 参数控制季节性差分阶数。

**组合差分**：同时做普通差分和季节差分

$$\nabla \nabla_s y_t = (y_t - y_{t-1}) - (y_{t-s} - y_{t-s-1})$$

### 4. ADF 检验与平稳性

对差分后的序列进行 ADF 检验，确认平稳性：

$$H_0: \text{序列非平稳}, \quad H_1: \text{序列平稳}$$

$p < 0.05$ 拒绝 $H_0$，序列可建模。

### 5. 参数选择

| 参数 | 含义 | 典型值 |
|------|------|--------|
| $p$ | 非季节 AR 阶数 | 0-3 |
| $d$ | 普通差分阶数 | 0-2 |
| $q$ | 非季节 MA 阶数 | 0-3 |
| $P$ | 季节 AR 阶数 | 0-2 |
| $D$ | 季节差分阶数 | 0-1 |
| $Q$ | 季节 MA 阶数 | 0-2 |
| $s$ | 季节周期 | 12（月）、7（周）、365（日） |

### 6. 预测评估指标

**MSE**（均方误差）：$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$

**MAE**（平均绝对误差）：$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$

**MAPE**（平均绝对百分比误差）：$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$

### 7. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `SARIMAX(序列, order=(p,d,q), seasonal_order=(P,D,Q,s))` | SARIMA 模型 |
| `seasonal_decompose(序列, model="additive", period=365)` | 加法季节分解 $y = T + S + R$ |
| `model.fit(disp=False)` | 最大似然估计参数 |
| `forecast(steps=len(test))` | 多步预测 |
| `mean_squared_error(test, pred)` | MSE 评估 |
| `acorr_ljungbox(resid)` | 残差白噪声检验 |
