# SARIMA：带季节性的 ARIMA

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