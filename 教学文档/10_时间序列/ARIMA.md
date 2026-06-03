# ARIMA：时间序列预测的经典模型

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