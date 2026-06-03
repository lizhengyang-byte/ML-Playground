# Prophet：Facebook 的时间序列利器

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