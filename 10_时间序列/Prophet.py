"""
Facebook Prophet 时间序列预测

Prophet 核心思想：将时间序列分解为三个成分的加法/乘法模型：
y(t) = g(t) + s(t) + h(t) + ε(t)
- g(t): 趋势项（线性或逻辑增长）
- s(t): 周期性季节项（年、周、日）
- h(t): 节假日/事件效应

优势：自动处理缺失值、异常值，支持自定义节假日，
默认适合商业时间序列（日/周/月粒度）。
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 1. 生成合成时间序列 ====================
np.random.seed(42)
n = 365 * 3  # 三年日数据

t = np.arange(n)
# 非线性趋势（饱和增长）
趋势 = 10 * np.log1p(t / 50)
# 年度季节性
年季节 = 2.5 * np.sin(2 * np.pi * t / 365) + 1.5 * np.cos(4 * np.pi * t / 365)
# 周季节性
周季节 = np.where(pd.Series(t).apply(lambda x: pd.Timestamp("2021-01-01") + pd.Timedelta(days=x)).dt.dayofweek >= 5, -1.5, 0)
噪声 = np.random.normal(0, 0.4, n)
数据 = 趋势 + 年季节 + 周季节 + 噪声

# 构造 Prophet 要求的 DataFrame 格式（ds + y）
dates = pd.date_range("2021-01-01", periods=n, freq="D")
df = pd.DataFrame({"ds": dates, "y": 数据})

print("=" * 55)
print("Facebook Prophet 时间序列预测")
print("=" * 55)
print(f"数据范围: {dates[0].date()} ~ {dates[-1].date()}, 共 {n} 天")
print(f"均值: {df['y'].mean():.4f}, 标准差: {df['y'].std():.4f}")

# ==================== 2. 训练/测试划分 ====================
train_size = int(n * 0.8)
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()
print(f"训练集: {len(train)} 天, 测试集: {len(test)} 天")

# ==================== 3. 自定义节假日 ====================
# 构造几个模拟"假期"
节假日 = pd.DataFrame({
    "holiday": "春节",
    "ds": pd.to_datetime(["2021-02-12", "2022-02-01", "2023-01-22"]),
    "lower_window": -3,
    "upper_window": 3,
})
国庆 = pd.DataFrame({
    "holiday": "国庆",
    "ds": pd.to_datetime(["2021-10-01", "2022-10-01", "2023-10-01"]),
    "lower_window": -1,
    "upper_window": 7,
})
所有节假日 = pd.concat([节假日, 国庆], ignore_index=True)

# ==================== 4. 模型拟合 ====================
print("\n--- 模型拟合 ---")
模型 = Prophet(
    yearly_seasonality=True,   # 年季节性
    weekly_seasonality=True,   # 周季节性
    daily_seasonality=False,   # 日粒度数据不需要日季节性
    holidays=所有节假日,         # 自定义节假日
    changepoint_prior_scale=0.05,  # 趋势灵活性（越大越灵活）
    seasonality_prior_scale=10.0,
)
模型.fit(train)
print(f"  变点数: {len(模型.changepoints)}")
print(f"  趋势变点（前 5 个）: {[d.date() for d in 模型.changepoints[:5]]}")

# ==================== 5. 预测 ====================
future = 模型.make_future_dataframe(periods=len(test))
forecast = 模型.predict(future)
预测 = forecast.iloc[train_size:].copy()

# ==================== 6. 评估 ====================
rmse = np.sqrt(mean_squared_error(test["y"], 预测["yhat"]))
mae = mean_absolute_error(test["y"], 预测["yhat"])
mape = np.mean(np.abs((test["y"].values - 预测["yhat"].values) / test["y"].values)) * 100

print(f"\n--- 预测评估 ---")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  MAPE: {mape:.2f}%")

print(f"\n--- 预测 vs 真实（前 10 个） ---")
print(f"{'日期':>12}  {'真实':>8}  {'预测':>8}  {'下界':>8}  {'上界':>8}")
for i in range(min(10, len(test))):
    print(f"  {str(test['ds'].iloc[i].date()):>10}  "
          f"{test['y'].iloc[i]:>8.3f}  "
          f"{预测['yhat'].iloc[i]:>8.3f}  "
          f"{预测['yhat_lower'].iloc[i]:>8.3f}  "
          f"{预测['yhat_upper'].iloc[i]:>8.3f}")

# ==================== 7. 趋势与季节性分解 ====================
print("\n--- 模型分解 ---")
趋势成分 = 预测["trend"]
年季节成分 = 预测["yearly"]
周季节成分 = 预测["weekly"]
节假日成分 = 预测["holidays"]

print(f"  趋势范围:     [{趋势成分.min():.3f}, {趋势成分.max():.3f}]")
print(f"  年季节范围:   [{年季节成分.min():.3f}, {年季节成分.max():.3f}]")
print(f"  周季节范围:   [{周季节成分.min():.3f}, {周季节成分.max():.3f}]")
print(f"  节假日效应:   [{节假日成分.min():.3f}, {节假日成分.max():.3f}]")

# ==================== 8. 未来预测（全量模型） ====================
print("\n--- 全量模型拟合 & 未来 60 天预测 ---")
全模型 = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    holidays=所有节假日,
    changepoint_prior_scale=0.05,
)
全模型.fit(df)
未来 = 全模型.make_future_dataframe(periods=60)
未来预测 = 全模型.predict(未来)
未来部分 = 未来预测.tail(60)

print(f"{'日期':>12}  {'预测值':>8}  {'下界':>8}  {'上界':>8}")
for i in range(60):
    row = 未来部分.iloc[i]
    print(f"  {str(row['ds'].date()):>10}  {row['yhat']:>8.3f}  {row['yhat_lower']:>8.3f}  {row['yhat_upper']:>8.3f}")
