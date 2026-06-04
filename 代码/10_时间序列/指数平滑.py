"""
指数平滑方法

三种经典指数平滑模型：
1. 简单指数平滑 (SES): 无趋势无季节性，适合平稳序列
   ŷ_{t+1} = α·y_t + (1-α)·ŷ_t

2. 双重指数平滑 (Holt): 含趋势，无季节性
   水平: l_t = α·y_t + (1-α)·(l_{t-1} + b_{t-1})
   趋势: b_t = β·(l_t - l_{t-1}) + (1-β)·b_{t-1}

3. 三重指数平滑 (Holt-Winters): 含趋势 + 季节性
   加法模型: y_t = (l_t + b_t) + s_{t-m}
   乘法模型: y_t = (l_t + b_t) · s_{t-m}
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 1. 生成合成时间序列 ====================
np.random.seed(42)
n = 240  # 20 年月度数据

t = np.arange(n)
趋势 = 0.15 * t
季节性 = 3 * np.sin(2 * np.pi * t / 12)
噪声 = np.random.normal(0, 0.8, n)
数据 = 20 + 趋势 + 季节性 + 噪声

序列 = pd.Series(数据, index=pd.date_range("2005-01-01", periods=n, freq="MS"))
print("=" * 55)
print("指数平滑方法对比")
print("=" * 55)
print(f"数据: {n} 个月 ({序列.index[0].date()} ~ {序列.index[-1].date()})")
print(f"均值: {序列.mean():.4f}, 标准差: {序列.std():.4f}")

# 训练/测试划分
train_size = int(n * 0.8)
train, test = 序列[:train_size], 序列[train_size:]
print(f"训练集: {len(train)} 个月, 测试集: {len(test)} 个月")

# ==================== 2. 平稳性检验 ====================
result = adfuller(train.dropna(), autolag="AIC")
print(f"\n--- 训练集 ADF 检验 ---")
print(f"  ADF 统计量: {result[0]:.4f}, p 值: {result[1]:.4f} → {'平稳' if result[1] < 0.05 else '非平稳'}")

# ==================== 3. 方法一：简单指数平滑 (SES) ====================
print("\n" + "-" * 55)
print("方法一: 简单指数平滑 (Simple Exponential Smoothing)")
print("-" * 55)
print("适用: 无趋势、无季节性的平稳序列")

模型_ses = SimpleExpSmoothing(train, initialization_method="estimated")
ses_fit = 模型_ses.fit(optimized=True)
print(f"  平滑参数 α: {ses_fit.params['smoothing_level']:.4f}")
print(f"  初始水平:   {ses_fit.params['initial_level']:.4f}")
print(f"  AIC:        {ses_fit.aic:.2f}")

ses_pred = ses_fit.forecast(steps=len(test))
rmse_ses = np.sqrt(mean_squared_error(test, ses_pred))
mae_ses = mean_absolute_error(test, ses_pred)
print(f"  RMSE: {rmse_ses:.4f}")
print(f"  MAE:  {mae_ses:.4f}")

# ==================== 4. 方法二：双重指数平滑 (Holt) ====================
print("\n" + "-" * 55)
print("方法二: 双重指数平滑 (Holt / Holt's Linear)")
print("-" * 55)
print("适用: 有趋势、无季节性的序列")

模型_holt = ExponentialSmoothing(train, trend="add", seasonal=None,
                                 initialization_method="estimated")
holt_fit = 模型_holt.fit(optimized=True)
print(f"  平滑参数 α: {holt_fit.params['smoothing_level']:.4f}")
print(f"  趋势参数 β: {holt_fit.params['smoothing_trend']:.4f}")
print(f"  初始水平:   {holt_fit.params['initial_level']:.4f}")
print(f"  初始趋势:   {holt_fit.params['initial_trend']:.4f}")
print(f"  AIC:        {holt_fit.aic:.2f}")

holt_pred = holt_fit.forecast(steps=len(test))
rmse_holt = np.sqrt(mean_squared_error(test, holt_pred))
mae_holt = mean_absolute_error(test, holt_pred)
print(f"  RMSE: {rmse_holt:.4f}")
print(f"  MAE:  {mae_holt:.4f}")

# ==================== 5. 方法三：Holt-Winters 加法模型 ====================
print("\n" + "-" * 55)
print("方法三: Holt-Winters 加法季节性模型")
print("-" * 55)
print("适用: 有趋势、有加法季节性的序列（季节幅度不随趋势变化）")

模型_hw_add = ExponentialSmoothing(train, trend="add", seasonal="add",
                                   seasonal_periods=12,
                                   initialization_method="estimated")
hw_add_fit = 模型_hw_add.fit(optimized=True)
print(f"  平滑参数 α:     {hw_add_fit.params['smoothing_level']:.4f}")
print(f"  趋势参数 β:     {hw_add_fit.params['smoothing_trend']:.4f}")
print(f"  季节参数 γ:     {hw_add_fit.params['smoothing_seasonal']:.4f}")
print(f"  初始水平:       {hw_add_fit.params['initial_level']:.4f}")
print(f"  AIC:            {hw_add_fit.aic:.2f}")

hw_add_pred = hw_add_fit.forecast(steps=len(test))
rmse_hw_add = np.sqrt(mean_squared_error(test, hw_add_pred))
mae_hw_add = mean_absolute_error(test, hw_add_pred)
print(f"  RMSE: {rmse_hw_add:.4f}")
print(f"  MAE:  {mae_hw_add:.4f}")

# ==================== 6. 方法四：Holt-Winters 乘法模型 ====================
print("\n" + "-" * 55)
print("方法四: Holt-Winters 乘法季节性模型")
print("-" * 55)
print("适用: 有趋势、有乘法季节性的序列（季节幅度随趋势增长）")

模型_hw_mul = ExponentialSmoothing(train, trend="add", seasonal="mul",
                                   seasonal_periods=12,
                                   initialization_method="estimated")
hw_mul_fit = 模型_hw_mul.fit(optimized=True)
print(f"  平滑参数 α:     {hw_mul_fit.params['smoothing_level']:.4f}")
print(f"  趋势参数 β:     {hw_mul_fit.params['smoothing_trend']:.4f}")
print(f"  季节参数 γ:     {hw_mul_fit.params['smoothing_seasonal']:.4f}")
print(f"  AIC:            {hw_mul_fit.aic:.2f}")

hw_mul_pred = hw_mul_fit.forecast(steps=len(test))
rmse_hw_mul = np.sqrt(mean_squared_error(test, hw_mul_pred))
mae_hw_mul = mean_absolute_error(test, hw_mul_pred)
print(f"  RMSE: {rmse_hw_mul:.4f}")
print(f"  MAE:  {mae_hw_mul:.4f}")

# ==================== 7. 模型对比 ====================
print("\n" + "=" * 55)
print("模型对比总结")
print("=" * 55)
print(f"{'模型':<30} {'RMSE':>8} {'MAE':>8}")
print("-" * 50)
print(f"{'简单指数平滑 (SES)':<28} {rmse_ses:>8.4f} {mae_ses:>8.4f}")
print(f"{'双重指数平滑 (Holt)':<28} {rmse_holt:>8.4f} {mae_holt:>8.4f}")
print(f"{'Holt-Winters 加法':<28} {rmse_hw_add:>8.4f} {mae_hw_add:>8.4f}")
print(f"{'Holt-Winters 乘法':<28} {rmse_hw_mul:>8.4f} {mae_hw_mul:>8.4f}")

best = min(
    [("SES", rmse_ses), ("Holt", rmse_holt), ("HW加法", rmse_hw_add), ("HW乘法", rmse_hw_mul)],
    key=lambda x: x[1],
)
print(f"\n  最优模型: {best[0]} (RMSE={best[1]:.4f})")

# ==================== 8. 最优模型预测展示 ====================
print(f"\n--- HW 加法模型预测 vs 真实（前 10 个月） ---")
print(f"{'日期':>12}  {'真实':>8}  {'预测':>8}  {'误差':>8}")
for i in range(min(10, len(test))):
    print(f"  {str(test.index[i].date()):>10}  {test.values[i]:>8.3f}  "
          f"{hw_add_pred.values[i]:>8.3f}  {test.values[i] - hw_add_pred.values[i]:>8.3f}")

# ==================== 9. 全量模型 + 未来预测 ====================
print(f"\n--- 全量模型拟合 & 未来 12 个月预测 ---")
全模型 = ExponentialSmoothing(序列, trend="add", seasonal="add",
                              seasonal_periods=12,
                              initialization_method="estimated")
全拟合 = 全模型.fit(optimized=True)
未来预测 = 全拟合.forecast(steps=12)

print(f"{'月份':>12}  {'预测值':>8}")
for i in range(12):
    print(f"  {str(未来预测.index[i].date()):>10}  {未来预测.values[i]:>8.3f}")
