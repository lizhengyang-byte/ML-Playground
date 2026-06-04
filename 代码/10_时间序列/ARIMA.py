"""
ARIMA 模型 —— 自回归积分滑动平均模型

ARIMA(p, d, q) 核心思想：
- AR(p): 自回归项，当前值与过去 p 个值线性相关
- I(d):  差分阶数，将非平稳序列变为平稳
- MA(q): 滑动平均项，当前值与过去 q 个误差线性相关

建模流程：平稳性检验 → 差分 → ACF/PACF 定阶 → 拟合 → 诊断 → 预测
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 1. 生成合成时间序列 ====================
np.random.seed(42)
n = 200

# 趋势 + 季节性 + 噪声
t = np.arange(n)
趋势 = 0.05 * t
季节性 = 2 * np.sin(2 * np.pi * t / 30)
噪声 = np.random.normal(0, 0.5, n)
数据 = 趋势 + 季节性 + 噪声

序列 = pd.Series(数据, index=pd.date_range("2020-01-01", periods=n, freq="D"))
print("=" * 50)
print("原始时间序列统计量：")
print(f"  样本数: {n}")
print(f"  均值: {序列.mean():.4f}")
print(f"  标准差: {序列.std():.4f}")
print("=" * 50)

# ==================== 2. 平稳性检验（ADF 检验） ====================
def adf检验(序列, 名称="序列"):
    """ADF 单位根检验，H0: 序列非平稳"""
    result = adfuller(序列.dropna(), autolag="AIC")
    print(f"\n--- ADF 检验: {名称} ---")
    print(f"  ADF 统计量:  {result[0]:.4f}")
    print(f"  p 值:        {result[1]:.4f}")
    print(f"  滞后阶数:    {result[2]}")
    for k, v in result[4].items():
        print(f"  临界值 ({k}): {v:.4f}")
    平稳 = result[1] < 0.05
    print(f"  结论: {'平稳' if 平稳 else '非平稳'}（显著性水平 0.05）")
    return 平稳

adf检验(序列)

# 差分处理
差分 = 序列.diff().dropna()
print(f"\n一阶差分后：")
adf检验(差分, "一阶差分")

# ==================== 3. ACF / PACF 定阶 ====================
acf值 = acf(差分, nlags=20)
pacf值 = pacf(差分, nlags=20)

print(f"\n--- ACF / PACF（前 10 阶） ---")
print(f"{'阶数':>4}  {'ACF':>8}  {'PACF':>8}")
for i in range(1, 11):
    print(f"  {i:>2}    {acf值[i]:>8.4f}  {pacf值[i]:>8.4f}")

# 简单定阶策略：找显著阶数（超过 95% 置信区间 ≈ 1.96/sqrt(n)）
边界 = 1.96 / np.sqrt(len(差分))
q_阶 = max([i for i in range(1, 11) if abs(acf值[i]) > 边界], default=1)
p_阶 = max([i for i in range(1, 11) if abs(pacf值[i]) > 边界], default=1)
d_阶 = 1  # 已做一阶差分
print(f"\n自动定阶建议: ARIMA({p_阶}, {d_阶}, {q_阶})")

# ==================== 4. 模型拟合 ====================
train_size = int(n * 0.8)
train, test = 序列[:train_size], 序列[train_size:]

模型 = ARIMA(train, order=(p_阶, d_阶, q_阶))
拟合结果 = 模型.fit()
print(f"\n--- 模型摘要（关键指标） ---")
print(f"  AIC:  {拟合结果.aic:.2f}")
print(f"  BIC:  {拟合结果.bic:.2f}")
print(f"  对数似然: {拟合结果.llf:.2f}")

# ==================== 5. 残差诊断 ====================
残差 = 拟合结果.resid
print(f"\n--- 残差统计 ---")
print(f"  均值:   {残差.mean():.6f}")
print(f"  标准差: {残差.std():.4f}")

# Ljung-Box 检验：H0 残差无自相关
lb_result = acorr_ljungbox(残差.dropna(), lags=[10], return_df=True)
p值 = lb_result["lb_pvalue"].values[0]
print(f"  Ljung-Box 检验 (lag=10): p 值 = {p值:.4f}")
print(f"  残差自相关: {'无显著自相关' if p值 > 0.05 else '存在自相关'}")

# ==================== 6. 预测与评估 ====================
forecast = 拟合结果.forecast(steps=len(test))
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100

print(f"\n--- 预测评估 ---")
print(f"  测试集长度: {len(test)}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  MAPE: {mape:.2f}%")

# 展示前 10 个预测值
print(f"\n--- 预测 vs 真实（前 10 个） ---")
print(f"{'日期':>12}  {'真实':>8}  {'预测':>8}  {'误差':>8}")
for i in range(min(10, len(test))):
    实际 = test.values[i]
    预测 = forecast.values[i]
    print(f"  {str(test.index[i].date()):>10}  {实际:>8.3f}  {预测:>8.3f}  {实际 - 预测:>8.3f}")

# ==================== 7. 全序列拟合后预测未来 ====================
全模型 = ARIMA(序列, order=(p_阶, d_阶, q_阶))
全拟合 = 全模型.fit()
未来预测 = 全拟合.forecast(steps=30)
print(f"\n--- 未来 30 天预测 ---")
print(f"{'日期':>12}  {'预测值':>8}")
for i in range(30):
    print(f"  {str(未来预测.index[i].date()):>10}  {未来预测.values[i]:>8.3f}")
