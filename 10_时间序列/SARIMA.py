"""
SARIMA 模型 —— 季节性自回归积分滑动平均模型

SARIMA(p, d, q)(P, D, Q, s) 在 ARIMA 基础上增加季节性分量：
- (P, D, Q): 季节性自回归、差分、滑动平均阶数
- s: 季节周期长度

建模流程：季节性分解 → 平稳性检验 → 确定季节周期 → SARIMA 拟合 → 预测评估
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 1. 生成具有季节性的合成时间序列 ====================
np.random.seed(42)
n = 365 * 2  # 两年日数据

t = np.arange(n)
趋势 = 0.02 * t
年度季节 = 3 * np.sin(2 * np.pi * t / 365)  # 年周期
周季节 = 0.8 * np.sin(2 * np.pi * t / 7)    # 周周期
噪声 = np.random.normal(0, 0.6, n)
数据 = 趋势 + 年度季节 + 周季节 + 噪声

序列 = pd.Series(数据, index=pd.date_range("2022-01-01", periods=n, freq="D"))
print("=" * 55)
print("SARIMA 季节性时间序列建模")
print("=" * 55)
print(f"序列长度: {n} 天（{序列.index[0].date()} ~ {序列.index[-1].date()}）")
print(f"均值: {序列.mean():.4f}, 标准差: {序列.std():.4f}")

# ==================== 2. 季节性分解 ====================
print("\n--- 季节性分解（周期=365） ---")
分解 = seasonal_decompose(序列, model="additive", period=365)
趋势成分 = 分解.trend.dropna()
季节成分 = 分解.seasonal.dropna()
残差成分 = 分解.resid.dropna()

print(f"  趋势成分范围: [{趋势成分.min():.3f}, {趋势成分.max():.3f}]")
print(f"  季节成分范围: [{季节成分.min():.3f}, {季节成分.max():.3f}]")
print(f"  残差标准差:   {残差成分.std():.4f}")

# ==================== 3. 平稳性检验 ====================
def adf检验(序列, 名称="序列"):
    result = adfuller(序列.dropna(), autolag="AIC")
    平稳 = result[1] < 0.05
    print(f"  ADF 检验 ({名称}): 统计量={result[0]:.4f}, p={result[1]:.4f} → {'平稳' if 平稳 else '非平稳'}")
    return 平稳

print("\n--- 平稳性检验 ---")
adf检验(序列, "原始序列")
差分1 = 序列.diff().dropna()
adf检验(差分1, "一阶差分")
差分季节 = 序列.diff(365).dropna()
adf检验(差分季节, "季节差分(s=365)")

# ==================== 4. SARIMA 模型拟合 ====================
print("\n--- 模型拟合 ---")
train_size = int(n * 0.8)
train, test = 序列[:train_size], 序列[train_size:]
print(f"训练集: {len(train)} 天, 测试集: {len(test)} 天")

# 参数选择策略：使用较小的参数组合避免过拟合
# 非季节性 (p,d,q) = (1,1,1), 季节性 (P,D,Q,s) = (1,1,1,7)
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 7)

print(f"参数: SARIMA{order}x{seasonal_order}")
模型 = SARIMAX(train, order=order, seasonal_order=seasonal_order,
               enforce_stationarity=False, enforce_invertibility=False)
拟合结果 = 模型.fit(disp=False, maxiter=200)

print(f"  AIC:  {拟合结果.aic:.2f}")
print(f"  BIC:  {拟合结果.bic:.2f}")

# ==================== 5. 残差诊断 ====================
残差 = 拟合结果.resid
print(f"\n--- 残差诊断 ---")
print(f"  均值:   {残差.mean():.6f}")
print(f"  标准差: {残差.std():.4f}")

lb = acorr_ljungbox(残差.dropna(), lags=[10, 20], return_df=True)
for lag, row in lb.iterrows():
    p = row["lb_pvalue"]
    print(f"  Ljung-Box (lag={lag}): p={p:.4f} → {'无自相关' if p > 0.05 else '存在自相关'}")

# ==================== 6. 预测与评估 ====================
forecast = 拟合结果.forecast(steps=len(test))
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100

print(f"\n--- 预测评估 ---")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  MAPE: {mape:.2f}%")

print(f"\n--- 预测 vs 真实（前 10 个） ---")
print(f"{'日期':>12}  {'真实':>8}  {'预测':>8}  {'误差':>8}")
for i in range(min(10, len(test))):
    print(f"  {str(test.index[i].date()):>10}  {test.values[i]:>8.3f}  {forecast.values[i]:>8.3f}  {test.values[i] - forecast.values[i]:>8.3f}")

# ==================== 7. 全序列拟合 + 未来预测 ====================
print("\n--- 全序列拟合 & 未来 30 天预测 ---")
全模型 = SARIMAX(序列, order=order, seasonal_order=seasonal_order,
                 enforce_stationarity=False, enforce_invertibility=False)
全拟合 = 全模型.fit(disp=False, maxiter=200)
未来预测 = 全拟合.forecast(steps=30)

print(f"{'日期':>12}  {'预测值':>8}")
for i in range(30):
    print(f"  {str(未来预测.index[i].date()):>10}  {未来预测.values[i]:>8.3f}")
