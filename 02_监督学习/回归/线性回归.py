"""
线性回归 —— 通过最小化残差平方和拟合线性关系 y = Xw + b
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ===================== 1. 生成回归数据 =====================
X, y = make_regression(n_samples=200, n_features=5, n_informative=3, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== 2. 线性回归拟合 =====================
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

print("=== 线性回归 ===")
print(f"系数: {lr.coef_.round(4)}")
print(f"截距: {lr.intercept_:.4f}")
print(f"\n训练集: MSE={mean_squared_error(y_train, y_pred_train):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}, "
      f"MAE={mean_absolute_error(y_train, y_pred_train):.4f}, "
      f"R²={r2_score(y_train, y_pred_train):.4f}")
print(f"测试集: MSE={mean_squared_error(y_test, y_pred_test):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}, "
      f"MAE={mean_absolute_error(y_test, y_pred_test):.4f}, "
      f"R²={r2_score(y_test, y_pred_test):.4f}")

# ===================== 3. 预测值 vs 真实值 =====================
print("\n=== 前 10 个预测值 vs 真实值 ===")
for i in range(10):
    print(f"  真实={y_test[i]:>8.2f}, 预测={y_pred_test[i]:>8.2f}, 残差={y_test[i]-y_pred_test[i]:>8.2f}")

# ===================== 4. 评估指标详解 =====================
print("\n=== 评估指标 ===")
print(f"MSE  (均方误差):     {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"RMSE (均方根误差):   {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"MAE  (平均绝对误差): {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"R²   (决定系数):     {r2_score(y_test, y_pred_test):.4f}")

# 手动计算 R²
ss_res = np.sum((y_test - y_pred_test) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2_manual = 1 - ss_res / ss_tot
print(f"R² (手动计算): {r2_manual:.4f}")

# ===================== 5. 多元线性回归系数解读 =====================
print("\n=== 系数解读 ===")
print("每个系数表示：该特征每增加 1 个单位，目标变量平均变化多少")
for i, coef in enumerate(lr.coef_):
    print(f"  特征{i}: 系数={coef:.4f} (每增加 1 单位, y 变化 {coef:.4f})")

# ===================== 6. 注意事项 =====================
print("\n=== 线性回归要点 ===")
print("- 假设：线性关系、误差独立同分布、无多重共线性")
print("- 优点：简单可解释、训练速度快")
print("- 缺点：对异常值敏感、无法捕捉非线性关系")
print("- 如果特征间高度相关，考虑岭回归或 Lasso 回归")
print("- 如果存在非线性关系，考虑多项式回归或树模型")
