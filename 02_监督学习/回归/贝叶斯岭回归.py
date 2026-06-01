"""
贝叶斯岭回归 —— 在岭回归基础上引入贝叶斯框架，提供参数的不确定性估计
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ===================== 1. 生成数据 =====================
np.random.seed(42)
X, y = make_regression(n_samples=200, n_features=5, n_informative=3, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== 2. 三种回归对比 =====================
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
br = BayesianRidge().fit(X_train, y_train)

print("=== 线性回归 vs 岭回归 vs 贝叶斯岭回归 ===")
print(f"{'方法':>20} {'测试 R²':>10}")
print(f"{'线性回归':>20} {lr.score(X_test, y_test):>10.4f}")
print(f"{'岭回归':>20} {ridge.score(X_test, y_test):>10.4f}")
print(f"{'贝叶斯岭回归':>20} {br.score(X_test, y_test):>10.4f}")

# ===================== 3. 不确定性估计 =====================
# predict 返回均值和标准差，标准差反映预测不确定性
y_pred_mean, y_pred_std = br.predict(X_test, return_std=True)
print(f"\n=== 预测不确定性 ===")
print(f"{'样本':>6} {'真实值':>10} {'预测均值':>10} {'预测标准差':>12} {'95%区间':>20}")
for i in range(10):
    lo = y_pred_mean[i] - 1.96 * y_pred_std[i]
    hi = y_pred_mean[i] + 1.96 * y_pred_std[i]
    print(f"  {i+1:>4} {y_test[i]:>10.2f} {y_pred_mean[i]:>10.2f} "
          f"{y_pred_std[i]:>12.4f} [{lo:>8.2f}, {hi:>8.2f}]")

# ===================== 4. 贝叶斯岭 vs 普通岭回归 =====================
print("\n=== 贝叶斯岭 vs 岭回归 ===")
print(f"贝叶斯岭 α (精度): {br.alpha_:.6f}")
print(f"贝叶斯岭 λ (噪声): {br.lambda_:.6f}")
print(f"贝叶斯岭 系数: {br.coef_.round(4)}")
print(f"岭回归   系数: {ridge.coef_.round(4)}")

# ===================== 5. 超参数调优 =====================
print("\n=== 关键超参数 ===")
print("BayesianRidge 的 alpha_1/alpha_2/lambda_1/lambda_2 是先验分布的超参数")
print(f"  alpha_1={br.alpha_1}, alpha_2={br.alpha_2} (控制 α 的 Gamma 先验)")
print(f"  lambda_1={br.lambda_1}, lambda_2={br.lambda_2} (控制 λ 的 Gamma 先验)")

# 手动调参
for alpha_1 in [1e-6, 1e-4, 1e-2]:
    br_t = BayesianRidge(alpha_1=alpha_1, max_iter=300)
    br_t.fit(X_train, y_train)
    print(f"  alpha_1={alpha_1}: R²={br_t.score(X_test, y_test):.4f}, "
          f"alpha_={br_t.alpha_:.4f}, lambda_={br_t.lambda_:.4f}")

# ===================== 6. 概率预测（多输出）=====================
# BayesianRidge 不支持原生多输出，需用 MultiOutputRegressor 包装
from sklearn.multioutput import MultiOutputRegressor

print("\n=== 多目标回归（MultiOutputRegressor + BayesianRidge）===")
y_multi = np.column_stack([y, y * 0.5 + np.random.randn(len(y)) * 5])
br_multi = MultiOutputRegressor(BayesianRidge())
br_multi.fit(X_train, y_multi[:len(X_train)])
y_pred_multi = br_multi.predict(X_test)
print(f"多输出预测形状: {y_pred_multi.shape}")
print(f"前 5 个样本预测值:\n{y_pred_multi[:5].round(2)}")

print("\n=== 贝叶斯岭回归要点 ===")
print("- 将权重视为随机变量，用贝叶斯推断估计后验分布")
print("- 自动学习正则化参数 alpha（权重精度）和 lambda（噪声精度）")
print("- 提供预测不确定性估计（标准差），适合需要置信区间的场景")
print("- 系数被 收缩 向 0（类似岭回归），但收缩强度由数据自适应决定")
print("- 适合：小样本、需要不确定性估计、不想手动调正则化参数")
