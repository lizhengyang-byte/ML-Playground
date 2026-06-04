"""
SVR（支持向量回归）—— SVM 的回归版本，通过 ε-不敏感损失函数拟合
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ===================== 1. 生成数据 =====================
X, y = make_regression(n_samples=200, n_features=5, n_informative=3, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVR 基于距离，必须缩放
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ===================== 2. 不同核函数对比 =====================
print("=== SVR 不同核函数 ===")
for kernel in ["linear", "rbf", "poly", "sigmoid"]:
    svr = SVR(kernel=kernel, C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train_scaled)
    y_pred_scaled = svr.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    r2 = r2_score(y_test, y_pred)
    print(f"  {kernel:>8}: R²={r2:.4f}, 支持向量数={svr.support_.shape[0]}")

# ===================== 3. 线性 SVR 详细分析 =====================
print("\n=== 线性 SVR 详细分析 ===")
svr_lin = SVR(kernel="linear", C=1.0, epsilon=0.1)
svr_lin.fit(X_train, y_train_scaled)
y_pred_scaled = svr_lin.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

print(f"测试 R²: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"系数: {svr_lin.coef_.round(4)}")
print(f"支持向量数: {svr_lin.support_.shape[0]}/{len(X_train)}")

# ===================== 4. C 参数（正则化）=====================
print("\n=== C 参数对比 ===")
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    svr_c = SVR(kernel="rbf", C=C, epsilon=0.1)
    svr_c.fit(X_train, y_train_scaled)
    y_pred_s = svr_c.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    print(f"  C={C:>6}: R²={r2_score(y_test, y_pred):.4f}, "
          f"支持向量={svr_c.support_.shape[0]}")

# ===================== 5. epsilon 参数 =====================
print("\n=== epsilon 参数对比 ===")
for eps in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
    svr_e = SVR(kernel="rbf", C=10.0, epsilon=eps)
    svr_e.fit(X_train, y_train_scaled)
    y_pred_s = svr_e.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    print(f"  eps={eps:>5}: R²={r2_score(y_test, y_pred):.4f}, "
          f"支持向量={svr_e.support_.shape[0]}")

# ===================== 6. RBF 核 gamma 参数 =====================
print("\n=== gamma 参数对比 ===")
for gamma in ["scale", "auto", 0.01, 0.1, 1.0, 10.0]:
    svr_g = SVR(kernel="rbf", C=10.0, gamma=gamma)
    svr_g.fit(X_train, y_train_scaled)
    y_pred_s = svr_g.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    print(f"  gamma={str(gamma):>6}: R²={r2_score(y_test, y_pred):.4f}")

print("\n=== SVR 要点 ===")
print("- ε-不敏感损失：预测误差在 ε 范围内不计入损失（管道模型）")
print("- C 越大 → 容忍误差越小，可能过拟合")
print("- epsilon 越大 → 管道越宽，支持向量越少，模型越简单")
print("- 特征和目标变量都需要缩放")
print("- 训练复杂度 O(n²~n³)，不适合大数据集")
