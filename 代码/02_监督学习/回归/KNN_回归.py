"""
KNN 回归 —— K 近邻回归，预测值为 K 个最近邻样本目标值的加权平均
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ===================== 1. 生成数据 =====================
X, y = make_regression(n_samples=300, n_features=5, n_informative=3, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 基于距离，必须缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== 2. K 值选择 =====================
print("=== K 值选择 ===")
for k in [1, 3, 5, 7, 10, 15, 20, 30, 50]:
    knn = KNeighborsRegressor(n_neighbors=k, weights="uniform")
    knn.fit(X_train, y_train)
    train_r2 = knn.score(X_train, y_train)
    test_r2 = knn.score(X_test, y_test)
    print(f"  K={k:>2}: 训练R²={train_r2:.4f}, 测试R²={test_r2:.4f}")

# ===================== 3. 权重策略 =====================
print("\n=== 权重策略对比 ===")
for weights in ["uniform", "distance"]:
    knn_w = KNeighborsRegressor(n_neighbors=5, weights=weights)
    knn_w.fit(X_train, y_train)
    test_r2 = knn_w.score(X_test, y_test)
    y_pred = knn_w.predict(X_test)
    print(f"  {weights:>10}: R²={test_r2:.4f}, MSE={mean_squared_error(y_test, y_pred):.4f}")

# ===================== 4. 距离度量 =====================
print("\n=== 距离度量对比 ===")
for p, name in [(1, "曼哈顿"), (2, "欧氏"), (3, "闵可夫斯基-3")]:
    knn_d = KNeighborsRegressor(n_neighbors=5, metric="minkowski", p=p)
    knn_d.fit(X_train, y_train)
    test_r2 = knn_d.score(X_test, y_test)
    print(f"  {name:>12} (p={p}): R²={test_r2:.4f}")

# ===================== 5. 最终模型与评估 =====================
print("\n=== 最终模型 (K=5, distance) ===")
knn_final = KNeighborsRegressor(n_neighbors=5, weights="distance", p=2)
knn_final.fit(X_train, y_train)
y_pred = knn_final.predict(X_test)

print(f"测试 R²: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

print("\n=== 预测值 vs 真实值（前 10 个）===")
for i in range(10):
    print(f"  真实={y_test[i]:>8.2f}, 预测={y_pred[i]:>8.2f}")

print("\n=== KNN 回归要点 ===")
print("- 预测值 = K 个邻居目标值的平均（uniform）或加权平均（distance）")
print("- K=1 时完全拟合训练数据（过拟合），K 大时趋于全局均值（欠拟合）")
print("- 特征缩放必须！距离计算受特征尺度影响")
print("- 不学习显式模型，预测时计算所有训练样本距离，速度慢")
print("- 适合低维、小数据集；高维大数据效果差（维度灾难）")
