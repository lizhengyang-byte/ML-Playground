"""
KNN 分类 —— K 近邻算法，基于距离的懒惰学习，不显式训练模型
"""
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ===================== 1. 生成数据 =====================
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# KNN 基于距离，必须缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== 2. 不同 K 值对比 =====================
print("=== K 值选择（月亮形数据）===")
for k in [1, 3, 5, 7, 10, 15, 20, 30]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)
    print(f"  K={k:>2}: 训练={train_acc:.4f}, 测试={test_acc:.4f}")

# ===================== 3. 不同距离度量 =====================
print("\n=== 距离度量对比 ===")
# p=1: 曼哈顿距离, p=2: 欧氏距离, p=3: 闵可夫斯基距离
for p, name in [(1, "曼哈顿"), (2, "欧氏"), (3, "闵可夫斯基")]:
    knn_d = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=p)
    knn_d.fit(X_train, y_train)
    test_acc = knn_d.score(X_test, y_test)
    print(f"  {name:>10} (p={p}): 测试准确率={test_acc:.4f}")

# ===================== 4. 权重策略 =====================
print("\n=== 权重策略对比 ===")
for weights in ["uniform", "distance"]:
    knn_w = KNeighborsClassifier(n_neighbors=5, weights=weights)
    knn_w.fit(X_train, y_train)
    test_acc = knn_w.score(X_test, y_test)
    print(f"  {weights:>10}: 测试准确率={test_acc:.4f}")
# uniform: 每个邻居投票权重相同
# distance: 距离越近权重越大（1/distance）

# ===================== 5. Iris 数据集 =====================
print("\n=== Iris 数据集 ===")
iris = load_iris()
X_i, y_i = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X_i, y_i, test_size=0.2, random_state=42, stratify=y_i)
scaler_i = StandardScaler()
X_tr = scaler_i.fit_transform(X_tr)
X_te = scaler_i.transform(X_te)

knn_iris = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2)
knn_iris.fit(X_tr, y_tr)
print(f"训练准确率: {knn_iris.score(X_tr, y_tr):.4f}")
print(f"测试准确率: {knn_iris.score(X_te, y_te):.4f}")
print(f"\n分类报告:\n{classification_report(y_te, knn_iris.predict(X_te), target_names=iris.target_names)}")

# ===================== 6. 预测概率 =====================
print("=== 预测概率示例 ===")
y_prob = knn_iris.predict_proba(X_te[:3])
y_pred = knn_iris.predict(X_te[:3])
for i in range(3):
    probs = ", ".join(f"{p:.3f}" for p in y_prob[i])
    print(f"  样本{i+1}: 预测={iris.target_names[y_pred[i]]}, 概率=[{probs}], "
          f"真实={iris.target_names[y_te[i]]}")

print("\n=== KNN 要点 ===")
print("- K 值：K 越大决策边界越平滑，但 K 过大会欠拟合")
print("- 特征缩放：必须！距离计算受特征尺度直接影响")
print("- 距离度量：欧氏距离最常用，高维数据可考虑余弦相似度")
print("- 时间复杂度：预测时 O(n×d)，大数据集应使用 KD-Tree 或 Ball-Tree 加速")
print("- 不适合高维稀疏数据（维度灾难：高维空间距离失去意义）")
