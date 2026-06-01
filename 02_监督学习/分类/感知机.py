"""
感知机 —— 最简单的神经网络，线性二分类器，支持在线学习
"""
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ===================== 1. 线性可分数据 =====================
X, y = make_classification(
    n_samples=300, n_features=10, n_informative=5,
    n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 感知机对特征缩放敏感
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== 2. 基础感知机 =====================
# Perceptron 本质上是随机梯度下降的线性模型
# eta0: 学习率, max_iter: 最大迭代次数
perc = Perceptron(eta0=1.0, max_iter=1000, random_state=42, early_stopping=True)
perc.fit(X_train, y_train)

print("=== 感知机分类 ===")
print(f"训练集准确率: {perc.score(X_train, y_train):.4f}")
print(f"测试集准确率: {perc.score(X_test, y_test):.4f}")
print(f"迭代次数: {perc.n_iter_}")
print(f"权重形状: {perc.coef_.shape}")
print(f"截距: {perc.intercept_}")

# ===================== 3. 学习率影响 =====================
print("\n=== 学习率 (eta0) 对比 ===")
for eta in [0.001, 0.01, 0.1, 1.0, 10.0]:
    perc_e = Perceptron(eta0=eta, max_iter=1000, random_state=42)
    perc_e.fit(X_train, y_train)
    print(f"  eta0={eta:>6}: 训练={perc_e.score(X_train, y_train):.4f}, "
          f"测试={perc_e.score(X_test, y_test):.4f}, 迭代={perc_e.n_iter_}")

# ===================== 4. 正则化（penalty）=====================
print("\n=== 正则化对比 ===")
for penalty, alpha in [("l2", 0.0001), ("l2", 0.01), ("l1", 0.0001), ("l1", 0.01), ("elasticnet", 0.0001)]:
    perc_p = Perceptron(penalty=penalty, alpha=alpha, random_state=42)
    perc_p.fit(X_train, y_train)
    test_acc = perc_p.score(X_test, y_test)
    n_nonzero = np.count_nonzero(perc_p.coef_)
    print(f"  {penalty:>12} alpha={alpha}: 测试={test_acc:.4f}, 非零权重={n_nonzero}/{perc_p.coef_.shape[1]}")

# ===================== 5. 在线学习（partial_fit）=====================
# 感知机支持增量学习，适合流式数据
print("\n=== 在线学习（partial_fit）===")
perc_online = Perceptron(eta0=0.1, max_iter=1, random_state=42)
classes = np.unique(y_train)

for epoch in range(10):
    perm = np.random.permutation(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]
    perc_online.partial_fit(X_shuffled, y_shuffled, classes=classes)
    acc = perc_online.score(X_test, y_test)
    if (epoch + 1) % 3 == 0:
        print(f"  Epoch {epoch+1:>2}: 测试准确率={acc:.4f}")

# ===================== 6. 多分类（Iris）=====================
print("\n=== 感知机在 Iris 上（多分类用 OvR）===")
iris = load_iris()
X_i, y_i = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X_i, y_i, test_size=0.2, random_state=42, stratify=y_i)
scaler_i = StandardScaler()
X_tr = scaler_i.fit_transform(X_tr)
X_te = scaler_i.transform(X_te)

perc_iris = Perceptron(max_iter=1000, random_state=42, early_stopping=True)
perc_iris.fit(X_tr, y_tr)
print(f"训练准确率: {perc_iris.score(X_tr, y_tr):.4f}")
print(f"测试准确率: {perc_iris.score(X_te, y_te):.4f}")
print(f"\n分类报告:\n{classification_report(y_te, perc_iris.predict(X_te), target_names=iris.target_names)}")

# ===================== 7. 感知机的局限性 =====================
X_xor, y_xor = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=2, random_state=42
)
perc_xor = Perceptron(max_iter=1000, random_state=42)
perc_xor.fit(X_xor, y_xor)
print(f"\n=== 感知机的局限（非线性可分数据）===")
print(f"测试准确率: {perc_xor.score(X_xor, y_xor):.4f}")
print("感知机只能处理线性可分问题，遇到 XOR 等非线性数据无法收敛")

print("\n=== 感知机要点 ===")
print("- 最早的神经网络模型（单层，无隐藏层）")
print("- 只能处理线性可分问题，非线性数据需要多层感知机（MLP）")
print("- 支持 partial_fit：可处理流式/在线学习场景")
print("- 收敛性：若数据线性可分，保证在有限步内收敛")
print("- 对学习率敏感：太小收敛慢，太大可能震荡不收敛")
print("- 与逻辑回归的区别：感知机直接输出类别，逻辑回归输出概率")
