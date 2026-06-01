"""
标签传播（Label Propagation）—— 基于图的半监督学习，通过图结构传播标签信息
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# ===================== 1. 加载数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 只保留少量标签（约 15%）
np.random.seed(42)
n_labeled = int(len(y_train) * 0.15)
labeled_idx = np.random.choice(len(y_train), n_labeled, replace=False)
y_train_semi = np.full_like(y_train, fill_value=-1)
y_train_semi[labeled_idx] = y_train[labeled_idx]

print(f"=== 数据划分 ===")
print(f"训练集: {len(X_train)} 样本 (有标签: {(y_train_semi >= 0).sum()}, "
      f"无标签: {(y_train_semi == -1).sum()})")

# ===================== 2. 基准：仅用有标签数据 =====================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[labeled_idx], y_train[labeled_idx])
print(f"仅用有标签数据 KNN: {knn.score(X_test, y_test):.4f}")

# ===================== 3. 标签传播算法 =====================
# kernel: 'knn' 或 'rbf'
# gamma: RBF 核参数
# n_neighbors: KNN 核的邻居数
lp = LabelPropagation(kernel="knn", n_neighbors=7, gamma=15, max_iter=1000)
lp.fit(X_train, y_train_semi)

y_pred = lp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n=== 标签传播 (LabelPropagation) ===")
print(f"测试准确率: {acc:.4f}")
print(f"标签传播矩阵形状: {label_distributions := lp.label_distributions_.shape}")
print(f"各类别分配概率（前 5 个样本）:")
print(lp.label_distributions_[:5].round(3))

# ===================== 4. 不同参数对比 =====================
print("\n=== n_neighbors 对比 ===")
for nn in [3, 5, 7, 10, 15, 30]:
    lp_n = LabelPropagation(kernel="knn", n_neighbors=nn, max_iter=1000)
    lp_n.fit(X_train, y_train_semi)
    acc = lp_n.score(X_test, y_test)
    print(f"  n_neighbors={nn:>2}: 测试准确率={acc:.4f}")

# ===================== 5. RBF 核 vs KNN 核 =====================
print("\n=== 核函数对比 ===")
for kernel, params in [("knn", {"n_neighbors": 7}), ("rbf", {"gamma": 15})]:
    lp_k = LabelPropagation(kernel=kernel, max_iter=1000, **params)
    lp_k.fit(X_train, y_train_semi)
    acc = lp_k.score(X_test, y_test)
    print(f"  kernel={kernel:>3}: 测试准确率={acc:.4f}")

# ===================== 6. LabelPropagation vs LabelSpreading =====================
from sklearn.semi_supervised import LabelSpreading
print("\n=== LabelPropagation vs LabelSpreading ===")
for name, model in [
    ("LabelPropagation", LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000)),
    ("LabelSpreading", LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=1000)),
]:
    model.fit(X_train, y_train_semi)
    acc = model.score(X_test, y_test)
    n_iter = model.n_iter_
    print(f"  {name:>20}: 测试准确率={acc:.4f}, 迭代次数={n_iter}")

# ===================== 7. 标签传播原理 =====================
print("\n=== 标签传播原理 ===")
print("1. 构建相似度图（每个样本是节点，边权 = 相似度）")
print("2. 标签沿边传播：每个节点的标签 = 邻居标签的加权平均")
print("3. 迭代直到收敛")
print("4. 最终每个节点的标签 = 收敛后的概率分布中最大的类别")

print("\n=== LabelPropagation vs LabelSpreading ===")
print("LabelPropagation: 标签完全转移（归一化到 [0,1]），可能不稳定")
print("LabelSpreading: 用 alpha 参数正则化，保留部分原始标签信息，更稳定")

print("\n=== 标签传播要点 ===")
print("- 优点：无需训练显式模型，利用数据几何结构")
print("- 适合：数据有明显的聚类结构，同类样本距离近")
print("- 对图构建参数（kernel, gamma, n_neighbors）敏感")
print("- 计算复杂度 O(n²)，大数据集较慢")
