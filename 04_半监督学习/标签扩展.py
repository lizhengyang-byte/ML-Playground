"""
标签扩展（Label Spreading）—— 标签传播的正则化版本，保留部分原始标签信息
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# ===================== 1. 加载数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 只保留少量标签（约 10%）
np.random.seed(42)
n_labeled = int(len(y_train) * 0.10)
labeled_idx = np.random.choice(len(y_train), n_labeled, replace=False)
y_train_semi = np.full_like(y_train, fill_value=-1)
y_train_semi[labeled_idx] = y_train[labeled_idx]

print(f"=== 数据划分 ===")
print(f"训练集: {len(X_train)} 样本 (有标签: {(y_train_semi >= 0).sum()}, "
      f"无标签: {(y_train_semi == -1).sum()})")

# ===================== 2. 基准 =====================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[labeled_idx], y_train[labeled_idx])
print(f"仅用有标签数据 KNN: {knn.score(X_test, y_test):.4f}")

# ===================== 3. Label Spreading =====================
# alpha: 正则化参数，控制保留多少原始标签信息
# alpha=0: 完全传播（≈LabelPropagation）
# alpha=1: 完全保留原始标签（不传播）
ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=1000)
ls.fit(X_train, y_train_semi)

y_pred = ls.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n=== 标签扩展 (alpha=0.2) ===")
print(f"测试准确率: {acc:.4f}")
print(f"迭代次数: {ls.n_iter_}")

# ===================== 4. alpha 参数影响 =====================
print("\n=== alpha 参数影响 ===")
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
    ls_a = LabelSpreading(kernel="knn", n_neighbors=7, alpha=alpha, max_iter=1000)
    ls_a.fit(X_train, y_train_semi)
    acc = ls_a.score(X_test, y_test)
    print(f"  alpha={alpha}: 测试准确率={acc:.4f}, 迭代={ls_a.n_iter_}")

# ===================== 5. 不同标签比例 =====================
print("\n=== 不同标签比例对比 ===")
for ratio in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
    n_lab = int(len(y_train) * ratio)
    idx = np.random.choice(len(y_train), n_lab, replace=False)
    y_semi = np.full_like(y_train, fill_value=-1)
    y_semi[idx] = y_train[idx]

    ls_r = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=1000)
    ls_r.fit(X_train, y_semi)
    acc = ls_r.score(X_test, y_test)
    print(f"  标签比例={ratio:>4.0%}: 测试准确率={acc:.4f}")

# ===================== 6. 不同核参数 =====================
print("\n=== n_neighbors 对比 ===")
for nn in [3, 5, 7, 10, 15, 30]:
    ls_n = LabelSpreading(kernel="knn", n_neighbors=nn, alpha=0.2, max_iter=1000)
    ls_n.fit(X_train, y_train_semi)
    acc = ls_n.score(X_test, y_test)
    print(f"  n_neighbors={nn:>2}: 测试准确率={acc:.4f}")

# ===================== 7. 分类报告 =====================
print(f"\n=== 分类报告 ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n=== 标签扩展原理 ===")
print("Y = (1-α) × Y_propagated + α × Y_initial")
print("- α 控制"保持原始标签"vs"接受传播标签"的平衡")
print("- α 越小 → 传播越强，类边界越平滑")
print("- α 越大 → 越保守，接近只用原始标签")
print("- 比 LabelPropagation 更稳定（不容易出现数值问题）")

print("\n=== 标签扩展要点 ===")
print("- 与 LabelPropagation 的核心区别：alpha 正则化")
print("- alpha=0.2 是常用的默认值")
print("- 适合：数据有清晰的流形/聚类结构")
print("- 需要足够的有标签数据启动传播")
