"""
SVM 分类 —— 支持向量机，通过最大间隔超平面分类，核技巧处理非线性问题
"""
from sklearn.datasets import make_classification, make_moons, make_circles, load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ===================== 1. 不同类型的数据 =====================
# 线性可分
X_linear, y_linear = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, random_state=42
)
# 月亮形（非线性）
X_moon, y_moon = make_moons(n_samples=200, noise=0.2, random_state=42)
# 同心圆（非线性）
X_circle, y_circle = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

datasets = {
    "线性可分": (X_linear, y_linear),
    "月亮形": (X_moon, y_moon),
    "同心圆": (X_circle, y_circle),
}

# ===================== 2. 不同核函数对比 =====================
kernels = ["linear", "poly", "rbf", "sigmoid"]

print("=== 不同核函数在不同数据上的表现 ===\n")
for ds_name, (X, y) in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"数据: {ds_name}")
    for kernel in kernels:
        svm = SVC(kernel=kernel, C=1.0, random_state=42)
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        n_sv = svm.n_support_.sum()
        print(f"  {kernel:>8}: 准确率={acc:.4f}, 支持向量数={n_sv}")
    print()

# ===================== 3. C 参数（正则化）=====================
print("=== C 参数对比（线性核）===")
X, y = X_linear, y_linear
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    svm_c = SVC(kernel="linear", C=C, random_state=42)
    svm_c.fit(X_train, y_train)
    train_acc = svm_c.score(X_train, y_train)
    test_acc = svm_c.score(X_test, y_test)
    print(f"  C={C:>6}: 训练={train_acc:.4f}, 测试={test_acc:.4f}, 支持向量={svm_c.n_support_.sum()}")

# ===================== 4. Iris 多分类 =====================
print("\n=== Iris 多分类（RBF 核）===")
iris = load_iris()
X_i, y_i = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X_i, y_i, test_size=0.2, random_state=42, stratify=y_i)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

svm_iris = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
svm_iris.fit(X_tr, y_tr)
print(f"训练准确率: {svm_iris.score(X_tr, y_tr):.4f}")
print(f"测试准确率: {svm_iris.score(X_te, y_te):.4f}")
print(f"\n分类报告:\n{classification_report(y_te, svm_iris.predict(X_te), target_names=iris.target_names)}")

# ===================== 5. gamma 参数 =====================
print("=== gamma 参数对比（RBF 核）===")
for gamma in ["scale", "auto", 0.01, 0.1, 1.0, 10.0]:
    svm_g = SVC(kernel="rbf", C=10.0, gamma=gamma, random_state=42)
    svm_g.fit(X_tr, y_tr)
    train_acc = svm_g.score(X_tr, y_tr)
    test_acc = svm_g.score(X_te, y_te)
    print(f"  gamma={str(gamma):>6}: 训练={train_acc:.4f}, 测试={test_acc:.4f}")

# ===================== 6. 概率预测 =====================
print("\n=== 概率预测（需 probability=True）===")
svm_prob = SVC(kernel="rbf", C=10.0, probability=True, random_state=42)
svm_prob.fit(X_tr, y_tr)
y_prob = svm_prob.predict_proba(X_te[:5])
for i in range(5):
    probs = ", ".join(f"{p:.3f}" for p in y_prob[i])
    print(f"  样本{i+1}: 预测={iris.target_names[svm_prob.predict(X_te[i:i+1])[0]]}, "
          f"概率=[{probs}]")

print("\n=== SVM 要点 ===")
print("- 核函数选择：线性核用于高维稀疏，RBF 用于一般非线性")
print("- C 参数：越大间隔越小（硬间隔），越容易过拟合")
print("- gamma 参数：RBF 核的宽度，越大决策边界越弯曲，越容易过拟合")
print("- 特征缩放对 SVM 非常重要（距离计算受尺度影响）")
print("- 训练复杂度约 O(n²~n³)，大数据集考虑 SGD 或近似方法")
