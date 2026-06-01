"""
逻辑回归 —— 经典二/多分类线性模型，通过 Sigmoid/Softmax 将线性输出映射为概率
"""
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ===================== 1. 加载数据 =====================
# 二分类场景
X_bin, y_bin = make_classification(
    n_samples=300, n_features=10, n_informative=5,
    n_classes=2, random_state=42
)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# 多分类场景（Iris）
iris = load_iris()
X_multi, y_multi = iris.data, iris.target
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# 特征缩放（逻辑回归对特征尺度敏感）
scaler_b = StandardScaler()
X_train_b = scaler_b.fit_transform(X_train_b)
X_test_b = scaler_b.transform(X_test_b)

scaler_m = StandardScaler()
X_train_m = scaler_m.fit_transform(X_train_m)
X_test_m = scaler_m.transform(X_test_m)

# ===================== 2. 二分类 =====================
# solver: 'lbfgs'(小数据), 'liblinear'(小数据+L1), 'saga'(大数据+弹性网)
# C 越小正则化越强；l1_ratio=0 为 L2, l1_ratio=1 为 L1
lr_bin = LogisticRegression(C=1.0, l1_ratio=0, solver="lbfgs", max_iter=1000, random_state=42)
lr_bin.fit(X_train_b, y_train_b)

y_pred_b = lr_bin.predict(X_test_b)
y_prob_b = lr_bin.predict_proba(X_test_b)

print("=== 二分类逻辑回归 ===")
print(f"训练集准确率: {lr_bin.score(X_train_b, y_train_b):.4f}")
print(f"测试集准确率: {accuracy_score(y_test_b, y_pred_b):.4f}")
print(f"模型系数: {lr_bin.coef_.round(4)}")
print(f"截距: {lr_bin.intercept_.round(4)}")
print(f"\n混淆矩阵:\n{confusion_matrix(y_test_b, y_pred_b)}")
print(f"\n分类报告:\n{classification_report(y_test_b, y_pred_b)}")

# ===================== 3. 多分类（Iris）=====================
lr_multi = LogisticRegression(C=1.0, l1_ratio=0, solver="lbfgs", max_iter=200, random_state=42)
lr_multi.fit(X_train_m, y_train_m)

y_pred_m = lr_multi.predict(X_test_m)
print("\n=== 多分类逻辑回归 (Iris) ===")
print(f"训练集准确率: {lr_multi.score(X_train_m, y_train_m):.4f}")
print(f"测试集准确率: {accuracy_score(y_test_m, y_pred_m):.4f}")
print(f"类别: {iris.target_names}")
print(f"模型系数形状: {lr_multi.coef_.shape} (类别数 × 特征数)")
print(f"\n分类报告:\n{classification_report(y_test_m, y_pred_m, target_names=iris.target_names)}")

# ===================== 4. 不同正则化对比 =====================
print("\n=== 正则化强度对比 (C 值) ===")
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    lr_c = LogisticRegression(C=C, l1_ratio=0, solver="lbfgs", max_iter=200, random_state=42)
    lr_c.fit(X_train_m, y_train_m)
    train_acc = lr_c.score(X_train_m, y_train_m)
    test_acc = lr_c.score(X_test_m, y_test_m)
    print(f"C={C:>6}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")

# ===================== 5. L1 正则化（特征选择）=====================
print("\n=== L1 正则化（稀疏系数）===")
lr_l1 = LogisticRegression(C=1.0, l1_ratio=1, solver="liblinear", max_iter=200, random_state=42)
lr_l1.fit(X_train_b, y_train_b)
n_nonzero = np.count_nonzero(lr_l1.coef_)
print(f"非零系数个数: {n_nonzero} / {lr_l1.coef_.shape[1]}")
print(f"系数: {lr_l1.coef_.round(4)}")
print(f"测试准确率: {lr_l1.score(X_test_b, y_test_b):.4f}")
