"""
混淆矩阵 (Confusion Matrix)

混淆矩阵展示分类器在每个类别上的预测情况，是理解模型错误模式的基础工具：
1. 二分类混淆矩阵
2. 多分类混淆矩阵
3. 归一化混淆矩阵
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 生成数据
X, y = make_classification(
    n_samples=600, n_features=10, n_informative=5,
    n_classes=3, n_clusters_per_class=1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ============================================================
# 1. 基础混淆矩阵
# ============================================================
print("=" * 60)
print("1. 基础混淆矩阵")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵形状: {cm.shape}")
print(f"矩阵内容:\n{cm}")

# 解释混淆矩阵
n_classes = cm.shape[0]
class_names = [f"类{i}" for i in range(n_classes)]

print(f"\n{'':>10}", end="")
for j in range(n_classes):
    print(f"{'预测'+class_names[j]:>12}", end="")
print()
for i in range(n_classes):
    print(f"{'真实'+class_names[i]:>10}", end="")
    for j in range(n_classes):
        print(f"{cm[i, j]:>12}", end="")
    print()

# ============================================================
# 2. 逐类别分析
# ============================================================
print("\n" + "=" * 60)
print("2. 逐类别分析")
print("=" * 60)

for i in range(n_classes):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp  # 其他类被预测为该类
    fn = cm[i, :].sum() - tp  # 该类被预测为其他类
    tn = cm.sum() - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{class_names[i]}:")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  精确率(Precision): {precision:.4f}")
    print(f"  召回率(Recall):    {recall:.4f}")
    print(f"  F1分数:            {f1:.4f}")

# ============================================================
# 3. 归一化混淆矩阵
# ============================================================
print("\n" + "=" * 60)
print("3. 归一化混淆矩阵")
print("=" * 60)

# 按行归一化 (每个真实类别的预测分布)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("按行归一化 (每行=1.0, 展示每个真实类别的预测分布):")
print(f"{'':>10}", end="")
for j in range(n_classes):
    print(f"{'预测'+class_names[j]:>10}", end="")
print()
for i in range(n_classes):
    print(f"{'真实'+class_names[i]:>10}", end="")
    for j in range(n_classes):
        print(f"{cm_normalized[i, j]:>10.3f}", end="")
    print()

print("\n解读:")
for i in range(n_classes):
    diag = cm_normalized[i, i]
    print(f"  {class_names[i]}: {diag*100:.1f}%被正确分类", end="")
    errors = []
    for j in range(n_classes):
        if j != i and cm_normalized[i, j] > 0.01:
            errors.append(f"{class_names[j]}({cm_normalized[i, j]*100:.1f}%)")
    if errors:
        print(f", 主要误判为: {', '.join(errors)}")
    else:
        print()

# ============================================================
# 4. 不同模型的混淆矩阵对比
# ============================================================
print("\n" + "=" * 60)
print("4. 不同模型的混淆矩阵对比")
print("=" * 60)

models = {
    '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred_m = m.predict(X_test)
    cm_m = confusion_matrix(y_test, y_pred_m)
    accuracy = np.trace(cm_m) / cm_m.sum()

    print(f"\n{name} (准确率={accuracy:.4f}):")
    print(f"{'':>10}", end="")
    for j in range(n_classes):
        print(f"{'预测'+class_names[j]:>10}", end="")
    print()
    for i in range(n_classes):
        print(f"{'真实'+class_names[i]:>10}", end="")
        for j in range(n_classes):
            print(f"{cm_m[i, j]:>10}", end="")
        print()

# ============================================================
# 5. 使用classification_report
# ============================================================
print("\n" + "=" * 60)
print("5. classification_report (综合报告)")
print("=" * 60)

print(classification_report(y_test, y_pred, target_names=class_names))

# ============================================================
# 6. 错误样本分析
# ============================================================
print("=" * 60)
print("6. 错误样本分析")
print("=" * 60)

misclassified = y_test != y_pred
n_errors = misclassified.sum()
n_total = len(y_test)

print(f"总样本数: {n_total}")
print(f"错误预测数: {n_errors}")
print(f"错误率: {n_errors/n_total:.4f}")

# 各类别的错误分布
print(f"\n各类别错误数:")
for i in range(n_classes):
    errors_i = np.sum((y_test == i) & misclassified)
    total_i = np.sum(y_test == i)
    print(f"  {class_names[i]}: {errors_i}/{total_i} ({errors_i/total_i*100:.1f}%)")

# 最常被混淆的类别对
print(f"\n最常被混淆的类别对:")
confused_pairs = []
for i in range(n_classes):
    for j in range(n_classes):
        if i != j:
            confused_pairs.append((class_names[i], class_names[j], cm[i, j]))
confused_pairs.sort(key=lambda x: -x[2])
for true_cls, pred_cls, count in confused_pairs[:5]:
    if count > 0:
        print(f"  真实={true_cls} -> 误判为{pred_cls}: {count}次")
