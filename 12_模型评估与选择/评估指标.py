"""
评估指标 (Evaluation Metrics)

分类和回归模型的常用评估指标：
1. 分类: 准确率、精确率、召回率、F1、AUC
2. 回归: MSE、MAE、RMSE、R-squared
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

# ============================================================
# 分类评估指标
# ============================================================
print("=" * 60)
print("分类评估指标")
print("=" * 60)

X_cls, y_cls = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_classes=3, n_clusters_per_class=1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
)

model_cls = RandomForestClassifier(n_estimators=100, random_state=42)
model_cls.fit(X_train, y_train)
y_pred = model_cls.predict(X_test)
y_proba = model_cls.predict_proba(X_test)

# 手动计算混淆矩阵
n_classes = 3
cm = np.zeros((n_classes, n_classes), dtype=int)
for true, pred in zip(y_test, y_pred):
    cm[true, pred] += 1

print("混淆矩阵 (手动计算):")
print(f"{'预测->':>10}", end="")
for j in range(n_classes):
    print(f"  类{j}  ", end="")
print()
for i in range(n_classes):
    print(f"  真实类{i} ", end="")
    for j in range(n_classes):
        print(f"  {cm[i, j]:>3}  ", end="")
    print()

# 1. 准确率 (Accuracy)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
manual_accuracy = np.trace(cm) / cm.sum()
print(f"\n1. 准确率 (Accuracy)")
print(f"   sklearn: {accuracy:.4f}")
print(f"   手动计算: {manual_accuracy:.4f}")
print(f"   含义: 正确预测的样本数 / 总样本数")

# 2. 精确率、召回率、F1 (逐类别)
from sklearn.metrics import precision_score, recall_score, f1_score

print(f"\n2. 精确率 (Precision) / 召回率 (Recall) / F1")
for avg in ['macro', 'micro', 'weighted']:
    p = precision_score(y_test, y_pred, average=avg)
    r = recall_score(y_test, y_pred, average=avg)
    f1 = f1_score(y_test, y_pred, average=avg)
    print(f"   {avg:>8}: 精确率={p:.4f}, 召回率={r:.4f}, F1={f1:.4f}")

# 逐类别
print(f"\n   逐类别明细:")
print(f"   {'类别':>6} {'精确率':>8} {'召回率':>8} {'F1':>8} {'支持数':>8}")
for i in range(n_classes):
    p_i = precision_score(y_test, y_pred, average=None)[i]
    r_i = recall_score(y_test, y_pred, average=None)[i]
    f1_i = f1_score(y_test, y_pred, average=None)[i]
    support = np.sum(y_test == i)
    print(f"   类{i:>4} {p_i:>8.4f} {r_i:>8.4f} {f1_i:>8.4f} {support:>8}")

# 3. AUC (二分类和多分类)
from sklearn.metrics import roc_auc_score

print(f"\n3. AUC (Area Under ROC Curve)")

# 二分类简化示例
y_binary = (y_test == 0).astype(int)
y_proba_binary = y_proba[:, 0]
auc_binary = roc_auc_score(y_binary, y_proba_binary)
print(f"   二分类AUC (类0 vs 其他): {auc_binary:.4f}")

# 多分类 AUC (one-vs-rest)
auc_ovr = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
auc_ovo = roc_auc_score(y_test, y_proba, multi_class='ovo', average='macro')
print(f"   多分类AUC (One-vs-Rest, macro): {auc_ovr:.4f}")
print(f"   多分类AUC (One-vs-One, macro):  {auc_ovo:.4f}")

# ============================================================
# 回归评估指标
# ============================================================
print("\n" + "=" * 60)
print("回归评估指标")
print("=" * 60)

X_reg, y_reg = make_regression(
    n_samples=300, n_features=5, n_informative=3,
    noise=20, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

model_reg = LinearRegression()
model_reg.fit(X_train_r, y_train_r)
y_pred_r = model_reg.predict(X_test_r)
residuals = y_test_r - y_pred_r

# 1. 均方误差 (MSE)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_r, y_pred_r)
manual_mse = np.mean(residuals ** 2)
print(f"1. 均方误差 (MSE)")
print(f"   sklearn: {mse:.4f}")
print(f"   手动计算: {manual_mse:.4f}")

# 2. 均方根误差 (RMSE)
rmse = np.sqrt(mse)
print(f"\n2. 均方根误差 (RMSE)")
print(f"   RMSE = {rmse:.4f}")
print(f"   与目标变量同单位，更直观")

# 3. 平均绝对误差 (MAE)
mae = mean_absolute_error(y_test_r, y_pred_r)
manual_mae = np.mean(np.abs(residuals))
print(f"\n3. 平均绝对误差 (MAE)")
print(f"   sklearn: {mae:.4f}")
print(f"   手动计算: {manual_mae:.4f}")

# 4. R-squared (决定系数)
r2 = r2_score(y_test_r, y_pred_r)
manual_r2 = 1 - np.sum(residuals ** 2) / np.sum((y_test_r - y_test_r.mean()) ** 2)
print(f"\n4. R-squared (决定系数)")
print(f"   sklearn: {r2:.4f}")
print(f"   手动计算: {manual_r2:.4f}")
print(f"   含义: 模型解释了目标变量{r2*100:.1f}%的方差")

# 5. 残差分析
print(f"\n5. 残差分析")
print(f"   残差均值: {residuals.mean():.4f} (应接近0)")
print(f"   残差标准差: {residuals.std():.4f}")
print(f"   残差最小值: {residuals.min():.4f}")
print(f"   残差最大值: {residuals.max():.4f}")

# ============================================================
# 综合对比
# ============================================================
print("\n" + "=" * 60)
print("指标选择指南")
print("=" * 60)
print("""
分类任务:
  - 准确率: 类别平衡时使用
  - 精确率: 关注误报（FP）时使用（如垃圾邮件检测）
  - 召回率: 关注漏报（FN）时使用（如疾病诊断）
  - F1: 精确率和召回率的调和平均，类别不平衡时使用
  - AUC: 不依赖阈值，评估整体排序能力

回归任务:
  - MSE/RMSE: 对大误差敏感，适合需要惩罚大偏差的场景
  - MAE: 对异常值更鲁棒
  - R-squared: 无量纲，便于跨数据集比较
""")
