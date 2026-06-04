"""
模型可解释性 (Model Interpretability)

理解模型"为什么"做出某个预测,是信任和调试模型的关键：
1. 特征重要性（树模型）
2. 排列重要性（模型无关）
3. 部分依赖分析
4. LIME 近似解释
5. SHAP 值解释
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import accuracy_score

# ============================================================
# 1. 准备数据和模型
# ============================================================
print("=" * 60)
print("1. 准备数据和模型")
print("=" * 60)

iris = load_iris()
feature_names = list(iris.feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"随机森林测试准确率: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

# 训练梯度提升
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
print(f"梯度提升测试准确率: {accuracy_score(y_test, gb.predict(X_test)):.4f}")

# ============================================================
# 2. 特征重要性（树模型内置）
# ============================================================
print("\n" + "=" * 60)
print("2. 特征重要性（树模型内置）")
print("=" * 60)

print("\n随机森林特征重要性 (基于不纯度减少):")
importances_rf = rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]

for i, idx in enumerate(indices_rf):
    bar_len = int(importances_rf[idx] / importances_rf.max() * 30)
    bar = '#' * bar_len
    print(f"  {feature_names[idx]:<18} {bar} {importances_rf[idx]:.4f}")

print("\n梯度提升特征重要性:")
importances_gb = gb.feature_importances_
indices_gb = np.argsort(importances_gb)[::-1]

for i, idx in enumerate(indices_gb):
    bar_len = int(importances_gb[idx] / importances_gb.max() * 30)
    bar = '#' * bar_len
    print(f"  {feature_names[idx]:<18} {bar} {importances_gb[idx]:.4f}")

# ============================================================
# 3. 排列重要性（模型无关方法）
# ============================================================
print("\n" + "=" * 60)
print("3. 排列重要性（模型无关方法）")
print("=" * 60)

# 对随机森林计算排列重要性
perm_result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

print("\n随机森林排列重要性:")
for i in perm_result.importances_mean.argsort()[::-1]:
    bar_len = int(perm_result.importances_mean[i] / perm_result.importances_mean.max() * 30)
    bar = '#' * bar_len
    print(f"  {feature_names[i]:<18} {bar} {perm_result.importances_mean[i]:.4f} "
          f"(±{perm_result.importances_std[i]:.4f})")

# 对逻辑回归也计算（说明模型无关性）
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

perm_lr = permutation_importance(lr, X_test, y_test, n_repeats=10, random_state=42)
print("\n逻辑回归排列重要性:")
for i in perm_lr.importances_mean.argsort()[::-1]:
    bar_len = int(perm_lr.importances_mean[i] / max(perm_lr.importances_mean.max(), 0.001) * 30)
    bar = '#' * bar_len
    print(f"  {feature_names[i]:<18} {bar} {perm_lr.importances_mean[i]:.4f}")

# ============================================================
# 4. 部分依赖分析
# ============================================================
print("\n" + "=" * 60)
print("4. 部分依赖分析")
print("=" * 60)

# 手动计算部分依赖（文本输出）
def partial_dependence_manual(model, X, feature_idx, feature_name, grid_resolution=10):
    """手动计算单个特征的部分依赖值"""
    feature_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), grid_resolution)
    pdp_values = []

    for val in feature_values:
        X_temp = X.copy()
        X_temp[:, feature_idx] = val
        preds = model.predict_proba(X_temp)
        pdp_values.append(preds.mean(axis=0))

    pdp_values = np.array(pdp_values)

    print(f"\n  {feature_name} 部分依赖:")
    max_bar = 25
    for i, val in enumerate(feature_values):
        # 显示每个类别的平均预测概率
        probs = pdp_values[i]
        bar = '#' * int(probs.max() * max_bar)
        print(f"    {val:>8.2f} |{bar:<{max_bar}} {probs.round(3)}")

    return feature_values, pdp_values

for feat_idx, feat_name in enumerate(feature_names):
    vals, pdp = partial_dependence_manual(rf, X_test, feat_idx, feat_name)

# ============================================================
# 5. LIME 近似解释
# ============================================================
print("\n" + "=" * 60)
print("5. LIME 近似解释（简化实现）")
print("=" * 60)

def simple_lime(model, X_train, instance, n_samples=500, feature_names=None):
    """
    简化版LIME: 在实例附近采样,用采样数据训练线性模型来近似解释
    """
    # 在实例附近生成扰动样本
    noise = np.random.randn(n_samples, X_train.shape[1]) * 0.1
    X_sample = instance + noise * np.std(X_train, axis=0)

    # 获取模型预测
    y_sample = model.predict_proba(X_sample)

    # 用线性模型拟合
    from sklearn.linear_model import Ridge
    n_classes = y_sample.shape[1]
    coefs = []

    for c in range(n_classes):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_sample, y_sample[:, c])
        coefs.append(ridge.coef_)

    coefs = np.array(coefs)

    # 输出解释
    print(f"\n  解释实例 (特征值: {instance.round(4)}):")
    if feature_names:
        # 对每个类别显示最重要的特征
        for c in range(n_classes):
            top_idx = np.argsort(np.abs(coefs[c]))[::-1][:3]
            print(f"    类别{c}:")
            for idx in top_idx:
                direction = "正向" if coefs[c][idx] > 0 else "负向"
                print(f"      {feature_names[idx]:<18} 系数={coefs[c][idx]:>+.4f} ({direction})")

    return coefs

# 解释一个测试样本
instance = X_test[0]
coefs = simple_lime(rf, X_train, instance, feature_names=feature_names)

# ============================================================
# 6. SHAP 值解释
# ============================================================
print("\n" + "=" * 60)
print("6. SHAP 值解释")
print("=" * 60)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[SKIP] shap 未安装，跳过本示例")
    import sys; sys.exit(0)
    HAS_SHAP = False
    print("shap 未安装, 使用简化实现")
    print("安装: pip install shap")

if HAS_SHAP:
    # 使用TreeExplainer（针对树模型优化）
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test[:5])

    print("\nSHAP值 (前5个测试样本):")
    if isinstance(shap_values, list):
        # 多分类: shap_values是列表,每个元素对应一个类别
        for sample_idx in range(min(3, len(X_test))):
            print(f"\n  样本{sample_idx+1}:")
            for c in range(len(shap_values)):
                print(f"    类别{c}:", end="")
                for f_idx, fname in enumerate(feature_names):
                    print(f" {fname}={shap_values[c][sample_idx][f_idx]:>+.4f}", end="")
                print()
    else:
        for sample_idx in range(min(3, len(X_test))):
            print(f"\n  样本{sample_idx+1}:")
            for f_idx, fname in enumerate(feature_names):
                print(f"    {fname}: {shap_values[sample_idx][f_idx]:>+.4f}")
else:
    # 简化的SHAP近似
    print("\n简化版特征贡献分析:")
    # 使用排列重要性+符号来近似SHAP
    instance = X_test[0]
    print(f"\n  样本特征值:")
    for f_idx, fname in enumerate(feature_names):
        print(f"    {fname}: {instance[f_idx]:.4f}")

    # 基于模型系数的近似贡献
    print(f"\n  预测类别: {rf.predict(instance.reshape(1, -1))[0]}")
    print(f"  各类别概率: {rf.predict_proba(instance.reshape(1, -1))[0].round(4)}")

# ============================================================
# 7. 可解释性方法对比
# ============================================================
print("\n" + "=" * 60)
print("7. 可解释性方法对比")
print("=" * 60)

print("""
方法              适用模型        优点                  局限
----              --------        ----                  ----
特征重要性(内置)   树模型          快速、直观            仅限树模型,对相关特征偏倚
排列重要性         任意模型        模型无关,理论扎实      计算慢,特征独立性假设
部分依赖           任意模型        展示特征-预测关系      只展示边际效应,忽略交互
LIME              任意模型        局部近似,直观          采样不稳定,解释可能不一致
SHAP              任意模型(树快)  博弈论基础,全局+局部   计算开销大,需要shap库

实践建议:
  1. 先用特征重要性做全局筛选
  2. 用SHAP深入分析单个预测
  3. 用部分依赖理解特征效应
  4. 不同方法结果不一致时,多角度验证
""")
