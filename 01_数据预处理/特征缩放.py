"""
特征缩放 —— 标准化、归一化等常见缩放方法对比
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
)

# ===================== 1. 加载示例数据 =====================
iris = load_iris()
X = iris.data
feature_names = iris.feature_names
print("=== 原始数据统计 ===")
print(f"特征: {feature_names}")
print(f"各特征范围:")
for i, name in enumerate(feature_names):
    print(f"  {name}: [{X[:, i].min():.1f}, {X[:, i].max():.1f}], "
          f"均值={X[:, i].mean():.2f}, 标准差={X[:, i].std():.2f}")

# ===================== 2. Z-score 标准化（StandardScaler）=====================
# (x - μ) / σ，使每个特征均值为 0，标准差为 1
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
print(f"\n=== StandardScaler (Z-score 标准化) ===")
for i, name in enumerate(feature_names):
    print(f"  {name}: [{X_std[:, i].min():.2f}, {X_std[:, i].max():.2f}], "
          f"均值={X_std[:, i].mean():.4f}, 标准差={X_std[:, i].std():.4f}")

# ===================== 3. Min-Max 归一化（MinMaxScaler）=====================
# (x - min) / (max - min)，缩放到 [0, 1]
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X)
print(f"\n=== MinMaxScaler (归一化到 [0,1]) ===")
for i, name in enumerate(feature_names):
    print(f"  {name}: [{X_mm[:, i].min():.2f}, {X_mm[:, i].max():.2f}]")

# 也可指定范围，如 [-1, 1]
scaler_mm2 = MinMaxScaler(feature_range=(-1, 1))
X_mm2 = scaler_mm2.fit_transform(X)
print(f"\n=== MinMaxScaler (归一化到 [-1,1]) ===")
for i, name in enumerate(feature_names):
    print(f"  {name}: [{X_mm2[:, i].min():.2f}, {X_mm2[:, i].max():.2f}]")

# ===================== 4. 鲁棒缩放（RobustScaler）=====================
# 使用中位数和 IQR 缩放，对异常值不敏感
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)
print(f"\n=== RobustScaler (基于中位数和 IQR) ===")
for i, name in enumerate(feature_names):
    print(f"  {name}: [{X_robust[:, i].min():.2f}, {X_robust[:, i].max():.2f}], "
          f"中位数={np.median(X_robust[:, i]):.4f}")

# ===================== 5. 最大绝对值缩放（MaxAbsScaler）=====================
# x / |max|，缩放到 [-1, 1]，适合稀疏数据
scaler_abs = MaxAbsScaler()
X_abs = scaler_abs.fit_transform(X)
print(f"\n=== MaxAbsScaler (最大绝对值缩放) ===")
for i, name in enumerate(feature_names):
    print(f"  {name}: [{X_abs[:, i].min():.2f}, {X_abs[:, i].max():.2f}]")

# ===================== 6. L2 归一化（Normalizer）=====================
# 将每个样本缩放为单位范数（L2 范数为 1），适合文本分类等
normalizer = Normalizer(norm="l2")
X_norm = normalizer.fit_transform(X)
print(f"\n=== Normalizer (L2 归一化) ===")
print(f"每个样本的 L2 范数（前 5 个）: {np.linalg.norm(X_norm[:5], axis=1).round(4)}")

# ===================== 7. 幂变换（PowerTransformer）=====================
# 使数据更接近正态分布（Yeo-Johnson 或 Box-Cox）
pt_yj = PowerTransformer(method="yeo-johnson")
X_yj = pt_yj.fit_transform(X)
print(f"\n=== PowerTransformer (Yeo-Johnson) ===")
for i, name in enumerate(feature_names):
    print(f"  {name}: 均值={X_yj[:, i].mean():.4f}, 标准差={X_yj[:, i].std():.4f}")

# Box-Cox 要求数据为正数
pt_bc = PowerTransformer(method="box-cox")
X_bc = pt_bc.fit_transform(X)  # Iris 数据均为正数
print(f"\n=== PowerTransformer (Box-Cox) ===")
for i, name in enumerate(feature_names):
    print(f"  {name}: 均值={X_bc[:, i].mean():.4f}, 标准差={X_bc[:, i].std():.4f}")

# ===================== 8. 分位数变换（QuantileTransformer）=====================
# 映射到均匀分布或正态分布，对异常值最鲁棒
qt_norm = QuantileTransformer(output_distribution="normal", random_state=42)
X_qt = qt_norm.fit_transform(X)
print(f"\n=== QuantileTransformer (映射到正态分布) ===")
for i, name in enumerate(feature_names):
    print(f"  {name}: 均值={X_qt[:, i].mean():.4f}, 标准差={X_qt[:, i].std():.4f}")

# ===================== 9. 缩放方法选择指南 =====================
print("\n=== 缩放方法选择指南 ===")
print("1. StandardScaler: 默认首选，适合大多数算法（SVM/KNN/PCA/逻辑回归）")
print("2. MinMaxScaler: 需要固定范围时使用（神经网络、图像像素）")
print("3. RobustScaler: 数据含较多异常值时使用")
print("4. MaxAbsScaler: 稀疏数据保持稀疏性")
print("5. Normalizer: 文本分类、余弦相似度计算")
print("6. PowerTransformer: 特征严重偏态时，先正态化再缩放")
print("7. QuantileTransformer: 异常值极多或分布极偏时使用")

# 注意事项
print("\n=== 注意事项 ===")
print("- 缩放器应在训练集上 fit，然后分别 transform 训练集和测试集")
print("- 避免数据泄露：绝不能用测试集信息来 fit 缩放器")
print("- 树模型（决策树/随机森林/XGBoost）通常不需要特征缩放")
print("- 距离-based 模型（KNN/SVM/线性回归/逻辑回归）强烈建议缩放")
