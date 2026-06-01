"""
PCA（主成分分析）—— 通过正交变换将数据投影到方差最大的方向，实现线性降维
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ===================== 1. 加载数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===================== 2. 保留全部成分 =====================
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

print("=== PCA（全部成分）===")
print(f"原始维度: {X_scaled.shape[1]}, 降维后: {X_pca_full.shape[1]}")
print(f"各主成分方差: {pca_full.explained_variance_.round(4)}")
print(f"各主成分方差占比: {pca_full.explained_variance_ratio_.round(4)}")
print(f"累积方差占比: {np.cumsum(pca_full.explained_variance_ratio_).round(4)}")
print(f"主成分方向（特征向量）:\n{pca_full.components_.round(4)}")

# ===================== 3. 指定降维维度 =====================
for n in [2, 3]:
    pca_n = PCA(n_components=n)
    X_pca = pca_n.fit_transform(X_scaled)
    print(f"\n=== PCA 降维到 {n} 维 ===")
    print(f"解释方差比: {pca_n.explained_variance_ratio_.round(4)}")
    print(f"累积解释方差: {pca_n.explained_variance_ratio_.sum():.4f}")
    print(f"降维后形状: {X_pca.shape}")

# ===================== 4. 按方差阈值选择维度 =====================
print("\n=== 按方差阈值选维度 ===")
for thresh in [0.80, 0.90, 0.95, 0.99]:
    pca_t = PCA(n_components=thresh)
    X_t = pca_t.fit_transform(X_scaled)
    print(f"  阈值={thresh}: 保留 {X_t.shape[1]} 维, "
          f"累积方差={pca_t.explained_variance_ratio_.sum():.4f}")

# ===================== 5. PCA 白化 =====================
# 白化后各主成分方差为 1，消除特征间相关性
pca_white = PCA(n_components=2, whiten=True)
X_white = pca_white.fit_transform(X_scaled)
print(f"\n=== PCA 白化 ===")
print(f"白化后各维度标准差: {X_white.std(axis=0).round(4)}")
print(f"白化后维度间相关系数:\n{np.corrcoef(X_white.T).round(4)}")

# ===================== 6. 增量 PCA（大数据集）=====================
from sklearn.decomposition import IncrementalPCA
print("\n=== 增量 PCA (IncrementalPCA) ===")
ipca = IncrementalPCA(n_components=2, batch_size=50)
X_ipca = ipca.fit_transform(X_scaled)
print(f"降维后形状: {X_ipca.shape}")
print(f"解释方差比: {ipca.explained_variance_ratio_.round(4)}")

# ===================== 7. PCA 作为预处理 =====================
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# 不降维
svm_full = SVC().fit(X_train, y_train)
acc_full = svm_full.score(X_test, y_test)
# PCA 降维到 2 维
pca_2 = PCA(n_components=2)
X_train_pca = pca_2.fit_transform(X_train)
X_test_pca = pca_2.transform(X_test)
svm_pca = SVC().fit(X_train_pca, y_train)
acc_pca = svm_pca.score(X_test_pca, y_test)
print(f"\n=== PCA 作为预处理 ===")
print(f"SVM 不降维准确率: {acc_full:.4f}")
print(f"SVM PCA→2D 准确率: {acc_pca:.4f}")

print("\n=== PCA 要点 ===")
print("- 无监督线性降维，寻找方差最大的投影方向")
print("- 必须先做特征缩放（PCA 对尺度敏感）")
print("- 可用于：降维可视化、去噪、加速后续模型训练")
print("- 核心参数：n_components（维度或方差阈值）")
print("- 缺点：只能捕捉线性结构，非线性数据考虑 t-SNE/UMAP")
