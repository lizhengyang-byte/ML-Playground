"""
SVD（奇异值分解）—— 矩阵分解降维，PCA 的底层算法基础，可用于推荐系统等
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.linalg import svd

# ===================== 1. 加载数据 =====================
iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

# ===================== 2. NumPy/SciPy 完整 SVD =====================
print("=== 完整 SVD 分解 ===")
U, S, Vt = svd(X_scaled, full_matrices=False)
print(f"X 形状: {X_scaled.shape}")
print(f"U 形状: {U.shape}, S 形状: {S.shape}, Vt 形状: {Vt.shape}")
print(f"奇异值: {S.round(4)}")
print(f"奇异值占比: {(S**2 / np.sum(S**2)).round(4)}")
print(f"累积占比: {np.cumsum(S**2 / np.sum(S**2)).round(4)}")

# 用截断 SVD 重建
k = 2
X_reduced = U[:, :k] @ np.diag(S[:k])
X_reconstructed = X_reduced @ Vt[:k, :]
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"\n截断到 k={k}: 重建误差 MSE = {reconstruction_error:.4f}")

# ===================== 3. TruncatedSVD（sklearn）=====================
# 与 PCA 类似，但不需要中心化，适合稀疏矩阵
print("\n=== TruncatedSVD（sklearn）===")
svd_model = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd_model.fit_transform(X_scaled)
print(f"降维后形状: {X_svd.shape}")
print(f"解释方差比: {svd_model.explained_variance_ratio_.round(4)}")
print(f"累积解释方差: {svd_model.explained_variance_ratio_.sum():.4f}")

# ===================== 4. 不同 k 值 =====================
print("\n=== 不同 k 值对比 ===")
for k in range(1, 5):
    svd_k = TruncatedSVD(n_components=k, random_state=42)
    X_k = svd_k.fit_transform(X_scaled)
    print(f"  k={k}: 解释方差={svd_k.explained_variance_ratio_.round(4)}, "
          f"累积={svd_k.explained_variance_ratio_.sum():.4f}")

# ===================== 5. SVD 与 PCA 的关系 =====================
print("\n=== SVD vs PCA ===")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
svd2 = TruncatedSVD(n_components=2, random_state=42)
X_svd2 = svd2.fit_transform(X_scaled)

# PCA = 中心化 + SVD
print(f"PCA 方差比:  {pca.explained_variance_ratio_.round(4)}")
print(f"SVD 方差比:  {svd2.explained_variance_ratio_.round(4)}")
print("PCA 先做中心化，然后执行 SVD；TruncatedSVD 跳过中心化")

# ===================== 6. SVD 用于推荐系统模拟 =====================
print("\n=== SVD 用于推荐系统模拟 ===")
# 构造用户-物品评分矩阵（稀疏）
np.random.seed(42)
n_users, n_items = 5, 6
R = np.random.randint(0, 6, (n_users, n_items)).astype(float)
# 随机遮盖部分评分（模拟缺失）
mask = np.random.rand(n_users, n_items) < 0.3
R_missing = R.copy()
R_missing[mask] = np.nan

print("原始评分矩阵:")
print(R.astype(int))
print(f"缺失率: {mask.mean():.0%}")

# SVD 降维后重建（填充缺失值）
R_filled = np.nan_to_num(R_missing, nan=0)
svd_rec = TruncatedSVD(n_components=2, random_state=42)
U_mat = svd_rec.fit_transform(R_filled)
V_mat = svd_rec.components_
R_reconstructed = U_mat @ V_mat

print(f"\n重建后评分矩阵:")
print(R_reconstructed.round(2))

# ===================== 7. 截断 SVD 用于文本 =====================
print("\n=== 截断 SVD 用于文本降维（潜在语义分析 LSA）===")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

documents = [
    "机器学习是人工智能的一个分支",
    "深度学习是机器学习的子领域",
    "自然语言处理涉及文本分析",
    "计算机视觉处理图像和视频",
    "神经网络是深度学习的基础",
]
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(documents)
print(f"TF-IDF 矩阵形状: {X_tfidf.shape}")

lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X_tfidf)
print(f"LSA 降维后: {X_lsa.shape}")
print(f"解释方差比: {lsa.explained_variance_ratio_.round(4)}")

print("\n=== SVD 要点 ===")
print("- X = U × Σ × Vᵀ，奇异值 Σ 反映各维度的重要性")
print("- 截断 SVD：保留前 k 个奇异值，实现降维")
print("- 与 PCA 的区别：SVD 不需要中心化，可处理稀疏矩阵")
print("- 适用于：文本降维（LSA）、推荐系统、图像压缩")
print("- TruncatedSVD 适合稀疏数据，PCA 适合稠密数据")
