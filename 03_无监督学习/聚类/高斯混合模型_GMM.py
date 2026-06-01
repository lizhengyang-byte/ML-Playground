"""
高斯混合模型（GMM）—— 基于概率的聚类，假设数据由多个高斯分布混合生成
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ===================== 1. 生成数据 =====================
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=[1.0, 0.5, 1.5, 0.8], random_state=42)
X = StandardScaler().fit_transform(X)

# ===================== 2. 基础 GMM =====================
gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42, n_init=10)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)

print("=== 高斯混合模型 (K=4) ===")
print(f"各簇样本数: {np.bincount(labels)}")
print(f"各簇权重 (mixing proportion): {gmm.weights_.round(4)}")
print(f"各簇均值:\n{gmm.means_.round(4)}")
print(f"BIC: {gmm.bic(X):.2f}")
print(f"AIC: {gmm.aic(X):.2f}")
print(f"对数似然: {gmm.score(X) * len(X):.2f}")
print(f"Silhouette: {silhouette_score(X, labels):.4f}")
print(f"ARI: {adjusted_rand_score(y_true, labels):.4f}")

# ===================== 3. 协方差类型 =====================
print("\n=== 协方差类型对比 ===")
for cov_type in ["full", "tied", "diag", "spherical"]:
    gmm_c = GaussianMixture(n_components=4, covariance_type=cov_type, random_state=42, n_init=10)
    gmm_c.fit(X)
    labels_c = gmm_c.predict(X)
    sil = silhouette_score(X, labels_c)
    n_params = gmm_c._n_parameters()
    print(f"  {cov_type:>10}: BIC={gmm_c.bic(X):>10.2f}, AIC={gmm_c.aic(X):>10.2f}, "
          f"参数数={n_params}, Silhouette={sil:.4f}")
print("\n  full: 完整协方差（参数最多，最灵活）")
print("  tied: 所有簇共享同一协方差矩阵")
print("  diag: 对角协方差（各特征独立）")
print("  spherical: 各簇方差相同（最简单）")

# ===================== 4. 用 BIC/AIC 选簇数 =====================
print("\n=== BIC/AIC 选最优 K ===")
bic_scores = []
aic_scores = []
for k in range(1, 10):
    gmm_k = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=10)
    gmm_k.fit(X)
    bic_scores.append(gmm_k.bic(X))
    aic_scores.append(gmm_k.aic(X))
    print(f"  K={k}: BIC={gmm_k.bic(X):>10.2f}, AIC={gmm_k.aic(X):>10.2f}")

best_k_bic = np.argmin(bic_scores) + 1
best_k_aic = np.argmin(aic_scores) + 1
print(f"\n  BIC 最优 K: {best_k_bic}")
print(f"  AIC 最优 K: {best_k_aic}")

# ===================== 5. 软聚类（概率分配）=====================
print("\n=== 软聚类（概率分配）===")
gmm_final = GaussianMixture(n_components=4, random_state=42, n_init=10)
gmm_final.fit(X)
probs = gmm_final.predict_proba(X)
print("前 5 个样本的簇概率:")
for i in range(5):
    prob_str = ", ".join(f"p{k}={p:.3f}" for k, p in enumerate(probs[i]))
    print(f"  样本{i+1}: {prob_str} → 簇 {labels[i]}")

# ===================== 6. 生成新样本 =====================
print("\n=== 从 GMM 生成新样本 ===")
X_new, y_new = gmm_final.sample(5)
print("生成的 5 个新样本:")
for i in range(5):
    print(f"  样本{i+1}: {X_new[i].round(4)} (来自簇 {y_new[i]})")

print("\n=== GMM 要点 ===")
print("- 与 KMeans 的区别：GMM 是软聚类（概率分配），KMeans 是硬聚类（确定分配）")
print("- 簇可以是椭圆形（由协方差矩阵控制形状）")
print("- 用 EM 算法（期望最大化）迭代优化")
print("- BIC 惩罚模型复杂度，通常选 BIC 最小的 K")
print("- AIC 倾向选择更复杂的模型（更大的 K）")
print("- 对初始值敏感，建议增加 n_init")
