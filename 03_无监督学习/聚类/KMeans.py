"""
KMeans 聚类 —— 基于距离的划分聚类，将数据分为 K 个簇，每个样本分配到最近的质心
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# ===================== 1. 生成聚类数据 =====================
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===================== 2. 基础 KMeans =====================
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = kmeans.fit_predict(X)

print("=== KMeans 聚类 (K=4) ===")
print(f"簇标签分布: {np.bincount(labels)}")
print(f"惯性 (Inertia): {kmeans.inertia_:.2f}")  # 各样本到其质心距离的平方和
print(f"质心位置:\n{kmeans.cluster_centers_.round(4)}")
print(f"迭代次数: {kmeans.n_iter_}")

# ===================== 3. 评估指标 =====================
sil = silhouette_score(X, labels)
ch = calinski_harabasz_score(X, labels)
ari = adjusted_rand_score(y_true, labels)
print(f"\n=== 评估指标 ===")
print(f"轮廓系数 (Silhouette): {sil:.4f} (越接近 1 越好)")
print(f"Calinski-Harabasz 指数: {ch:.2f} (越大越好)")
print(f"调整兰德指数 (ARI): {ari:.4f} (越接近 1 越好)")

# ===================== 4. 肘部法则选 K =====================
print("\n=== 肘部法则 (Elbow Method) ===")
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))
    print(f"  K={k:>2}: Inertia={km.inertia_:>10.2f}, Silhouette={silhouette_score(X, km.labels_):.4f}")

# ===================== 5. KMeans++ 初始化 =====================
# KMeans++ 使初始质心尽可能分散，加速收敛，避免差的初始化
print("\n=== KMeans++ 初始化 ===")
kmeans_pp = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
labels_pp = kmeans_pp.fit_predict(X)
print(f"Inertia: {kmeans_pp.inertia_:.2f}")
print(f"Silhouette: {silhouette_score(X, labels_pp):.4f}")

# ===================== 6. 不同 n_init 的影响 =====================
print("\n=== n_init（初始化次数）影响 ===")
for n_init in [1, 5, 10, 20]:
    km_n = KMeans(n_clusters=4, n_init=n_init, random_state=42)
    km_n.fit(X)
    print(f"  n_init={n_init:>2}: Inertia={km_n.inertia_:.2f}")

# ===================== 7. 小批量 KMeans =====================
# 适合大数据集，每次迭代用 mini-batch 更新质心
from sklearn.cluster import MiniBatchKMeans
mbk = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42)
labels_mbk = mbk.fit_predict(X)
print(f"\n=== MiniBatchKMeans ===")
print(f"Inertia: {mbk.inertia_:.2f}")
print(f"Silhouette: {silhouette_score(X, labels_mbk):.4f}")

print("\n=== KMeans 要点 ===")
print("- 需要预设 K 值（簇数），可用肘部法则/轮廓系数辅助选择")
print("- 假设簇为球形且大小相近，不适合非凸形状的簇")
print("- 对初始质心敏感，KMeans++ 初始化缓解此问题")
print("- 时间复杂度 O(n×K×d×iter)，适合中等规模数据")
print("- MiniBatchKMeans 适合大数据集")
