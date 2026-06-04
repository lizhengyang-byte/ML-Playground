"""
谱聚类 —— 基于图论的聚类，利用数据的相似度图的拉普拉斯矩阵特征分解进行降维后再聚类
"""
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ===================== 1. 生成不同形状的数据 =====================
X_moon, y_moon = make_moons(n_samples=300, noise=0.1, random_state=42)
X_circle, y_circle = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
X_blob, y_blob = make_blobs(n_samples=300, centers=3, random_state=42)

# ===================== 2. 基础谱聚类 =====================
X = StandardScaler().fit_transform(X_moon)
sc = SpectralClustering(n_clusters=2, affinity="rbf", random_state=42)
labels = sc.fit_predict(X)

print("=== 谱聚类（月亮形数据）===")
print(f"簇标签分布: {np.bincount(labels)}")
print(f"Silhouette: {silhouette_score(X, labels):.4f}")
print(f"ARI: {adjusted_rand_score(y_moon, labels):.4f}")

# ===================== 3. affinity（相似度度量）=====================
print("\n=== affinity 对比 ===")
X_c = StandardScaler().fit_transform(X_circle)
for aff in ["rbf", "nearest_neighbors"]:
    if aff == "nearest_neighbors":
        sc_a = SpectralClustering(n_clusters=2, affinity=aff, n_neighbors=10, random_state=42)
    else:
        sc_a = SpectralClustering(n_clusters=2, affinity=aff, random_state=42)
    labels_a = sc_a.fit_predict(X_c)
    sil = silhouette_score(X_c, labels_a)
    ari = adjusted_rand_score(y_circle, labels_a)
    print(f"  affinity={aff:>17}: Silhouette={sil:.4f}, ARI={ari:.4f}")

# ===================== 4. gamma 参数（RBF 核）=====================
print("\n=== gamma 参数对比（RBF 核）===")
for gamma in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    sc_g = SpectralClustering(n_clusters=2, affinity="rbf", gamma=gamma, random_state=42)
    labels_g = sc_g.fit_predict(X)
    sil = silhouette_score(X, labels_g)
    ari = adjusted_rand_score(y_moon, labels_g)
    print(f"  gamma={gamma:>5}: Silhouette={sil:.4f}, ARI={ari:.4f}")

# ===================== 5. n_neighbors（nearest_neighbors affinity）=====================
print("\n=== n_neighbors 参数对比 ===")
for nn in [5, 10, 15, 20, 30]:
    sc_n = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                               n_neighbors=nn, random_state=42)
    labels_n = sc_n.fit_predict(X)
    sil = silhouette_score(X, labels_n)
    print(f"  n_neighbors={nn:>2}: Silhouette={sil:.4f}")

# ===================== 6. 不同数据集对比 =====================
print("\n=== 不同数据集上谱聚类 vs KMeans ===")
from sklearn.cluster import KMeans
datasets = {"月亮形": (X_moon, y_moon, 2), "同心圆": (X_circle, y_circle, 2), "球形": (X_blob, y_blob, 3)}
for name, (X_d, y_d, k) in datasets.items():
    X_d = StandardScaler().fit_transform(X_d)
    sc_d = SpectralClustering(n_clusters=k, affinity="rbf", random_state=42)
    km_d = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_sc = sc_d.fit_predict(X_d)
    labels_km = km_d.fit_predict(X_d)
    ari_sc = adjusted_rand_score(y_d, labels_sc)
    ari_km = adjusted_rand_score(y_d, labels_km)
    print(f"  {name}: 谱聚类 ARI={ari_sc:.4f}, KMeans ARI={ari_km:.4f}")

print("\n=== 谱聚类要点 ===")
print("1. 构建相似度图（affinity matrix）")
print("2. 计算拉普拉斯矩阵的前 K 个特征向量")
print("3. 在特征向量空间中执行 KMeans 聚类")
print("- 优点：能发现非凸形状的簇（不像 KMeans 只能发现球形簇）")
print("- 缺点：计算量大 O(n³)、对参数（gamma/n_neighbors）敏感")
print("- affinity='rbf' 的 gamma 控制相似度衰减速度")
print("- affinity='nearest_neighbors' 用 K 近邻图构建相似度矩阵")
