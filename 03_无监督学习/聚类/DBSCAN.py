"""
DBSCAN —— 基于密度的聚类算法，可发现任意形状的簇并识别噪声点
"""
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ===================== 1. 生成不同形状的数据 =====================
# 月亮形（非凸）
X_moon, y_moon = make_moons(n_samples=300, noise=0.1, random_state=42)
# 同心圆
from sklearn.datasets import make_circles
X_circle, y_circle = make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=42)
# 带噪声的球形
X_blob, y_blob = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

datasets = {
    "月亮形": (X_moon, y_moon),
    "同心圆": (X_circle, y_circle),
    "球形": (X_blob, y_blob),
}

# ===================== 2. eps 和 min_samples 参数搜索 =====================
print("=== DBSCAN 参数搜索（月亮形数据）===")
X = StandardScaler().fit_transform(X_moon)

best_sil = -1
best_params = None
for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    for min_samples in [3, 5, 7, 10]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        if n_clusters >= 2:
            # 只对非噪声点计算轮廓系数
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = silhouette_score(X[mask], labels[mask])
                if sil > best_sil:
                    best_sil = sil
                    best_params = (eps, min_samples)
                print(f"  eps={eps}, min_samples={min_samples}: 簇数={n_clusters}, "
                      f"噪声点={n_noise}, Silhouette={sil:.4f}")

print(f"\n最佳参数: eps={best_params[0]}, min_samples={best_params[1]}, Silhouette={best_sil:.4f}")

# ===================== 3. 各数据集上最佳参数 =====================
print("\n=== 不同数据集上的 DBSCAN ===")
for ds_name, (X_d, y_d) in datasets.items():
    X_d = StandardScaler().fit_transform(X_d)
    # 根据数据特点选参数
    if ds_name == "月亮形":
        db = DBSCAN(eps=0.2, min_samples=5)
    elif ds_name == "同心圆":
        db = DBSCAN(eps=0.3, min_samples=5)
    else:
        db = DBSCAN(eps=0.5, min_samples=5)

    labels = db.fit_predict(X_d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    ari = adjusted_rand_score(y_d, labels)
    print(f"  {ds_name}: 簇数={n_clusters}, 噪声点={n_noise}, ARI={ari:.4f}")

# ===================== 4. eps 距离图 =====================
print("\n=== eps 选择方法：k-距离图 ===")
print("对每个点计算到第 k 个最近邻的距离，排序后找"拐点"作为 eps")
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X)
distances, _ = nn.kneighbors(X)
# 取第 k 近邻距离
k_distances = np.sort(distances[:, -1])
print(f"  距离范围: [{k_distances.min():.4f}, {k_distances.max():.4f}]")
print(f"  中位数: {np.median(k_distances):.4f}")
print(f"  拐点建议 eps: 约 {np.percentile(k_distances, 90):.4f}")

# ===================== 5. min_samples 的影响 =====================
print("\n=== min_samples 对噪声点数的影响 ===")
X = StandardScaler().fit_transform(X_moon)
for ms in [3, 5, 7, 10, 15, 20]:
    db = DBSCAN(eps=0.2, min_samples=ms)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  min_samples={ms:>2}: 簇数={n_clusters}, 噪声点={n_noise} ({n_noise/len(X)*100:.1f}%)")

print("\n=== DBSCAN 要点 ===")
print("- 无需预设簇数，自动确定")
print("- 可发现任意形状的簇（不像 KMeans 只能发现球形簇）")
print("- 能识别噪声点（label=-1）")
print("- eps: 邻域半径；min_samples: 核心点的最小邻居数")
print("- 对参数敏感：eps 太大导致所有点合并，太小导致全是噪声")
print("- 不适合密度差异大的数据集")
print("- 核心概念：核心点（邻居足够多）→ 边界点（在核心点邻域内）→ 噪声点")
