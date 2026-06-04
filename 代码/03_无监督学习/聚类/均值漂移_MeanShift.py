"""
均值漂移（Mean Shift）—— 基于核密度估计的聚类算法，自动确定簇数
"""
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ===================== 1. 生成数据 =====================
X_moon, y_moon = make_moons(n_samples=200, noise=0.1, random_state=42)
X_blob, y_blob = make_blobs(n_samples=200, centers=3, random_state=42)
X = StandardScaler().fit_transform(X_moon)

# ===================== 2. 自动估计带宽（bandwidth）=====================
# bandwidth 是核函数的带宽，控制邻域大小
bw_estimated = estimate_bandwidth(X, quantile=0.2, n_samples=200)
print(f"=== 自动估计带宽 ===")
print(f"估计的 bandwidth: {bw_estimated:.4f}")

# ===================== 3. 基础 MeanShift =====================
ms = MeanShift(bandwidth=bw_estimated, bin_seeding=True)
labels = ms.fit_predict(X)
n_clusters = len(set(labels))

print(f"\n=== MeanShift 聚类 ===")
print(f"自动确定的簇数: {n_clusters}")
print(f"各簇样本数: {np.bincount(labels)}")
print(f"聚类中心:\n{ms.cluster_centers_.round(4)}")
if n_clusters >= 2:
    print(f"Silhouette: {silhouette_score(X, labels):.4f}")
print(f"ARI: {adjusted_rand_score(y_moon, labels):.4f}")

# ===================== 4. 不同带宽的影响 =====================
print("\n=== bandwidth 参数影响 ===")
for bw_mult in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    bw = bw_estimated * bw_mult
    ms_b = MeanShift(bandwidth=bw, bin_seeding=True)
    labels_b = ms_b.fit_predict(X)
    nc = len(set(labels_b))
    print(f"  bandwidth={bw:.3f} (×{bw_mult}): 簇数={nc}, "
          f"各簇大小={np.bincount(labels_b)}")

# ===================== 5. bin_seeding 加速 =====================
print("\n=== bin_seeding 加速 ===")
import time
for bin_seed in [False, True]:
    t0 = time.time()
    ms_t = MeanShift(bandwidth=bw_estimated, bin_seeding=bin_seed)
    ms_t.fit(X)
    t = time.time() - t0
    print(f"  bin_seeding={bin_seed}: 耗时={t:.4f}s, 簇数={len(set(ms_t.labels_))}")

# ===================== 6. 球形数据测试 =====================
print("\n=== MeanShift 在球形数据上 ===")
X_b = StandardScaler().fit_transform(X_blob)
bw_b = estimate_bandwidth(X_b, quantile=0.2)
ms_blob = MeanShift(bandwidth=bw_b, bin_seeding=True)
labels_blob = ms_blob.fit_predict(X_b)
nc_blob = len(set(labels_blob))
print(f"带宽: {bw_b:.4f}")
print(f"自动确定簇数: {nc_blob}")
print(f"真实簇数: 3")
if nc_blob >= 2:
    print(f"Silhouette: {silhouette_score(X_b, labels_blob):.4f}")

print("\n=== Mean Shift 要点 ===")
print("- 核心思想：每个点向其邻域内所有点的均值方向漂移，最终收敛到密度峰值")
print("- 无需预设簇数，由数据密度自动决定")
print("- bandwidth 是唯一关键参数：太大 → 所有点合并；太小 → 每个点一个簇")
print("- bin_seeding=True 可加速：只在网格点上初始化，减少计算量")
print("- 时间复杂度 O(n²)，大数据集较慢")
print("- 适合中等规模、密度均匀的簇")
