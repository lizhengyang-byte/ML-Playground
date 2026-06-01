"""
层次聚类 —— 自底向上（凝聚）或自顶向下（分裂）构建聚类层次树（树状图）
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# ===================== 1. 生成数据 =====================
X, y_true = make_blobs(n_samples=150, centers=4, cluster_std=1.0, random_state=42)
X = StandardScaler().fit_transform(X)

# ===================== 2. 凝聚层次聚类（AgglomerativeClustering）=====================
print("=== 凝聚层次聚类 ===")
# linkage: "ward"(最小化方差), "complete"(最大距离), "average"(平均距离), "single"(最小距离)
for linkage_method in ["ward", "complete", "average", "single"]:
    ac = AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
    labels = ac.fit_predict(X)
    sil = silhouette_score(X, labels)
    ari = adjusted_rand_score(y_true, labels)
    print(f"  linkage={linkage_method:>8}: Silhouette={sil:.4f}, ARI={ari:.4f}")

# ===================== 3. Ward 链接法详细分析 =====================
print("\n=== Ward 链接法 (n_clusters=4) ===")
ac_ward = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels_ward = ac_ward.fit_predict(X)
print(f"簇标签分布: {np.bincount(labels_ward)}")
print(f"Silhouette: {silhouette_score(X, labels_ward):.4f}")

# ===================== 4. 不同簇数对比 =====================
print("\n=== 不同 n_clusters 对比 ===")
for k in range(2, 8):
    ac_k = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_k = ac_k.fit_predict(X)
    sil = silhouette_score(X, labels_k)
    print(f"  K={k}: Silhouette={sil:.4f}")

# ===================== 5. 树状图（用 scipy）=====================
print("\n=== 树状图（Dendrogram）===")
# 计算层次链接矩阵
Z = linkage(X, method="ward")
print(f"链接矩阵形状: {Z.shape}")
print(f"最后 5 次合并:")
for i in range(-5, 0):
    print(f"  合并 {int(Z[i, 0]):>5} 和 {int(Z[i, 1]):>5} → {int(Z[i, 0])+int(Z[i, 1]):>5}, "
          f"距离={Z[i, 2]:.4f}")

# 用 fcluster 从链接矩阵切割出簇
for k in [2, 3, 4, 5]:
    labels_f = fcluster(Z, t=k, criterion="maxclust")
    sil = silhouette_score(X, labels_f)
    print(f"  fcluster K={k}: Silhouette={sil:.4f}")

# ===================== 6. distance_threshold 自动确定簇数 =====================
print("\n=== distance_threshold 自动切割 ===")
ac_auto = AgglomerativeClustering(n_clusters=None, distance_threshold=3.0, linkage="ward")
labels_auto = ac_auto.fit_predict(X)
n_clusters = len(set(labels_auto))
print(f"阈值=3.0 时自动确定簇数: {n_clusters}")

# 不同阈值
for thresh in [1.0, 2.0, 3.0, 5.0, 10.0]:
    ac_t = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh, linkage="ward")
    labels_t = ac_t.fit_predict(X)
    nc = len(set(labels_t))
    print(f"  阈值={thresh:>5}: 簇数={nc}")

print("\n=== 层次聚类要点 ===")
print("- 凝聚（自底向上）：每个点初始为一个簇，逐步合并最相似的簇")
print("- 分裂（自顶向下）：所有点在一个簇，逐步分裂")
print("- linkage 决定簇间距离的计算方式")
print("- Ward 法：合并后总方差增加最小的两个簇（默认推荐）")
print("- Single 法：容易出现链式效应（chaining）")
print("- 优点：不需要预设 K（可用阈值切割）、可产生树状图")
print("- 缺点：时间复杂度 O(n²~n³)、合并/分裂不可撤销（贪心）")
