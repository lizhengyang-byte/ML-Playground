"""
UMAP（均匀流形近似与投影）—— 比 t-SNE 更快、更好保持全局结构的非线性降维
需要安装: pip install umap-learn
"""
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# 尝试导入 umap，不可用则跳过
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP 未安装，请运行: pip install umap-learn")
    print("以下演示仅展示 UMAP 的使用方式和参数说明\n")

# ===================== 1. 加载数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)

# ===================== 2. 基础 UMAP =====================
if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X)

    print("=== UMAP 降维（Iris: 4D → 2D）===")
    print(f"降维后形状: {X_umap.shape}")
    for c in range(3):
        mask = y == c
        print(f"  类别{c} ({iris.target_names[c]}): "
              f"中心=({X_umap[mask, 0].mean():.2f}, {X_umap[mask, 1].mean():.2f})")

    # ===================== 3. n_neighbors 参数 =====================
    print("\n=== n_neighbors 参数影响 ===")
    for nn in [5, 10, 15, 30, 50, 100]:
        r = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1, random_state=42)
        X_n = r.fit_transform(X)
        knn = KNeighborsClassifier(n_neighbors=5)
        acc = cross_val_score(knn, X_n, y, cv=5).mean()
        print(f"  n_neighbors={nn:>3}: 1-NN CV准确率={acc:.4f}")

    # ===================== 4. min_dist 参数 =====================
    print("\n=== min_dist 参数影响 ===")
    for md in [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]:
        r = umap.UMAP(n_components=2, n_neighbors=15, min_dist=md, random_state=42)
        X_m = r.fit_transform(X)
        print(f"  min_dist={md}: 点间最小距离={md}")

    # ===================== 5. 不同维度 =====================
    print("\n=== 不同 n_components ===")
    for dim in [2, 3, 5, 10]:
        r = umap.UMAP(n_components=dim, random_state=42)
        X_d = r.fit_transform(X)
        knn = KNeighborsClassifier(n_neighbors=5)
        acc = cross_val_score(knn, X_d, y, cv=5).mean()
        print(f"  n_components={dim:>2}: 降维后形状={X_d.shape}, 1-NN CV准确率={acc:.4f}")

    # ===================== 6. metric 参数 =====================
    print("\n=== 距离度量对比 ===")
    for metric in ["euclidean", "cosine", "manhattan", "chebyshev"]:
        r = umap.UMAP(n_components=2, metric=metric, random_state=42)
        X_m = r.fit_transform(X)
        knn = KNeighborsClassifier(n_neighbors=5)
        acc = cross_val_score(knn, X_m, y, cv=5).mean()
        print(f"  metric={metric:>12}: 1-NN CV准确率={acc:.4f}")

    # ===================== 7. UMAP vs t-SNE vs PCA =====================
    from sklearn.manifold import TSNE
    print("\n=== UMAP vs t-SNE vs PCA ===")
    pca_2d = PCA(n_components=2).fit_transform(X)
    tsne_2d = TSNE(n_components=2, init="pca", random_state=42).fit_transform(X)
    umap_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=5)
    for name, X_d in [("PCA", pca_2d), ("t-SNE", tsne_2d), ("UMAP", umap_2d)]:
        acc = cross_val_score(knn, X_d, y, cv=5).mean()
        print(f"  {name:>6}: 1-NN CV准确率={acc:.4f}")

# ===================== 8. 参数说明（即使未安装也展示）=====================
print("\n=== UMAP 参数说明 ===")
print("- n_components: 目标维度（通常 2 或 3）")
print("- n_neighbors: 邻域大小，越大越关注全局结构（默认 15）")
print("- min_dist: 簇内点的最小距离，越小簇越紧凑（默认 0.1）")
print("- metric: 距离度量（euclidean, cosine, manhattan 等）")
print("- random_state: 随机种子，保证可复现")
print()
print("=== UMAP 要点 ===")
print("- 比 t-SNE 快得多，适合大数据集")
print("- 比 t-SNE 更好地保持全局结构")
print("- 可以用于特征工程（不只是可视化），因为支持 transform")
print("- 支持监督和半监督降维（transform_supervised）")
print("- 学术上比 t-SNE 更有理论支撑（基于黎曼几何和代数拓扑）")
