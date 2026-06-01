"""
t-SNE —— 基于概率分布的非线性降维方法，擅长保持局部结构，常用于高维数据可视化
"""
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ===================== 1. 加载数据 =====================
# Iris: 4维 → 2维（小数据集示例）
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_iris = StandardScaler().fit_transform(X_iris)

# Digits: 64维 → 2维（中等数据集示例）
digits = load_digits()
X_digits, y_digits = digits.data, digits.target
X_digits = StandardScaler().fit_transform(X_digits)

# ===================== 2. t-SNE 降维（Iris）=====================
print("=== t-SNE 降维（Iris: 4D → 2D）===")
tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
            init="pca", random_state=42)
X_tsne = tsne.fit_transform(X_iris)
print(f"降维后形状: {X_tsne.shape}")
print(f"KL 散度（损失）: {tsne.kl_divergence_:.4f}")

# 各类别在降维空间的中心
for c in range(3):
    mask = y_iris == c
    print(f"  类别{c} ({iris.target_names[c]}): "
          f"中心=({X_tsne[mask, 0].mean():.2f}, {X_tsne[mask, 1].mean():.2f}), "
          f"样本数={mask.sum()}")

# ===================== 3. perplexity 参数 =====================
print("\n=== perplexity 参数影响 ===")
for perp in [5, 10, 20, 30, 50]:
    tsne_p = TSNE(n_components=2, perplexity=perp, init="pca", random_state=42)
    X_p = tsne_p.fit_transform(X_iris)
    print(f"  perplexity={perp:>2}: KL散度={tsne_p.kl_divergence_:.4f}")

# ===================== 4. 不同 n_components =====================
print("\n=== n_components=3 ===")
tsne_3d = TSNE(n_components=3, perplexity=30, init="pca", random_state=42)
X_3d = tsne_3d.fit_transform(X_iris)
print(f"降维后形状: {X_3d.shape}")

# ===================== 5. Digits 数据集（64维 → 2维）=====================
print("\n=== t-SNE 降维（Digits: 64D → 2D）===")
# 先用 PCA 降到 30 维加速（t-SNE 在高维数据上很慢）
pca = PCA(n_components=30, random_state=42)
X_digits_pca = pca.fit_transform(X_digits)
print(f"PCA 预降维: {X_digits.shape[1]}D → {X_digits_pca.shape[1]}D "
      f"(保留 {pca.explained_variance_ratio_.sum():.2%} 方差)")

tsne_d = TSNE(n_components=2, perplexity=30, init="pca", random_state=42)
X_d_tsne = tsne_d.fit_transform(X_digits_pca)
print(f"t-SNE 降维后: {X_d_tsne.shape}")
print(f"KL 散度: {tsne_d.kl_divergence_:.4f}")

# ===================== 6. t-SNE vs PCA 对比 =====================
print("\n=== t-SNE vs PCA ===")
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_iris)
print(f"PCA  方差比: {pca_2d.explained_variance_ratio_.round(4)}, "
      f"累积: {pca_2d.explained_variance_ratio_.sum():.4f}")
print(f"t-SNE KL散度: {tsne.kl_divergence_:.4f}")

# 用 1-NN 分类器评估降维质量
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
acc_pca = cross_val_score(knn, X_pca, y_iris, cv=5).mean()
acc_tsne = cross_val_score(knn, X_tsne, y_iris, cv=5).mean()
print(f"PCA  → 1-NN CV准确率: {acc_pca:.4f}")
print(f"t-SNE → 1-NN CV准确率: {acc_tsne:.4f}")

print("\n=== t-SNE 要点 ===")
print("- 非线性降维，擅长保持高维数据的局部邻域结构")
print("- 最常用于 2D/3D 可视化，不建议用于特征工程（不可逆、不稳定）")
print("- perplexity：控制邻域大小（通常 5~50）")
print("- 学习率：'auto' 通常效果好，手动设需小心")
print("- init='pca'：用 PCA 初始化可加速收敛、提高稳定性")
print("- 计算复杂度 O(n²)，大数据集先用 PCA 预降维")
print("- 每次运行结果可能不同（受随机种子影响），不保持全局结构")
