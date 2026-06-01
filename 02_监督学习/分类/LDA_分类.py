"""
LDA 分类 —— 线性判别分析，通过最大化类间/类内方差比找到最佳投影方向
同时可用作有监督降维（与 PCA 对比）
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ===================== 1. 加载数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 2. LDA 分类 =====================
lda = LinearDiscriminantAnalysis(solver="svd")  # svd 适合大数据，lsqr 适合小数据
lda.fit(X_train, y_train)

print("=== LDA 分类 ===")
print(f"训练集准确率: {lda.score(X_train, y_train):.4f}")
print(f"测试集准确率: {lda.score(X_test, y_test):.4f}")
print(f"解释方差比: {lda.explained_variance_ratio_.round(4)}")
print(f"投影维度: {lda.coef_.shape[1]} → {len(lda.classes_) - 1} (类别数-1)")
print(f"\n分类报告:\n{classification_report(y_test, lda.predict(X_test), target_names=iris.target_names)}")

# ===================== 3. LDA 作为有监督降维 =====================
# LDA 最多投影到 (类别数-1) 维，Iris 3 类 → 2 维
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)
print(f"\n=== LDA 降维 ===")
print(f"原始维度: {X_train.shape[1]}, 降维后: {X_train_lda.shape[1]}")
print(f"降维后前 5 个样本:\n{X_train_lda[:5].round(4)}")

# ===================== 4. LDA vs PCA 对比 =====================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

print(f"\n=== LDA vs PCA 降维对比 ===")
print(f"PCA  解释方差比: {pca.explained_variance_ratio_.round(4)} (无监督)")
print(f"LDA  解释方差比: {lda.explained_variance_ratio_.round(4)} (有监督)")

# 降维后用 KNN 分类对比效果
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
pca_acc = knn.score(pca.transform(X_test), y_test)
knn.fit(X_train_lda, y_train)
lda_acc = knn.score(X_test_lda, y_test)
print(f"PCA  → KNN 准确率: {pca_acc:.4f}")
print(f"LDA  → KNN 准确率: {lda_acc:.4f}")

# ===================== 5. 不同求解器 =====================
print("\n=== 求解器对比 ===")
for solver in ["svd", "lsqr", "eigen"]:
    lda_s = LinearDiscriminantAnalysis(solver=solver, shrinkage="auto" if solver != "svd" else None)
    lda_s.fit(X_train, y_train)
    acc = lda_s.score(X_test, y_test)
    print(f"  solver={solver:>4}: 测试准确率={acc:.4f}")

# ===================== 6. Shrinkage（收缩）=====================
print("\n=== Shrinkage 参数对比 ===")
for shrinkage in [None, "auto", 0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    lda_sh = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
    lda_sh.fit(X_train, y_train)
    acc = lda_sh.score(X_test, y_test)
    shrink_str = str(shrinkage) if shrinkage is not None else "None"
    print(f"  shrinkage={shrink_str:>6}: 测试准确率={acc:.4f}")

print("\n=== LDA 要点 ===")
print("- 最大化类间散度/类内散度比，找到最具判别力的投影方向")
print("- 最多投影到 (类别数-1) 维")
print("- 假设：各类别数据服从同协方差的高斯分布")
print("- 相比 PCA：LDA 是有监督的，投影方向更有判别力")
print("- 正则化：shrinkage 可缓解小样本时协方差矩阵估计不准的问题")
print("- 也可直接作为分类器使用（不只用于降维）")
