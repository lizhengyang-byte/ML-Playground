"""
朴素贝叶斯 —— 基于贝叶斯定理 + 特征条件独立假设的概率分类器
"""
from sklearn.datasets import fetch_20newsgroups, load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

# ===================== 1. 高斯朴素贝叶斯（连续特征）=====================
print("=== 高斯朴素贝叶斯 (Iris) ===")
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"训练准确率: {gnb.score(X_train, y_train):.4f}")
print(f"测试准确率: {gnb.score(X_test, y_test):.4f}")
print(f"各类别先验概率: {dict(zip(iris.target_names, gnb.class_prior_.round(4)))}")
print(f"各类别均值形状: {gnb.theta_.shape}")
print(f"\n分类报告:\n{classification_report(y_test, gnb.predict(X_test), target_names=iris.target_names)}")

# ===================== 2. 多项式朴素贝叶斯（离散特征/计数）=====================
print("\n=== 多项式朴素贝叶斯（文本分类）===")
# 使用小规模文本数据集
categories = ["sci.space", "rec.sport.baseball", "comp.graphics"]
train_data = fetch_20newsgroups(subset="train", categories=categories, random_state=42)
test_data = fetch_20newsgroups(subset="test", categories=categories, random_state=42)

# 词袋模型向量化
vectorizer = CountVectorizer(max_features=5000, stop_words="english")
X_train_counts = vectorizer.fit_transform(train_data.data)
X_test_counts = vectorizer.transform(test_data.data)

mnb = MultinomialNB(alpha=1.0)  # alpha 为拉普拉斯平滑参数
mnb.fit(X_train_counts, train_data.target)
print(f"训练准确率: {mnb.score(X_train_counts, train_data.target):.4f}")
print(f"测试准确率: {mnb.score(X_test_counts, test_data.target):.4f}")
print(f"类别: {train_data.target_names}")

# ===================== 3. 拉普拉斯平滑（alpha）=====================
print("\n=== alpha 参数（平滑）对多项式 NB 的影响 ===")
for alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    mnb_a = MultinomialNB(alpha=alpha)
    mnb_a.fit(X_train_counts, train_data.target)
    test_acc = mnb_a.score(X_test_counts, test_data.target)
    print(f"  alpha={alpha:>6}: 测试准确率={test_acc:.4f}")

# ===================== 4. 伯努利朴素贝叶斯（二值特征）=====================
print("\n=== 伯努利朴素贝叶斯 ===")
# 将词频转为 0/1（是否出现）
X_train_bin = (X_train_counts > 0).astype(int)
X_test_bin = (X_test_counts > 0).astype(int)

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_bin, train_data.target)
print(f"测试准确率: {bnb.score(X_test_bin, test_data.target):.4f}")

# ===================== 5. ComplementNB（互补朴素贝叶斯）=====================
# 针对类别不平衡优化的 MultinomialNB 变体
print("\n=== ComplementNB（类别不平衡场景）===")
cnb = ComplementNB(alpha=1.0)
cnb.fit(X_train_counts, train_data.target)
print(f"测试准确率: {cnb.score(X_test_counts, test_data.target):.4f}")

# ===================== 6. 各朴素贝叶斯变体对比 =====================
print("\n=== 各变体在文本分类上的对比 ===")
models = {
    "MultinomialNB": MultinomialNB(alpha=1.0),
    "BernoulliNB": BernoulliNB(alpha=1.0),
    "ComplementNB": ComplementNB(alpha=1.0),
}
for name, model in models.items():
    if name == "BernoulliNB":
        model.fit(X_train_bin, train_data.target)
        acc = model.score(X_test_bin, test_data.target)
    else:
        model.fit(X_train_counts, train_data.target)
        acc = model.score(X_test_counts, test_data.target)
    print(f"  {name:>15}: 测试准确率={acc:.4f}")

print("\n=== 朴素贝叶斯要点 ===")
print("- 核心假设：各特征在给定类别下条件独立（现实中很少完全满足，但效果仍好）")
print("- GaussianNB: 适合连续特征（假设特征服从高斯分布）")
print("- MultinomialNB: 适合离散计数数据（文本分类的经典选择）")
print("- BernoulliNB: 适合二值特征（词是否出现，而非词频）")
print("- ComplementNB: 适合类别不平衡的文本分类")
print("- 优点：训练快、小样本友好、可处理多分类")
print("- 缺点：独立假设在强相关特征上表现差")
