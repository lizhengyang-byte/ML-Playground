"""
词袋模型（Bag of Words）—— 将文本转为词频向量，不考虑词序
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# ===================== 1. 示例文档 =====================
corpus = [
    "机器学习是人工智能的基础",
    "深度学习是机器学习的子领域",
    "自然语言处理用到深度学习",
    "计算机视觉也依赖机器学习",
    "强化学习是另一个重要方向",
]
print("=== 语料库 ===")
for i, doc in enumerate(corpus):
    print(f"  文档{i+1}: {doc}")

# ===================== 2. 中文分词 =====================
try:
    import jieba
    corpus_cut = [" ".join(jieba.cut(doc)) for doc in corpus]
except ImportError:
    # 简单按字符分词
    corpus_cut = [" ".join(doc) for doc in corpus]

print("\n=== 分词后 ===")
for i, doc in enumerate(corpus_cut):
    print(f"  文档{i+1}: {doc}")

# ===================== 3. CountVectorizer =====================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus_cut)

print(f"\n=== 词袋矩阵 ===")
print(f"矩阵形状: {X.shape} (文档数 × 词汇表大小)")
print(f"词汇表: {vectorizer.get_feature_names_out()}")

# 稀疏矩阵转稠密显示
print(f"\n词频矩阵:")
for i, doc in enumerate(corpus_cut):
    freq = X[i].toarray().flatten()
    non_zero = [(vectorizer.get_feature_names_out()[j], freq[j]) for j in range(len(freq)) if freq[j] > 0]
    print(f"  文档{i+1}: {non_zero}")

# ===================== 4. 参数设置 =====================
print("\n=== CountVectorizer 参数 ===")

# min_df: 忽略出现次数低于此阈值的词
vec_min_df = CountVectorizer(min_df=2)
X_min = vec_min_df.fit_transform(corpus_cut)
print(f"min_df=2: 词汇表={vec_min_df.get_feature_names_out()}")

# max_df: 忽略出现频率高于此比例的词
vec_max_df = CountVectorizer(max_df=0.8)
X_max = vec_max_df.fit_transform(corpus_cut)
print(f"max_df=0.8: 词汇表={vec_max_df.get_feature_names_out()}")

# max_features: 限制词汇表大小
vec_top = CountVectorizer(max_features=5)
X_top = vec_top.fit_transform(corpus_cut)
print(f"max_features=5: 词汇表={vec_top.get_feature_names_out()}")

# ===================== 5. N-gram =====================
print("\n=== N-gram ===")
# unigram
vec_1 = CountVectorizer(ngram_range=(1, 1))
X_1 = vec_1.fit_transform(corpus_cut)
print(f"Unigram (1,1): {len(vec_1.vocabulary_)} 个特征")
print(f"  特征: {vec_1.get_feature_names_out()}")

# bigram
vec_2 = CountVectorizer(ngram_range=(1, 2))
X_2 = vec_2.fit_transform(corpus_cut)
print(f"Bigram (1,2): {len(vec_2.vocabulary_)} 个特征")

# ===================== 6. 词频 vs TF-IDF =====================
print("\n=== 词袋模型的局限 ===")
print("1. 高频词主导（如'的'、'是'），需要停用词过滤")
print("2. 不考虑词序（'学习机器'和'机器学习'相同）")
print("3. 所有词权重相同（'的'和'人工智能'同等重要）")
print("4. 改进: TF-IDF（考虑词的区分度）")

# ===================== 7. 实际应用示例 =====================
print("\n=== 词袋模型用于分类 ===")
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# 模拟文本分类
docs = [
    "机器学习 深度学习 神经网络 算法",
    "自然语言处理 文本 分词 语义",
    "图像识别 卷积神经网络 视觉 目标检测",
    "股票 基金 投资 市场 经济",
    "足球 比赛 冠军 联赛 球员",
] * 10  # 重复增加样本数
labels = [0, 0, 0, 1, 2] * 10

vec = CountVectorizer()
X = vec.fit_transform(docs)
nb = MultinomialNB()
scores = cross_val_score(nb, X, labels, cv=3, scoring="accuracy")
print(f"词袋 + 朴素贝叶斯 CV准确率: {scores.mean():.4f}")

print("\n=== 词袋模型要点 ===")
print("- 简单高效，是文本分类的常用 baseline")
print("- 适合：文本分类、信息检索、主题建模")
print("- 不适合：需要理解语义或词序的任务")
