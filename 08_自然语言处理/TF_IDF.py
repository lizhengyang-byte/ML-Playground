"""
TF-IDF —— 词频-逆文档频率，衡量一个词在文档中的重要程度
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

# ===================== 1. 示例语料库 =====================
corpus = [
    "机器学习是人工智能的基础技术",
    "深度学习是机器学习的子领域，用到神经网络",
    "自然语言处理是人工智能的应用之一",
    "计算机视觉用卷积神经网络处理图像",
    "强化学习通过奖励信号训练智能体",
    "Python是机器学习最常用的编程语言",
]

# ===================== 2. 中文分词 =====================
try:
    import jieba
    corpus_cut = [" ".join(jieba.cut(doc)) for doc in corpus]
except ImportError:
    corpus_cut = [" ".join(doc) for doc in corpus]

print("=== 语料库 ===")
for i, doc in enumerate(corpus_cut):
    print(f"  文档{i+1}: {doc}")

# ===================== 3. TF-IDF 计算 =====================
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus_cut)

print(f"\n=== TF-IDF 矩阵 ===")
print(f"形状: {X.shape}")
feature_names = tfidf.get_feature_names_out()

# 展示每个文档的 Top-3 词
for i in range(len(corpus_cut)):
    row = X[i].toarray().flatten()
    top_idx = row.argsort()[::-1][:3]
    print(f"  文档{i+1} Top-3: {[(feature_names[j], f'{row[j]:.3f}') for j in top_idx]}")

# ===================== 4. TF-IDF 公式 =====================
print("\n=== TF-IDF 公式 ===")
print("TF(t,d) = 词 t 在文档 d 中的出现次数 / 文档 d 的总词数")
print("IDF(t) = log(总文档数 / 包含词 t 的文档数)")
print("TF-IDF(t,d) = TF(t,d) × IDF(t)")
print()
print("含义: 一个词在当前文档中频繁出现（TF高）且在其他文档中罕见（IDF高），则 TF-IDF 高")

# 手动计算一个词的 TF-IDF 验证
print("\n手动验证（'机器' 在文档 1 中）:")
# 找到'机器'的索引
machine_idx = list(feature_names).index("机器")
tf = X[0, machine_idx]  # sklearn 已归一化
print(f"  TF-IDF (sklearn): {tf:.4f}")

# ===================== 5. 参数对比 =====================
print("\n=== 参数对比 ===")

# sublinear_tf: 对 TF 取 log(1+TF)，缓解高频词主导
for sublinear in [False, True]:
    vec = TfidfVectorizer(sublinear_tf=sublinear)
    X_t = vec.fit_transform(corpus_cut)
    print(f"  sublinear_tf={sublinear}: 形状={X_t.shape}")

# max_df / min_df
vec_df = TfidfVectorizer(min_df=1, max_df=0.8)
X_df = vec_df.fit_transform(corpus_cut)
print(f"  min_df=1, max_df=0.8: 词汇={len(vec_df.vocabulary_)}")

# norm: L1 或 L2 归一化
for norm in ["l1", "l2", None]:
    vec_n = TfidfVectorizer(norm=norm)
    X_n = vec_n.fit_transform(corpus_cut)
    if norm:
        l2_norms = np.linalg.norm(X_n.toarray(), axis=1)[:3]
        print(f"  norm={norm}: 样本L2范数={l2_norms.round(4)}")

# ===================== 6. TF-IDF vs CountVectorizer =====================
print("\n=== TF-IDF vs 词袋模型 ===")
from sklearn.feature_extraction.text import CountVectorizer

docs_class = [
    "机器学习 深度学习 神经网络",
    "自然语言处理 文本 语义",
    "机器学习 算法 模型",
    "图像 卷积神经网络 视觉",
    "文本 分词 自然语言",
    "深度学习 图像 模型",
] * 5
labels = [0, 1, 0, 2, 1, 2] * 5

for name, vec_cls in [("CountVectorizer", CountVectorizer), ("TfidfVectorizer", TfidfVectorizer)]:
    pipe = make_pipeline(vec_cls(), MultinomialNB())
    scores = cross_val_score(pipe, docs_class, labels, cv=3, scoring="accuracy")
    print(f"  {name:>20} + NB: CV准确率={scores.mean():.4f}")

print("\n=== TF-IDF 要点 ===")
print("- 比词袋模型更好：惩罚在所有文档都出现的常见词")
print("- sublinear_tf=True: 常用设置，对 TF 取 log")
print("- 适合：文本分类、信息检索、关键词提取")
print("- 不考虑词序：仍然无法捕捉'学习机器'vs'机器学习'的区别")
