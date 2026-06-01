"""
情感分析 —— 判断文本的情感倾向（正面/负面）
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

# ===================== 1. 构造中文情感分析数据 =====================
positive = [
    "这部电影非常精彩，值得一看",
    "服务态度很好，环境优美",
    "产品性价比很高，推荐购买",
    "味道不错，分量也足",
    "体验非常好，下次还来",
    "内容丰富，讲解清晰",
    "物流很快，包装完好",
    "价格实惠，质量不错",
    "非常满意，五星好评",
    "剧情紧凑，演员演技在线",
    "酒店位置好，房间干净整洁",
    "课程实用，学到很多知识",
] * 5  # 增加样本数

negative = [
    "太失望了，完全不值这个价",
    "服务态度很差，再也不来了",
    "质量很差，用了一天就坏了",
    "味道一般，分量太少",
    "体验很差，浪费时间",
    "内容空洞，没什么干货",
    "物流太慢，等了一个星期",
    "价格太贵，不如买别家的",
    "非常失望，差评",
    "剧情拖沓，演技尴尬",
    "房间很脏，设施老旧",
    "课程太浅，学不到东西",
] * 5

texts = positive + negative
labels = [1] * len(positive) + [0] * len(negative)

# ===================== 2. 文本向量化 + 分类 =====================
# 用 TF-IDF 提取特征
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000),
)

# 交叉验证
scores = cross_val_score(model, texts, labels, cv=5, scoring="accuracy")
print("=== 情感分析（TF-IDF + 逻辑回归）===")
print(f"5 折交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# ===================== 3. 训练和预测 =====================
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

print(f"\n训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
y_pred = model.predict(X_test)
print(f"测试准确率: {model.score(X_test, y_test):.4f}")

# ===================== 4. 预测新文本 =====================
print("\n=== 预测新文本 ===")
new_texts = [
    "这个产品真的太好了，强烈推荐！",
    "质量太差了，完全是骗钱的",
    "还行吧，一般般",
    "服务热情，环境不错，下次还来",
]
probs = model.predict_proba(new_texts)
preds = model.predict(new_texts)
for text, pred, prob in zip(new_texts, preds, probs):
    sentiment = "正面" if pred == 1 else "负面"
    print(f"  [{sentiment}] {text} (置信度: {max(prob):.3f})")

# ===================== 5. 特征词分析 =====================
print("\n=== 影响情感判断的关键词 ===")
tfidf = model.named_steps["tfidfvectorizer"]
lr = model.named_steps["logisticregression"]
feature_names = tfidf.get_feature_names_out()
coefs = lr.coef_[0]

# 正面词
top_pos = np.argsort(coefs)[::-1][:10]
print("正面情感关键词:")
for idx in top_pos:
    print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")

# 负面词
top_neg = np.argsort(coefs)[:10]
print("负面情感关键词:")
for idx in top_neg:
    print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")

# ===================== 6. 不同模型对比 =====================
print("\n=== 不同分类器对比 ===")
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

for name, clf in [("LogisticRegression", LogisticRegression(max_iter=1000)),
                   ("MultinomialNB", MultinomialNB()),
                   ("LinearSVC", LinearSVC(max_iter=1000)),
                   ("RandomForest", RandomForestClassifier(n_estimators=50))]:
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), clf)
    scores = cross_val_score(pipe, texts, labels, cv=5, scoring="accuracy")
    print(f"  {name:>20}: {scores.mean():.4f} ± {scores.std():.4f}")

print("\n=== 情感分析要点 ===")
print("- 中文需要先分词（jieba），英文可直接用空格分割")
print("- TF-IDF + 简单分类器是很好的 baseline")
print("- N-gram (1,2) 可以捕捉短语信息（如'不太好'）")
print("- 预训练模型（BERT）效果更好但需要 GPU")
print("- 数据质量是关键：标注一致性比数据量更重要")
