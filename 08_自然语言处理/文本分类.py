"""
文本分类 —— 将文本自动分类到预定义类别
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

# ===================== 1. 构造多分类文本数据 =====================
categories = {
    "科技": [
        "人工智能技术正在改变世界",
        "新发布的手机处理器性能强大",
        "云计算服务帮助企业发展",
        "量子计算取得重大突破",
        "自动驾驶技术日趋成熟",
        "芯片制造工艺不断进步",
        "区块链技术应用场景广泛",
        "5G网络覆盖范围扩大",
    ],
    "体育": [
        "足球世界杯决赛精彩纷呈",
        "篮球运动员表现出色",
        "游泳比赛打破世界纪录",
        "奥运健儿为国争光",
        "网球选手晋级大满贯决赛",
        "田径比赛竞争激烈",
        "电竞比赛观众人数创新高",
        "马拉松赛事成功举办",
    ],
    "娱乐": [
        "电影票房突破十亿",
        "综艺节目收视率很高",
        "歌手发布新专辑",
        "明星参加慈善活动",
        "电视剧剧情精彩",
        "导演获得国际大奖",
        "演员演技获得好评",
        "音乐节阵容强大",
    ],
}

texts, labels = [], []
for i, (cat, docs) in enumerate(categories.items()):
    texts.extend(docs * 3)  # 重复增加样本
    labels.extend([i] * len(docs) * 3)

label_names = list(categories.keys())

# ===================== 2. 数据划分 =====================
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
print(f"类别分布: {dict(zip(label_names, [labels.count(i) for i in range(len(label_names))]))}")

# ===================== 3. 不同模型对比 =====================
print("\n=== 不同模型 + 特征组合 ===")
for vec_name, vec_cls, vec_params in [
    ("CountVec", CountVectorizer, {}),
    ("TF-IDF", TfidfVectorizer, {}),
    ("TF-IDF(1,2)", TfidfVectorizer, {"ngram_range": (1, 2)}),
]:
    for clf_name, clf in [("NB", MultinomialNB()),
                           ("LR", LogisticRegression(max_iter=1000)),
                           ("SVM", LinearSVC(max_iter=1000))]:
        pipe = make_pipeline(vec_cls(**vec_params), clf)
        scores = cross_val_score(pipe, texts, labels, cv=5, scoring="accuracy")
        print(f"  {vec_name:>12} + {clf_name:>3}: {scores.mean():.4f} ± {scores.std():.4f}")

# ===================== 4. 最佳模型详细评估 =====================
print("\n=== 最佳模型（TF-IDF + SVM）详细评估 ===")
best_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LinearSVC(max_iter=1000)
)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print(f"测试准确率: {best_model.score(X_test, y_test):.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=label_names))

# ===================== 5. 混淆矩阵 =====================
print("=== 混淆矩阵 ===")
cm = confusion_matrix(y_test, y_pred)
print(f"{'':>10} {'预测科技':>8} {'预测体育':>8} {'预测娱乐':>8}")
for i, name in enumerate(label_names):
    print(f"真实{name:>4}  {cm[i]}")

# ===================== 6. 预测新文本 =====================
print("\n=== 预测新文本 ===")
new_texts = [
    "最新发布的人工智能模型性能大幅提升",
    "世界杯预选赛今晚开打",
    "新电影上映首日票房破纪录",
]
preds = best_model.predict(new_texts)
for text, pred in zip(new_texts, preds):
    print(f"  [{label_names[pred]}] {text}")

# ===================== 7. 超参数调优 =====================
print("\n=== 超参数调优 ===")
from sklearn.model_selection import GridSearchCV
param_grid = {
    "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidfvectorizer__max_df": [0.8, 0.9, 1.0],
    "linearsvc__C": [0.1, 1.0, 10.0],
}
pipe = make_pipeline(TfidfVectorizer(), LinearSVC(max_iter=1000))
gs = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(texts, labels)
print(f"最佳参数: {gs.best_params_}")
print(f"最佳 CV 准确率: {gs.best_score_:.4f}")

print("\n=== 文本分类要点 ===")
print("- TF-IDF + SVM/LR 是经典且强大的文本分类 baseline")
print("- N-gram 可以捕捉词组信息")
print("- 数据增强: 同义词替换、回译等可提升效果")
print("- 预训练模型(BERT)在大量数据上效果更好")
print("- 类别不平衡时考虑 class_weight='balanced'")
