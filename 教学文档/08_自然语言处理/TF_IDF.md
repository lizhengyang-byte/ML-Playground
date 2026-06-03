# TF-IDF：词频-逆文档频率

> 所属模块：08_自然语言处理 | 源文件：TF_IDF.py | 核心功能：TF-IDF 加权、文本特征提取

## 概述

TF-IDF 是词袋模型的改进。TF（词频）衡量词在文档中的频率，IDF（逆文档频率）降低常见词的权重。两者相乘，让"对当前文档重要但在整个语料中不常见"的词获得高权重。

## 关键代码解释

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True)
X = tfidf.fit_transform(texts)
```

`sublinear_tf=True` 用 1+log(tf) 替代 tf，压缩高频词的影响。

## 注意事项

1. 仍然是稀疏高维向量
2. IDF 在训练集上计算，应用到测试集
3. 中文需要先分词

## 延伸思考

- **BM25**：TF-IDF 的改进版，搜索引擎的标准
- **TF-IDF + SVM**：经典的文本分类组合
- **Transformer 时代**：BERT 等预训练模型是否完全取代了 TF-IDF