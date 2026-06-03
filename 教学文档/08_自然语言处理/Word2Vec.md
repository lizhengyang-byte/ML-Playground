# Word2Vec：词向量的革命

> 所属模块：08_自然语言处理 | 源文件：Word2Vec.py | 核心功能：Skip-gram/CBOW、词向量训练、语义相似度

## 概述

Word2Vec（2013）将每个词映射到一个低维稠密向量，语义相似的词在向量空间中距离近。经典发现：king - man + woman ≈ queen。

## 关键代码解释

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
# sg=1: Skip-gram, sg=0: CBOW
model.wv.most_similar("机器学习", topn=5)
```

## 注意事项

1. 需要大量语料训练
2. 一词多义问题无法解决（同一个词只有一个向量）
3. OOV（词表外）词没有向量

## 延伸思考

- **FastText**：子词级别的词向量，解决 OOV
- **GloVe**：基于共现矩阵的词向量
- **上下文词向量**：ELMo/BERT 每个词的向量随上下文变化