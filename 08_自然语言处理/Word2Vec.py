"""
Word2Vec —— 词向量模型，将词映射为稠密向量，捕捉语义关系
需要安装: pip install gensim
"""
import numpy as np

try:
    from gensim.models import Word2Vec, KeyedVectors
    from gensim.utils import simple_preprocess
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("gensim 未安装，请运行: pip install gensim\n")

if HAS_GENSIM:
    # ===================== 1. 训练语料 =====================
    sentences = [
        list("机器学习是人工智能的基础"),
        list("深度学习是机器学习的子领域"),
        list("自然语言处理用到深度学习"),
        list("计算机视觉依赖卷积神经网络"),
        list("强化学习通过奖励训练智能体"),
        list("Python是数据科学的首选语言"),
        list("神经网络是深度学习的核心模型"),
        list("特征工程对传统机器学习很重要"),
        list("过拟合是机器学习的常见问题"),
        list("交叉验证用于评估模型泛化能力"),
    ]

    print("=== 训练语料 ===")
    for i, s in enumerate(sentences):
        print(f"  文档{i+1}: {''.join(s)}")

    # ===================== 2. 训练 Word2Vec =====================
    # vector_size: 词向量维度
    # window: 上下文窗口大小
    # min_count: 忽略出现次数低于此值的词
    # sg: 0=CBOW, 1=Skip-gram
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=3,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=100,
        workers=1,
        seed=42,
    )

    print(f"\n=== Word2Vec 模型 ===")
    print(f"词汇表大小: {len(model.wv)}")
    print(f"词向量维度: {model.wv.vector_size}")

    # ===================== 3. 词向量查询 =====================
    word = "学习"
    if word in model.wv:
        vec = model.wv[word]
        print(f"\n'{word}' 的词向量 (前10维): {vec[:10].round(4)}")
        print(f"向量范数: {np.linalg.norm(vec):.4f}")

    # ===================== 4. 相似词查询 =====================
    print("\n=== 最相似的词 ===")
    for word in ["学习", "神经", "深度"]:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=3)
            print(f"  '{word}' 最相似: {[(w, f'{s:.3f}') for w, s in similar]}")

    # ===================== 5. 词向量运算 =====================
    print("\n=== 词向量运算（类比推理）===")
    # king - man + woman ≈ queen
    # 简单示例: 深度 - 机器 + 特征 ≈ ?
    try:
        result = model.wv.most_similar(
            positive=["深度", "特征"],
            negative=["机器"],
            topn=3
        )
        print(f"  '深度' - '机器' + '特征' ≈ {[(w, f'{s:.3f}') for w, s in result]}")
    except KeyError:
        print("  部分词不在词汇表中")

    # ===================== 6. 两种训练模式 =====================
    print("\n=== CBOW vs Skip-gram ===")
    print("CBOW (sg=0): 用上下文预测中心词，速度快，适合高频词")
    print("Skip-gram (sg=1): 用中心词预测上下文，适合低频词，语义更丰富")

    # 训练 CBOW 对比
    model_cbow = Word2Vec(sentences, vector_size=50, window=3, sg=0, epochs=100, workers=1, seed=42)
    model_sg = Word2Vec(sentences, vector_size=50, window=3, sg=1, epochs=100, workers=1, seed=42)

    if "学习" in model_cbow.wv and "学习" in model_sg.wv:
        print(f"\nCBOW '学习'向量前5维: {model_cbow.wv['学习'][:5].round(4)}")
        print(f"Skip-gram '学习'向量前5维: {model_sg.wv['学习'][:5].round(4)}")

    # ===================== 7. 参数影响 =====================
    print("\n=== 参数影响 ===")
    for vs in [10, 50, 100]:
        m = Word2Vec(sentences, vector_size=vs, window=3, min_count=1, sg=1, epochs=100, workers=1, seed=42)
        print(f"  vector_size={vs:>3}: 词汇={len(m.wv)}, 向量维度={m.wv.vector_size}")

    for win in [2, 3, 5]:
        m = Word2Vec(sentences, vector_size=50, window=win, min_count=1, sg=1, epochs=100, workers=1, seed=42)
        if "学习" in m.wv and "深度" in m.wv:
            sim = m.wv.similarity("学习", "深度")
            print(f"  window={win}: '学习'-'深度' 相似度={sim:.4f}")

print("\n=== Word2Vec 要点 ===")
print("- 分布式假设: 上下文相似的词，语义也相似")
print("- 词向量可以捕捉语义关系（类比推理）")
print("- vector_size 通常 100~300（越大表达力越强但需更多数据）")
print("- 预训练词向量: Google Word2Vec / GloVe / FastText")
print("- 局限: 每个词只有一个向量（多义词问题 → BERT）")
