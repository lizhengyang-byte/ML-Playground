"""
文本预处理 —— 分词、去停用词、词干提取、文本向量化前的清洗步骤
"""
import re
from collections import Counter

# ===================== 1. 示例文本 =====================
texts = [
    "机器学习是人工智能的一个重要分支，深度学习是机器学习的子领域。",
    "自然语言处理（NLP）是AI领域中最热门的研究方向之一。",
    "Python是最好的编程语言，没有之一！",
    "The quick brown fox jumps over the lazy dog.",
    "I am learning natural language processing with Python.",
]
print("=== 原始文本 ===")
for i, t in enumerate(texts):
    print(f"  {i+1}. {t}")

# ===================== 2. 中文分词 =====================
# 使用 jieba 分词（如未安装则用简单示例）
try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    print("\njieba 未安装，使用简单分词示例")

if HAS_JIEBA:
    print("\n=== jieba 中文分词 ===")
    for text in texts[:3]:
        words = list(jieba.cut(text))
        print(f"  原文: {text}")
        print(f"  分词: {'|'.join(words)}")

    # 添加自定义词典
    jieba.add_word("自然语言处理")
    jieba.add_word("深度学习")
    print("\n添加自定义词后:")
    words = list(jieba.cut("自然语言处理是深度学习的应用"))
    print(f"  分词: {'|'.join(words)}")

# ===================== 3. 英文分词 + 小写化 =====================
print("\n=== 英文预处理 ===")
en_text = "The Quick Brown FOX jumps over the lazy dog. It's a beautiful day!"
# 小写化
lower = en_text.lower()
# 去除标点和特殊字符
clean = re.sub(r"[^a-zA-Z0-9\s]", "", lower)
# 分词
words = clean.split()
print(f"  原文: {en_text}")
print(f"  清洗后: {' '.join(words)}")

# ===================== 4. 去停用词 =====================
# 中文停用词（简化版）
zh_stopwords = {"的", "是", "了", "在", "和", "有", "我", "不", "这", "就",
                "也", "都", "而", "及", "与", "或", "一个", "没有", "我们",
                "人们", "把", "那", "你", "他", "它", "她", "被", "让", "到"}
# 英文停用词
en_stopwords = {"the", "a", "an", "is", "are", "was", "were", "it", "its",
                "i", "you", "he", "she", "we", "they", "am", "to", "of",
                "in", "on", "at", "for", "with", "over", "about"}

if HAS_JIEBA:
    print("\n=== 去停用词（中文）===")
    text = "机器学习是人工智能的一个重要分支"
    words = list(jieba.cut(text))
    filtered = [w for w in words if w not in zh_stopwords and len(w.strip()) > 0]
    print(f"  原文: {text}")
    print(f"  分词: {words}")
    print(f"  去停用词: {filtered}")

print("\n=== 去停用词（英文）===")
words = clean.split()
filtered = [w for w in words if w not in en_stopwords]
print(f"  原文: {words}")
print(f"  去停用词: {filtered}")

# ===================== 5. 词干提取 / 词形还原 =====================
print("\n=== 词干提取 / 词形还原 ===")
try:
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    words_en = ["running", "runs", "ran", "easily", "fairly", "studies", "studying"]
    print(f"  原词:     {words_en}")
    print(f"  词干提取: {[stemmer.stem(w) for w in words_en]}")
    print(f"  词形还原: {[lemmatizer.lemmatize(w) for w in words_en]}")
except ImportError:
    print("  NLTK 未安装，跳过词干提取演示")

# ===================== 6. 正则清洗 =====================
print("\n=== 常用正则清洗模式 ===")
text = "用户 ID:12345 的邮箱 test@email.com，电话 138-0000-1234，金额 ¥100.50"
# 去除邮箱
no_email = re.sub(r"\S+@\S+", "[邮箱]", text)
print(f"  去邮箱: {no_email}")
# 去除电话
no_phone = re.sub(r"\d{3}[-.]?\d{4}[-.]?\d{4}", "[电话]", no_email)
print(f"  去电话: {no_phone}")
# 去除数字
no_num = re.sub(r"\d+", "[数字]", text)
print(f"  去数字: {no_num}")

# ===================== 7. 文本长度统计 =====================
print("\n=== 文本长度统计 ===")
for text in texts:
    char_len = len(text)
    word_len = len(text.split())
    print(f"  字符数={char_len:>3}, 词数={word_len:>2}: {text[:30]}...")

print("\n=== 文本预处理流程总结 ===")
print("1. 小写化/统一大小写")
print("2. 去除标点、特殊字符、HTML标签")
print("3. 分词（中文用 jieba/jieba-fast，英文用空格分割）")
print("4. 去停用词（高频无意义词）")
print("5. 词干提取/词形还原（统一词形）")
print("6. 去除过短的词（长度<2）")
