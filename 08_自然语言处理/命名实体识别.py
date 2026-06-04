"""
命名实体识别（NER）—— 从文本中识别出人名、地名、组织名等实体
"""
# ===================== 1. 基于规则的 NER =====================
import re

print("=== 基于规则的 NER ===")
text = "张三在北京大学工作,他毕业于清华大学,住在北京市海淀区。"
print(f"文本: {text}")

# 简单规则：匹配"XX大学"、"XX市"
patterns = {
    "组织": re.compile(r"[一-龥]{2,6}(?:大学|公司|研究院|实验室)"),
    "地点": re.compile(r"[一-龥]{2,6}(?:市|省|区|县|镇)"),
}
for entity_type, pattern in patterns.items():
    entities = pattern.findall(text)
    print(f"  {entity_type}: {entities}")

# ===================== 2. 基于预训练模型的 NER =====================
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[SKIP] transformers 未安装，跳过本示例")
    import sys; sys.exit(0)
    HAS_TF = False
    print("\ntransformers 未安装,跳过预训练模型 NER 演示")

if HAS_TF:
    # 使用中文 BERT NER 模型
    model_name = "bert-base-chinese"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 使用简单的标注示例代替专门的 NER 模型
        print("\n=== BERT 分词示例 ===")
        text_en = "Apple Inc. is located in California."
        inputs = tokenizer(text_en, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"  原文: {text_en}")
        print(f"  Token: {tokens}")
    except Exception as e:
        print(f"  模型加载失败: {e}")

# ===================== 3. CoNLL 格式标注数据 =====================
print("\n=== CoNLL 标注格式 (BIO) ===")
# B-: 实体开始, I-: 实体内部, O: 非实体
conll_data = [
    ("张", "B-PER"), ("三", "I-PER"), ("在", "O"), ("北", "B-LOC"),
    ("京", "I-LOC"), ("大", "I-LOC"), ("学", "I-LOC"), ("工", "O"),
    ("作", "O"),
]
for word, tag in conll_data:
    print(f"  {word}\t{tag}")

print("\n常见标签:")
print("  B-PER / I-PER: 人名")
print("  B-LOC / I-LOC: 地名")
print("  B-ORG / I-ORG: 组织名")
print("  B-TIME / I-TIME: 时间")
print("  O: 非实体")

# ===================== 4. 简化 NER 模型（BiLSTM-CRF）=====================
import torch
import torch.nn as nn

class SimpleNERModel(nn.Module):
    """简化版 BiLSTM NER 模型"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, n_tags)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        return logits

print("\n=== BiLSTM NER 模型 ===")
model = SimpleNERModel(vocab_size=5000, embed_dim=128, hidden_dim=64, n_tags=7)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")
print(f"结构: Embedding → BiLSTM → Linear → CRF(可选)")

# ===================== 5. NER 评估指标 =====================
print("\n=== NER 评估指标 ===")
print("实体级别评估（严格匹配）:")
print("  精确率: 预测实体中正确的比例")
print("  召回率: 真实实体中被找到的比例")
print("  F1: 精确率和召回率的调和平均")
print()
print("片段级别评估:")
print("  只要实体类型正确就算对（不要求边界完全匹配）")

# ===================== 6. 常用 NER 工具和模型 =====================
print("\n=== 常用 NER 工具 ===")
print("1. spaCy: 工业级 NLP 工具,内置 NER")
print("   en_core_web_sm / zh_core_web_sm")
print("2. Hugging Face Transformers:")
print("   bert-base-chinese-finetuned-cluener")
print("3. HanLP: 中文 NLP 工具包,NER 效果好")
print("4. LAC (百度): 中文词法分析（分词+词性+NER）")

print("\n=== NER 要点 ===")
print("- 中文 NER 需要先分词（分词错误会传播）")
print("- BERT + CRF 是当前最强的 NER 架构之一")
print("- 标注数据是 NER 的主要瓶颈（标注成本高）")
print("- 少样本场景: 尝试 few-shot NER 或数据增强")
print("- 实体嵌套问题: 传统 BIO 标注难以处理嵌套实体")
