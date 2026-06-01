"""
文本生成 —— 根据给定上下文生成新的文本
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================== 1. 简单字符级 RNN 文本生成 =====================
print("=== 字符级 RNN 文本生成 ===")

# 训练文本
text = """人工智能是计算机科学的一个分支，它企图了解智能的实质。
机器学习是人工智能的核心，通过算法让计算机从数据中学习规律。
深度学习利用神经网络模拟人脑的工作方式。
自然语言处理让计算机理解和生成人类语言。"""

# 构建字符到索引的映射
chars = sorted(list(set(text)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(chars)
seq_len = 20

print(f"文本长度: {len(text)} 字符")
print(f"词汇表大小: {vocab_size}")
print(f"字符: {''.join(chars[:20])}...")

# 构造训练数据
X_data, y_data = [], []
for i in range(0, len(text) - seq_len):
    X_data.append([char2idx[c] for c in text[i:i+seq_len]])
    y_data.append([char2idx[c] for c in text[i+1:i+seq_len+1]])

X_tensor = torch.LongTensor(X_data)
y_tensor = torch.LongTensor(y_data)

# ===================== 2. 定义模型 =====================
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.rnn(embeds, hidden)
        logits = self.fc(out)
        return logits, hidden

model = CharRNN(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ===================== 3. 训练 =====================
print("\n=== 训练 ===")
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    output, _ = model(X_tensor)
    loss = criterion(output.reshape(-1, vocab_size), y_tensor.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}")

# ===================== 4. 文本生成函数 =====================
def generate_text(model, start_text, length=50, temperature=0.8):
    """从起始文本生成指定长度的新文本"""
    model.eval()
    # 将起始文本转为索引
    input_seq = torch.LongTensor([[char2idx.get(c, 0) for c in start_text]])
    hidden = None
    result = list(start_text)

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_seq, hidden)
            # temperature: 控制随机性（越低越确定，越高越随机）
            probs = torch.softmax(logits[0, -1] / temperature, dim=0)
            idx = torch.multinomial(probs, 1).item()
            result.append(idx2char[idx])
            input_seq = torch.LongTensor([[idx]])

    return "".join(result)

# ===================== 5. 生成文本示例 =====================
print("\n=== 生成文本（不同 temperature）===")
start = "人工智能"
for temp in [0.5, 0.8, 1.0, 1.2]:
    generated = generate_text(model, start, length=40, temperature=temp)
    print(f"  T={temp}: {generated}")

# ===================== 6. 不同 temperature 的效果 =====================
print("\n=== Temperature 参数 ===")
print("T < 1.0: 更保守，选择概率高的词，文本更连贯但单调")
print("T = 1.0: 按原始概率采样")
print("T > 1.0: 更随机，文本更多样但可能不通顺")

# ===================== 7. 词级生成（简化示例）=====================
print("\n=== 词级生成思路 ===")
print("1. 分词 → 构建词表（限制大小，如 top-5000）")
print("2. 训练 RNN/LSTM/Transformer 预测下一个词")
print("3. 用采样策略（贪心/Top-k/Top-p/Beam Search）生成")
print()
print("采样策略:")
print("  贪心: 始终选概率最高的词（确定性，缺乏多样性）")
print("  Top-k: 只从前 k 个候选中采样")
print("  Top-p (Nucleus): 从累积概率达到 p 的最小候选集中采样")
print("  Beam Search: 保留多个最优序列，适合翻译等任务")

print("\n=== 文本生成要点 ===")
print("- 自回归生成: 每次预测下一个 token，逐步生成")
print("- GPT 系列是当前最强的文本生成模型")
print("- Temperature 是控制生成质量/多样性的关键参数")
print("- 长文本生成需要处理重复和一致性问题")
print("- 评估困难: BLEU/ROUGE 只衡量表面相似度")
