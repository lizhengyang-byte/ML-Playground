"""
Transformer —— 基于自注意力机制的序列模型，完全并行化，取代 RNN/LSTM 成为 NLP 主流
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# ===================== 1. 位置编码 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# ===================== 2. 简化 Transformer 编码器 =====================
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, n_classes=10, max_len=200, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len) - token indices
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_enc(x)

        # 生成注意力掩码（padding mask）
        src_key_padding_mask = (x.sum(dim=-1) == 0)  # 简化：全零为 padding

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        # 取 [CLS] 或平均池化
        x = x.mean(dim=1)  # 平均池化
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ===================== 3. 自注意力机制详解 =====================
print("=== 自注意力机制 (Self-Attention) ===")
print("Q = X × W_Q, K = X × W_K, V = X × W_V")
print("Attention(Q,K,V) = softmax(Q × K^T / √d_k) × V")
print()
print("多头注意力 (Multi-Head Attention):")
print("  将 Q,K,V 分成 h 个头，分别计算注意力后拼接")
print("  MultiHead = Concat(head_1, ..., head_h) × W_O")
print("  每个头关注不同的子空间信息")

# ===================== 4. 构造分类数据 =====================
np.random.seed(42)
torch.manual_seed(42)

# 模拟文本分类数据
n_samples = 1000
vocab_size = 1000
max_len = 50
n_classes = 5

# 生成随机序列和标签
X = torch.randint(1, vocab_size, (n_samples, max_len))
# 简单规则：前几个 token 的均值决定类别
y = (X[:, :5].float().mean(dim=1) % n_classes).long()

X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# ===================== 5. 训练 Transformer =====================
model = TransformerClassifier(vocab_size=vocab_size, n_classes=n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"\n=== Transformer 模型结构 ===")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

print("\n=== 训练 ===")
n_epochs = 30
for epoch in range(n_epochs):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
            test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
        print(f"  Epoch {epoch+1:>2}: Loss={loss.item():.4f}, "
              f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

# ===================== 6. 注意力权重可视化 =====================
print("\n=== 注意力权重分析 ===")
# 获取注意力权重需要修改模型输出中间层，此处简化演示
with torch.no_grad():
    sample = X_test[:1]
    emb = model.embedding(sample) * math.sqrt(model.embedding.embedding_dim)
    emb = model.pos_enc(emb)
    # 手动计算 Q, K
    q = model.transformer.layers[0].self_attn.in_proj_weight[:64]  # 简化
    print(f"嵌入维度: {emb.shape}")
    print(f"输出维度: {model.transformer(emb).shape}")

print("\n=== Transformer vs RNN/LSTM ===")
print("Transformer: 并行计算所有位置（快），捕捉全局依赖（好）")
print("RNN/LSTM: 序列计算（慢），长距离依赖受限")
print("但 Transformer 对长序列的内存消耗大（O(n²)）")

print("\n=== Transformer 要点 ===")
print("- 自注意力: O(1) 路径长度，直接建模任意距离的依赖")
print("- 多头注意力: 每个头学习不同的关注模式")
print("- 位置编码: 注入序列顺序信息（Transformer 本身无位置感知）")
print("- 残差连接 + LayerNorm: 稳定训练")
print("- 应用: BERT(编码), GPT(解码), Vision Transformer(图像)")
