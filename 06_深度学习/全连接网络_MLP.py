"""
全连接网络（MLP）—— 多层感知机，深度学习的基础架构
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===================== 1. 加载数据 =====================
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转为 Tensor
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ===================== 2. 定义 MLP 模型 =====================
class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[128, 64], n_classes=10, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"=== MLP 模型结构 ===")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# ===================== 3. 训练 =====================
print("\n=== 训练 ===")
n_epochs = 30
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        output = model(batch_X)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(dim=1) == batch_y).sum().item()
        total += batch_y.size(0)

    if (epoch + 1) % 5 == 0:
        train_acc = correct / total
        model.eval()
        with torch.no_grad():
            test_output = model(X_test_t)
            test_acc = (test_output.argmax(dim=1) == y_test_t).float().mean().item()
        print(f"  Epoch {epoch+1:>2}: Loss={total_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

# ===================== 4. 不同网络结构对比 =====================
print("\n=== 不同网络结构对比 ===")
configs = {
    "浅 (64→10)": [64, 10],
    "中 (64→128→10)": [64, 128, 10],
    "深 (64→128→64→10)": [64, 128, 64, 10],
    "宽 (64→256→128→10)": [64, 256, 128, 10],
}
for name, hidden in configs.items():
    m = MLP(hidden_dims=hidden[1:-1] if len(hidden) > 2 else [], n_classes=10)
    opt = optim.Adam(m.parameters(), lr=0.001)
    for _ in range(20):
        m.train()
        for bx, by in train_loader:
            out = m(bx)
            loss = criterion(out, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(X_test_t).argmax(1) == y_test_t).float().mean().item()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  {name:>25}: Test Acc={acc:.4f}, 参数量={n_params:,}")

# ===================== 5. 常用技巧 =====================
print("\n=== 深度学习常用技巧 ===")
print("1. Dropout: 随机丢弃神经元，防止过拟合")
print("2. BatchNorm: 稳定训练，加速收敛")
print("3. 学习率调度: 随训练逐步降低学习率")
print("4. 权重初始化: Xavier/He 初始化")
print("5. 早停 (Early Stopping): 验证集性能不再提升时停止")
