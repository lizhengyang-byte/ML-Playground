"""
GRU（门控循环单元）—— LSTM 的简化版，用更少的参数达到相近效果
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================== 1. 序列数据生成 =====================
np.random.seed(42)
timesteps = 100

def generate_sine_data(n_samples, timesteps):
    X, y = [], []
    for _ in range(n_samples):
        start = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(start, start + 4 * np.pi, timesteps + 1)
        data = np.sin(t) + 0.1 * np.random.randn(len(t))
        X.append(data[:-1])
        y.append(data[-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = generate_sine_data(500, timesteps)
X_train, X_test = torch.FloatTensor(X[:400]).unsqueeze(-1), torch.FloatTensor(X[400:]).unsqueeze(-1)
y_train, y_test = torch.FloatTensor(y[:400]), torch.FloatTensor(y[400:])

# ===================== 2. GRU 模型 =====================
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

model = GRUModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"=== GRU 模型结构 ===")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# ===================== 3. GRU 门控机制 =====================
print("\n=== GRU 门控机制 ===")
print("更新门 z_t = σ(W_z × [h_{t-1}, x_t]) — 决定保留多少旧状态")
print("重置门 r_t = σ(W_r × [h_{t-1}, x_t]) — 决定遗忘多少旧状态来计算候选值")
print("候选值 h̃_t = tanh(W × [r_t × h_{t-1}, x_t])")
print("新状态 h_t = (1 - z_t) × h_{t-1} + z_t × h̃_t")
print()
print("与 LSTM 的区别:")
print("  GRU 有 2 个门（更新门、重置门），LSTM 有 3 个门（遗忘门、输入门、输出门）")
print("  GRU 没有独立的细胞状态，参数量约为 LSTM 的 3/4")

# ===================== 4. 训练 =====================
print("\n=== 训练 ===")
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_rmse = torch.sqrt(criterion(model(X_test), y_test)).item()
        print(f"  Epoch {epoch+1:>2}: Train RMSE={torch.sqrt(loss).item():.4f}, "
              f"Test RMSE={test_rmse:.4f}")

# ===================== 5. GRU vs LSTM vs RNN 对比 =====================
from torch.nn import RNN as SimpleRNN

class SimpleRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = SimpleRNN(1, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.rnn(x, torch.zeros(2, x.size(0), 64))
        return self.fc(out[:, -1, :]).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x, (torch.zeros(2, x.size(0), 64), torch.zeros(2, x.size(0), 64)))
        return self.fc(out[:, -1, :]).squeeze(-1)

print("\n=== RNN vs GRU vs LSTM 对比 ===")
for name, Model in [("RNN", SimpleRNNModel), ("GRU", GRUModel), ("LSTM", LSTMModel)]:
    m = Model()
    n_params = sum(p.numel() for p in m.parameters())
    opt = optim.Adam(m.parameters(), lr=0.001)
    for _ in range(50):
        m.train()
        loss = criterion(m(X_train), y_train)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        rmse = torch.sqrt(criterion(m(X_test), y_test)).item()
    print(f"  {name}: 参数量={n_params:>6}, Test RMSE={rmse:.4f}")

print("\n=== GRU 要点 ===")
print("- 参数比 LSTM 少，训练更快")
print("- 效果通常与 LSTM 相当（有时更好，有时稍差）")
print("- 适合：数据量较小或需要快速实验时")
print("- 与 LSTM 一样可捕捉长距离依赖")
