"""
LSTM（长短期记忆网络）—— RNN 的改进版本，通过门控机制解决梯度消失问题
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

# ===================== 2. LSTM 模型 =====================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"=== LSTM 模型结构 ===")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# ===================== 3. LSTM 门控机制 =====================
print("\n=== LSTM 门控机制 ===")
print("遗忘门 f_t = σ(W_f × [h_{t-1}, x_t] + b_f)  — 决定丢弃多少旧信息")
print("输入门 i_t = σ(W_i × [h_{t-1}, x_t] + b_i)  — 决定写入多少新信息")
print("候选值 C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)")
print("细胞状态 C_t = f_t × C_{t-1} + i_t × C̃_t   — 核心：加法更新，梯度流通好")
print("输出门 o_t = σ(W_o × [h_{t-1}, x_t] + b_o)")
print("隐状态 h_t = o_t × tanh(C_t)")

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
            test_pred = model(X_test)
            test_rmse = torch.sqrt(criterion(test_pred, y_test)).item()
        print(f"  Epoch {epoch+1:>2}: Train RMSE={torch.sqrt(loss).item():.4f}, "
              f"Test RMSE={test_rmse:.4f}")

# ===================== 5. 预测对比 =====================
print("\n=== 预测效果 ===")
model.eval()
with torch.no_grad():
    preds = model(X_test).numpy()
print(f"{'真实值':>10} {'预测值':>10} {'误差':>10}")
for i in range(10):
    print(f"{y_test[i].item():>10.4f} {preds[i]:>10.4f} {abs(y_test[i].item()-preds[i]):>10.4f}")

# ===================== 6. LSTM vs 简单 RNN 对比 =====================
from torch.nn import RNN as SimpleRNN

class SimpleRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = SimpleRNN(1, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :]).squeeze(-1)

print("\n=== LSTM vs RNN 训练对比 ===")
for name, Model in [("RNN", SimpleRNNModel), ("LSTM", LSTMModel)]:
    m = Model()
    opt = optim.Adam(m.parameters(), lr=0.001)
    for _ in range(50):
        m.train()
        out = m(X_train)
        loss = criterion(out, y_train)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        rmse = torch.sqrt(criterion(m(X_test), y_test)).item()
    print(f"  {name}: Test RMSE={rmse:.4f}")

print("\n=== LSTM 要点 ===")
print("- 解决了 RNN 的梯度消失问题，可学习长距离依赖")
print("- 细胞状态 C_t 通过加法更新（而非乘法），梯度流通顺畅")
print("- 参数量约为 RNN 的 4 倍（3 个门 + 候选值）")
print("- 适合：机器翻译、文本生成、语音识别、时间序列预测")
