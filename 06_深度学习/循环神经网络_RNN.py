"""
循环神经网络（RNN）—— 处理序列数据的神经网络，具有"记忆"能力
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================== 1. 序列数据生成 =====================
# 任务：根据正弦波序列预测下一个值
np.random.seed(42)
timesteps = 100
n_samples = 500

def generate_sine_data(n_samples, timesteps):
    X, y = [], []
    for _ in range(n_samples):
        start = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(start, start + 4 * np.pi, timesteps + 1)
        data = np.sin(t) + 0.1 * np.randn(len(t))  # 加噪声
        X.append(data[:-1])  # 输入: 前 100 个点
        y.append(data[-1])   # 目标: 第 101 个点
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = generate_sine_data(n_samples, timesteps)
# 划分训练/测试
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# 转为 (batch, seq_len, features) 形状
X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)  # (400, 100, 1)
X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
y_train_t = torch.FloatTensor(y_train)
y_test_t = torch.FloatTensor(y_test)

# ===================== 2. 定义 RNN 模型 =====================
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h0: 初始隐状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)  # out: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out.squeeze(-1)

model = RNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"=== RNN 模型结构 ===")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# ===================== 3. 训练 =====================
print("\n=== 训练 ===")
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪：防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t)
            train_rmse = torch.sqrt(loss).item()
            test_rmse = torch.sqrt(test_loss).item()
        print(f"  Epoch {epoch+1:>2}: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}")

# ===================== 4. 预测效果 =====================
print("\n=== 预测效果（前 10 个测试样本）===")
model.eval()
with torch.no_grad():
    preds = model(X_test_t).numpy()
for i in range(10):
    print(f"  真实={y_test[i]:>8.4f}, 预测={preds[i]:>8.4f}, "
          f"误差={abs(y_test[i]-preds[i]):>8.4f}")

# ===================== 5. RNN 的梯度问题 =====================
print("\n=== RNN 的梯度问题 ===")
print("梯度消失: 长序列中，梯度在反向传播时指数衰减，导致远距离依赖难以学习")
print("梯度爆炸: 梯度在反向传播时指数增长，导致训练不稳定")

# 演示梯度消失
print("\n梯度范数示例:")
for name, param in model.named_parameters():
    if "weight" in name and param.grad is not None:
        print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")

# ===================== 6. LSTM 和 GRU 解决方案 =====================
print("\n=== 解决方案: LSTM / GRU ===")
print("LSTM: 引入门控机制（遗忘门、输入门、输出门）和细胞状态，缓解梯度消失")
print("GRU: LSTM 的简化版（重置门、更新门），参数更少，效果相近")

print("\n=== RNN 要点 ===")
print("- 输入: (batch, seq_len, input_size)")
print("- 隐状态 h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)")
print("- 适合：文本、时间序列、语音等变长序列数据")
print("- 缺点：难以捕捉长距离依赖（被 LSTM/GRU/Transformer 取代）")
