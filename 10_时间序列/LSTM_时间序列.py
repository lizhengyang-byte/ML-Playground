"""
LSTM 时间序列预测

LSTM（长短期记忆网络）核心思想：
- 通过门控机制（遗忘门、输入门、输出门）解决长期依赖问题
- 适合捕捉时间序列中的非线性模式和长期依赖关系
- 滑动窗口构建样本：用过去 look_back 步预测未来 horizon 步

流程：数据构造 → 滑动窗口 → 模型定义 → 训练 → 预测 → 评估
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==================== 1. 生成合成时间序列 ====================
np.random.seed(42)
torch.manual_seed(42)

n = 500
t = np.arange(n, dtype=np.float32)
数据 = (np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 25) +
        0.01 * t + np.random.normal(0, 0.2, n)).astype(np.float32)

print("=" * 55)
print("LSTM 时间序列预测")
print("=" * 55)
print(f"数据长度: {n}, 范围: [{数据.min():.3f}, {数据.max():.3f}]")

# ==================== 2. 数据预处理 ====================
# 归一化到 [0, 1]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(数据.reshape(-1, 1)).flatten()

# 滑动窗口构造监督学习样本
def create_sequences(data, look_back, horizon):
    """滑动窗口：用过去 look_back 步预测未来 horizon 步"""
    X, y = [], []
    for i in range(len(data) - look_back - horizon + 1):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back : i + look_back + horizon])
    return np.array(X), np.array(y)

look_back = 30   # 输入窗口
horizon = 5      # 预测步数

X, y = create_sequences(data_scaled, look_back, horizon)
print(f"滑动窗口: look_back={look_back}, horizon={horizon}")
print(f"样本数: {len(X)}, 特征维度: {X.shape}")

# 训练/测试划分（按时间顺序）
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

# 转为 PyTorch 张量 [samples, time_steps, features]
X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)  # (N, look_back, 1)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
y_test_t = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==================== 3. LSTM 模型定义 ====================
class LSTM模型(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=horizon):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]    # 取最后一步 (batch, hidden)
        out = self.fc(last_hidden)          # (batch, horizon)
        return out

模型 = LSTM模型(input_size=1, hidden_size=64, num_layers=2, output_size=horizon)
参数量 = sum(p.numel() for p in 模型.parameters() if p.requires_grad)
print(f"\n模型结构:")
print(f"  LSTM 层数: 2, 隐藏维度: 64")
print(f"  可训练参数: {参数量:,}")

# ==================== 4. 训练 ====================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(模型.parameters(), lr=0.001)
epochs = 50

print(f"\n--- 训练 ({epochs} 轮) ---")
模型.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        预测 = 模型(batch_X)
        loss = criterion(预测, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch_X)
    avg_loss = epoch_loss / len(X_train)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch + 1:>3}/{epochs}: MSE = {avg_loss:.6f}")

# ==================== 5. 预测与评估 ====================
模型.eval()
with torch.no_grad():
    pred_scaled = 模型(X_test_t).numpy()

# 反归一化（保持二维形状）
def inverse_2d(data):
    return scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)

y_test_orig = inverse_2d(y_test)
pred_orig = inverse_2d(pred_scaled)

# 整体评估
rmse = np.sqrt(mean_squared_error(y_test_orig, pred_orig))
mae = mean_absolute_error(y_test_orig, pred_orig)
print(f"\n--- 测试集评估（整体） ---")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")

# 逐步评估（每一步的预测精度）
print(f"\n--- 逐步预测精度 ---")
for step in range(horizon):
    rmse_step = np.sqrt(mean_squared_error(y_test_orig[:, step], pred_orig[:, step]))
    mae_step = mean_absolute_error(y_test_orig[:, step], pred_orig[:, step])
    print(f"  第 {step + 1} 步: RMSE={rmse_step:.4f}, MAE={mae_step:.4f}")

# 展示部分预测结果
print(f"\n--- 预测 vs 真实（前 5 个样本，第 1 步） ---")
print(f"{'样本':>6}  {'真实':>8}  {'预测':>8}  {'误差':>8}")
for i in range(min(5, len(y_test_orig))):
    print(f"  {i + 1:>4}   {y_test_orig[i, 0]:>8.3f}  {pred_orig[i, 0]:>8.3f}  {y_test_orig[i, 0] - pred_orig[i, 0]:>8.3f}")

# ==================== 6. 单步迭代预测（模拟自回归） ====================
print(f"\n--- 自回归预测（用模型自身输出迭代 20 步） ---")
模型.eval()
初始输入 = data_scaled[-look_back:].copy()
当前窗口 = 初始输入.copy()

自回归预测 = []
with torch.no_grad():
    for _ in range(4):
        x = torch.FloatTensor(当前窗口).unsqueeze(0).unsqueeze(-1)  # (1, look_back, 1)
        out = 模型(x).numpy().flatten()
        自回归预测.extend(out[:5])
        # 更新窗口：移除前 5 步，加入预测值
        当前窗口 = np.concatenate([当前窗口[5:], out[:5]])

自回归值 = scaler.inverse_transform(np.array(自回归预测).reshape(-1, 1)).flatten()
print(f"{'步数':>4}  {'预测值':>8}")
for i, v in enumerate(自回归值):
    print(f"  {i + 1:>2}   {v:>8.3f}")
