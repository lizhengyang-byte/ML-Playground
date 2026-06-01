"""
自编码器 —— 无监督学习，通过编码-解码结构学习数据的低维表示
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
X = StandardScaler().fit_transform(digits.data)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=64, shuffle=True)

# ===================== 2. 自编码器模型 =====================
class Autoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=16):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),  # 输出范围 [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

model = Autoencoder(input_dim=64, latent_dim=16)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"=== 自编码器模型 ===")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# ===================== 3. 训练 =====================
print("\n=== 训练 ===")
n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for batch_X, _ in train_loader:
        recon, _ = model(batch_X)
        loss = criterion(recon, batch_X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_recon, _ = model(X_test_t)
            test_loss = criterion(test_recon, X_test_t).item()
        print(f"  Epoch {epoch+1:>2}: Train Loss={total_loss/len(train_loader):.6f}, "
              f"Test Loss={test_loss:.6f}")

# ===================== 4. 重建质量 =====================
print("\n=== 重建效果 ===")
model.eval()
with torch.no_grad():
    test_recon, _ = model(X_test_t)
    # 逐样本重建误差
    errors = ((X_test_t - test_recon) ** 2).mean(dim=1)
    print(f"平均重建误差: {errors.mean():.6f}")
    print(f"最大重建误差: {errors.max():.6f}")
    print(f"最小重建误差: {errors.min():.6f}")

# ===================== 5. 不同潜在维度 =====================
print("\n=== 不同 latent_dim 对比 ===")
for ld in [2, 4, 8, 16, 32]:
    m = Autoencoder(input_dim=64, latent_dim=ld)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(50):
        m.train()
        for bx, _ in train_loader:
            recon, _ = m(bx)
            loss = criterion(recon, bx)
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        test_recon, _ = m(X_test_t)
        test_loss = criterion(test_recon, X_test_t).item()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  latent_dim={ld:>2}: Test MSE={test_loss:.6f}, 参数量={n_params:,}")

# ===================== 6. 去噪自编码器 =====================
print("\n=== 去噪自编码器 (Denoising Autoencoder) ===")
noise_factor = 0.5
X_train_noisy = X_train_t + noise_factor * torch.randn_like(X_train_t)
X_test_noisy = X_test_t + noise_factor * torch.randn_like(X_test_t)

denoise_model = Autoencoder(input_dim=64, latent_dim=16)
opt_d = optim.Adam(denoise_model.parameters(), lr=1e-3)
train_loader_d = DataLoader(TensorDataset(X_train_noisy, X_train_t), batch_size=64, shuffle=True)

for epoch in range(50):
    denoise_model.train()
    for noisy, clean in train_loader_d:
        recon, _ = denoise_model(noisy)
        loss = criterion(recon, clean)
        opt_d.zero_grad()
        loss.backward()
        opt_d.step()

denoise_model.eval()
with torch.no_grad():
    noisy_recon, _ = denoise_model(X_test_noisy)
    denoise_loss = criterion(noisy_recon, X_test_t).item()
    # 对比直接用原始自编码器处理噪声数据
    clean_recon, _ = model(X_test_noisy)
    clean_loss = criterion(clean_recon, X_test_t).item()
print(f"去噪自编码器重建损失: {denoise_loss:.6f}")
print(f"普通自编码器处理噪声: {clean_loss:.6f}")

print("\n=== 自编码器要点 ===")
print("- 瓶颈层（latent_dim）迫使网络学习数据的压缩表示")
print("- 去噪自编码器：输入加噪声，目标是原始数据，增强鲁棒性")
print("- 可用于：降维、去噪、特征学习、数据生成（变体）")
print("- latent_dim 太小 → 信息丢失；太大 → 过拟合/退化为恒等映射")
