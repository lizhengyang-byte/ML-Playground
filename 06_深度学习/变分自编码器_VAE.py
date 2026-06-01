"""
变分自编码器（VAE）—— 在自编码器基础上引入概率建模，可生成新样本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ===================== 1. 加载数据 =====================
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
train_loader = DataLoader(TensorDataset(X_train_t, X_train_t), batch_size=64, shuffle=True)

# ===================== 2. VAE 模型 =====================
class VAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, latent_dim=16):
        super().__init__()
        # 编码器: 输入 → 隐层 → 均值 + 对数方差
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # 均值 μ
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)    # 对数方差 log(σ²)

        # 解码器: 潜在变量 → 隐层 → 重建
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = μ + σ × ε, ε ~ N(0,1)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# ===================== 3. VAE 损失函数 =====================
def vae_loss(x_recon, x, mu, logvar):
    """
    VAE 损失 = 重建损失 + KL 散度
    重建损失: -E[log p(x|z)]  (用 MSE 或交叉熵近似)
    KL 散度: -0.5 × Σ(1 + log σ² - μ² - σ²)  (使潜在分布接近标准正态)
    """
    recon_loss = nn.MSELoss(reduction="sum")(x_recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# ===================== 4. 训练 =====================
model = VAE(input_dim=64, hidden_dim=128, latent_dim=16)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"=== VAE 模型 ===")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

print("\n=== 训练 ===")
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    for batch_X, _ in train_loader:
        x_recon, mu, logvar = model(batch_X)
        loss, recon, kl = vae_loss(x_recon, batch_X, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    if (epoch + 1) % 20 == 0:
        n = len(X_train)
        print(f"  Epoch {epoch+1:>3}: Total={total_loss/n:.4f}, "
              f"Recon={total_recon/n:.4f}, KL={total_kl/n:.4f}")

# ===================== 5. 生成新样本 =====================
print("\n=== 从 VAE 生成新样本 ===")
model.eval()
with torch.no_grad():
    # 从标准正态分布采样潜在向量
    z_new = torch.randn(10, 16)
    generated = model.decode(z_new).numpy()
print(f"生成 {generated.shape[0]} 个新样本，维度: {generated.shape[1]}")
print(f"生成样本的像素范围: [{generated.min():.3f}, {generated.max():.3f}]")

# ===================== 6. 潜在空间插值 =====================
print("\n=== 潜在空间插值 ===")
with torch.no_grad():
    # 取两个真实样本的潜在表示，在中间插值
    z1, _ = model.encode(X_test_t[:1])
    z2, _ = model.encode(X_test_t[1:2])
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        x_interp = model.decode(z_interp).numpy()
        print(f"  α={alpha:.2f}: 像素均值={x_interp.mean():.3f}")

# ===================== 7. 重建质量 =====================
with torch.no_grad():
    test_recon, mu, logvar = model(X_test_t)
    recon_loss = nn.MSELoss()(test_recon, X_test_t).item()
    print(f"\n=== 测试集重建 MSE: {recon_loss:.6f} ===")

print("\n=== VAE 要点 ===")
print("- 重参数化技巧: z = μ + σ × ε, 使得采样过程可微分")
print("- KL 散度: 约束潜在分布接近标准正态 N(0,I)")
print("- 生成新样本: 从 N(0,I) 采样 z → 解码器 → 新样本")
print("- 潜在空间连续: 相似的输入在潜在空间中也相近")
print("- 应用: 图像生成、异常检测、数据增强、表示学习")
