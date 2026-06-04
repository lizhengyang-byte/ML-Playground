"""
生成对抗网络（GAN）—— 生成器和判别器博弈训练，学习数据分布并生成逼真样本
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
train_loader = DataLoader(TensorDataset(X_train_t), batch_size=64, shuffle=True)

# ===================== 2. 生成器 =====================
class Generator(nn.Module):
    """随机噪声 z → 生成假样本"""
    def __init__(self, latent_dim=32, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh(),  # 输出 [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

# ===================== 3. 判别器 =====================
class Discriminator(nn.Module):
    """输入样本 → 输出真/假概率"""
    def __init__(self, input_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# ===================== 4. 初始化 =====================
latent_dim = 32
G = Generator(latent_dim=latent_dim)
D = Discriminator(input_dim=64)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

print("=== GAN 模型 ===")
print(f"生成器参数: {sum(p.numel() for p in G.parameters()):,}")
print(f"判别器参数: {sum(p.numel() for p in D.parameters()):,}")

# ===================== 5. 训练 =====================
print("\n=== 训练 ===")
n_epochs = 100
d_losses, g_losses = [], []

for epoch in range(n_epochs):
    epoch_d_loss = 0
    epoch_g_loss = 0

    for (real_data,) in train_loader:
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # === 训练判别器 ===
        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z).detach()  # 切断生成器梯度

        d_real = D(real_data)
        d_fake = D(fake_data)
        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # === 训练生成器 ===
        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z)
        d_fake = D(fake_data)
        g_loss = criterion(d_fake, real_labels)  # 生成器想让判别器判为真

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()

    d_losses.append(epoch_d_loss / len(train_loader))
    g_losses.append(epoch_g_loss / len(train_loader))

    if (epoch + 1) % 20 == 0:
        # 计算判别器准确率
        with torch.no_grad():
            z = torch.randn(len(X_test), latent_dim)
            fake = G(z)
            d_real_acc = (D(torch.FloatTensor(X_test)) > 0.5).float().mean().item()
            d_fake_acc = (D(fake) < 0.5).float().mean().item()
        print(f"  Epoch {epoch+1:>3}: D_loss={d_losses[-1]:.4f}, G_loss={g_losses[-1]:.4f}, "
              f"D_acc: real={d_real_acc:.2f}, fake={d_fake_acc:.2f}")

# ===================== 6. 生成新样本 =====================
print("\n=== 生成新样本 ===")
G.eval()
with torch.no_grad():
    z = torch.randn(10, latent_dim)
    generated = G(z).numpy()
    print(f"生成 {generated.shape[0]} 个样本，像素范围: [{generated.min():.3f}, {generated.max():.3f}]")

# ===================== 7. 训练动态 =====================
print("\n=== 训练动态 ===")
print(f"最终 D_loss: {d_losses[-1]:.4f}, G_loss: {g_losses[-1]:.4f}")
print("理想状态: D_loss ≈ ln(4) ≈ 1.386, D(real) ≈ 0.5, D(fake) ≈ 0.5")

print("\n=== GAN 要点 ===")
print("生成器 G: 噪声 z → 假样本，目标：骗过判别器")
print("判别器 D: 样本 → 真/假，目标：区分真假")
print("博弈目标: min_G max_D V(D,G) = E[log D(x)] + E[log(1-D(G(z)))]")
print()
print("常见问题:")
print("1. 模式坍塌: 生成器只生成少数样本")
print("2. 训练不稳定: D 太强 → G 梯度消失；D 太弱 → G 无指导")
print("3. 改进: WGAN, Progressive GAN, StyleGAN")
