"""
自编码器降维 —— 用神经网络学习数据的低维表示，是非线性版本的 PCA
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# ===================== 1. 加载数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_tensor = torch.FloatTensor(X_train)
dataset = TensorDataset(X_tensor, X_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ===================== 2. 定义自编码器 =====================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # 编码器：输入 → 隐层 → 潜在表示
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # 解码器：潜在表示 → 隐层 → 重建输入
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)

# ===================== 3. 训练自编码器 =====================
input_dim = X_train.shape[1]
hidden_dim = 8
latent_dim = 2  # 目标降维维度

model = Autoencoder(input_dim, hidden_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("=== 训练自编码器 ===")
n_epochs = 200
for epoch in range(n_epochs):
    epoch_loss = 0
    for batch_X, _ in loader:
        recon, _ = model(batch_X)
        loss = criterion(recon, batch_X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1:>3}: Loss={avg_loss:.6f}")

# ===================== 4. 降维结果 =====================
model.eval()
with torch.no_grad():
    X_train_encoded = model.encode(torch.FloatTensor(X_train)).numpy()
    X_test_encoded = model.encode(torch.FloatTensor(X_test)).numpy()

print(f"\n=== 降维结果 ===")
print(f"原始维度: {X_train.shape[1]}, 降维后: {X_train_encoded.shape[1]}")
print(f"训练集降维后形状: {X_train_encoded.shape}")
print(f"测试集降维后形状: {X_test_encoded.shape}")

# ===================== 5. 对比 PCA =====================
print("\n=== 自编码器 vs PCA ===")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
# 自编码器降维
acc_ae = cross_val_score(knn, X_train_encoded, y_train, cv=5).mean()
# PCA 降维
acc_pca = cross_val_score(knn, X_train_pca, y_train, cv=5).mean()
print(f"自编码器 → 1-NN CV准确率: {acc_ae:.4f}")
print(f"PCA      → 1-NN CV准确率: {acc_pca:.4f}")
print(f"PCA 解释方差: {pca.explained_variance_ratio_.round(4)}")

# ===================== 6. 不同潜在维度 =====================
print("\n=== 不同 latent_dim 对比 ===")
for ld in [1, 2, 3]:
    model_ld = Autoencoder(input_dim, 8, ld)
    opt = torch.optim.Adam(model_ld.parameters(), lr=0.01)
    for _ in range(200):
        for batch_X, _ in loader:
            recon, _ = model_ld(batch_X)
            loss = criterion(recon, batch_X)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model_ld.eval()
    with torch.no_grad():
        X_enc = model_ld.encode(torch.FloatTensor(X_train)).numpy()
    knn_ld = KNeighborsClassifier(n_neighbors=5)
    acc = cross_val_score(knn_ld, X_enc, y_train, cv=5).mean()
    print(f"  latent_dim={ld}: 1-NN CV准确率={acc:.4f}")

# ===================== 7. 重建质量 =====================
with torch.no_grad():
    X_recon, _ = model(torch.FloatTensor(X_test))
    recon_error = criterion(X_recon, torch.FloatTensor(X_test)).item()
print(f"\n=== 重建质量 ===")
print(f"测试集重建 MSE: {recon_error:.6f}")

print("\n=== 自编码器降维要点 ===")
print("- 编码器将高维数据压缩到低维，解码器从低维重建原始数据")
print("- 是 PCA 的非线性推广：线性自编码器 ≈ PCA")
print("- 优势：可捕捉非线性流形结构")
print("- 劣势：需要调参（网络结构、学习率、训练轮数）、不如 PCA 稳定")
print("- 可扩展：变分自编码器（VAE）可生成新样本")
