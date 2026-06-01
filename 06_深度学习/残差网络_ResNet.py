"""
残差网络（ResNet）—— 通过跳跃连接解决深层网络的退化问题
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np

# ===================== 1. 加载 MNIST 数据 =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# ===================== 2. 残差块（Residual Block）=====================
class ResidualBlock(nn.Module):
    """基础残差块: Conv → BN → ReLU → Conv → BN + skip connection"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 跳跃连接: F(x) + x
        return self.relu(self.block(x) + x)

# ===================== 3. 简化 ResNet =====================
class SimpleResNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # 第一层: 1 → 32 通道
        self.prep = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 残差层: 32 通道
        self.res_blocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
        )
        # 全局平均池化 + 分类器
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.res_blocks(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

# ===================== 4. 对比：有/无残差连接 =====================
class PlainNet(nn.Module):
    """普通网络（无残差连接）"""
    def __init__(self, n_classes=10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.blocks(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# ===================== 5. 训练对比 =====================
print("=== ResNet vs PlainNet ===")
for name, Model in [("PlainNet", PlainNet), ("ResNet", SimpleResNet)]:
    model = Model()
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        correct = 0
        total = 0
        for batch_X, batch_y in train_loader:
            output = model(batch_X)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (output.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            output = model(batch_X)
            test_correct += (output.argmax(1) == batch_y).sum().item()
            test_total += batch_y.size(0)

    print(f"  {name:>10}: 参数量={n_params:>6}, Test Acc={test_correct/test_total:.4f}")

# ===================== 6. 残差连接原理 =====================
print("\n=== 残差连接原理 ===")
print("传统网络: H(x) = F(x)  — 直接学习目标映射")
print("ResNet:   H(x) = F(x) + x  — 学习残差 F(x) = H(x) - x")
print()
print("为什么有效:")
print("1. 如果最优映射接近恒等，学习 F(x)≈0 比学习 H(x)≈x 容易")
print("2. 梯度可通过 skip connection 直接回传，缓解梯度消失")
print("3. 使得训练 100+ 层的网络成为可能")

print("\n=== ResNet 要点 ===")
print("- 跳跃连接（Skip Connection）: x + F(x)")
print("- BatchNorm: 稳定训练，加速收敛")
print("- 全局平均池化: 减少参数，替代全连接层")
print("- 变体: ResNet-18/34/50/101/152（层数不同）")
print("- ResNet-50+ 使用 Bottleneck 结构（1×1 → 3×3 → 1×1）减少参数")
print("- 应用: 图像分类、目标检测、语义分割的基础骨干网络")
