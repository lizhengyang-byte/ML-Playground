"""
图像分类 — 使用 PyTorch 在 MNIST 数据集上训练 CNN 分类器

核心思路：
  卷积神经网络（CNN）通过卷积层提取局部特征，池化层降低空间维度，
  全连接层完成分类。本示例构建一个简单的 CNN，在 MNIST 手写数字
  数据集上训练并评估分类准确率。

网络结构：
  输入(1,28,28) → Conv(32) → ReLU → MaxPool
                → Conv(64) → ReLU → MaxPool
                → Flatten → FC(128) → ReLU → FC(10)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ======================== 模型定义 ========================

class SimpleCNN(nn.Module):
    """简单的卷积神经网络，用于 MNIST 手写数字分类"""

    def __init__(self):
        super().__init__()
        # 特征提取：两层卷积 + 池化
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (1,28,28) → (32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (32,14,14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # → (64,7,7)
        )
        # 分类器：全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 个数字类别
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================== 数据加载 ========================

def get_dataloaders(batch_size=64):
    """加载 MNIST 数据集，返回训练和测试 DataLoader"""
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为张量并归一化到 [0,1]
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ======================== 训练函数 ========================

def train(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个 epoch，返回平均损失"""
    model.train()
    total_loss = 0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches
    print(f"  Epoch {epoch}  训练损失: {avg_loss:.4f}")
    return avg_loss


# ======================== 评估函数 ========================

def evaluate(model, test_loader, device):
    """评估模型在测试集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # 取概率最大的类别
            correct += (pred == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    return accuracy


# ======================== 主流程 ========================

def main():
    # 超参数
    batch_size = 64
    lr = 1e-3
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, test_loader = get_dataloaders(batch_size)
    print(f"训练集大小: {len(train_loader.dataset)}  测试集大小: {len(test_loader.dataset)}")

    # 构建模型、损失函数、优化器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    print("\n开始训练:")
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, device, epoch)

    # 评估
    accuracy = evaluate(model, test_loader, device)
    print(f"\n测试集准确率: {accuracy * 100:.2f}%")

    # 打印模型结构摘要
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")


if __name__ == "__main__":
    main()
