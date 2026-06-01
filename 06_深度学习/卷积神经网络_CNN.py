"""
卷积神经网络（CNN）—— 利用卷积核提取局部特征，适合图像/网格数据
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np

# ===================== 1. 加载 MNIST 数据 =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 均值和标准差
])

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# ===================== 2. 定义 CNN 模型 =====================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层: 1×28×28 → 32×14×14 → 64×7×7
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1→32 通道
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28→14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32→64 通道
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14→7
        )
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"=== CNN 模型结构 ===")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# ===================== 3. 查看卷积核 =====================
print("\n=== 卷积层信息 ===")
for name, param in model.named_parameters():
    if "weight" in name and "features" in name:
        print(f"  {name}: shape={param.shape}")

# ===================== 4. 训练 =====================
print("\n=== 训练 ===")
n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        output = model(batch_X)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == batch_y).sum().item()
        total += batch_y.size(0)

    # 测试
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            output = model(batch_X)
            test_correct += (output.argmax(1) == batch_y).sum().item()
            test_total += batch_y.size(0)

    print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
          f"Train Acc={correct/total:.4f}, Test Acc={test_correct/test_total:.4f}")

# ===================== 5. 特征图可视化 =====================
print("\n=== 卷积特征提取示意 ===")
# 展示中间层输出形状
sample = train_dataset[0][0].unsqueeze(0)  # 取一个样本
model.eval()
with torch.no_grad():
    feat1 = model.features[0](sample)  # 第一层卷积输出
    feat2 = model.features[3](model.features[2](model.features[1](feat1)))  # 第二层
print(f"输入:       {sample.shape}")
print(f"Conv1 输出: {feat1.shape}")
print(f"Conv2 输出: {feat2.shape}")

# ===================== 6. CNN 要点 =====================
print("\n=== CNN 关键组件 ===")
print("1. 卷积层 (Conv2d): 局部连接 + 权值共享，提取局部特征")
print("2. 池化层 (MaxPool2d): 降采样，增加平移不变性")
print("3. 全连接层: 将特征图映射到类别输出")
print("4. ReLU: 非线性激活")
print("5. Dropout: 正则化，防过拟合")

print("\n=== CNN 要点 ===")
print("- 卷积核大小: 3×3 最常用（VGG 风格）")
print("- 通道数: 逐层增加（1→32→64→128）")
print("- 每次池化分辨率减半，通道数翻倍")
print("- 参数共享使 CNN 比全连接网络参数量少得多")
print("- 适合：图像分类、目标检测、语义分割等视觉任务")
