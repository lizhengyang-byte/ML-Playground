"""
数据增强 — 使用 torchvision.transforms 实现常用图像增强技术

核心思路：
  数据增强通过对训练图像施加随机变换来扩充数据集，
  提高模型的泛化能力，缓解过拟合。这是计算机视觉中
  最简单且最有效的正则化手段之一。

  本示例展示：
  1. 各种数据增强操作的参数和效果
  2. 组合增强流水线
  3. 在 MNIST 上训练对比：有增强 vs 无增强
  4. 增强操作的统计信息
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ======================== 增强技术详解 ========================

def demonstrate_transforms():
    """展示各种数据增强操作的参数和含义"""
    print("--- 常用数据增强技术 ---\n")

    augmentation_info = [
        ("RandomCrop(28, padding=4)",
         "随机裁剪：先在图像四周填充 4 像素，再随机裁剪回 28×28。\n"
         "    作用：模拟物体在图像中不同位置，提高平移不变性。"),
        ("RandomHorizontalFlip(p=0.5)",
         "随机水平翻转：以 50% 概率左右镜像。\n"
         "    作用：增加水平方向的样本多样性。"),
        ("RandomRotation(15)",
         "随机旋转：在 [-15°, +15°] 范围内随机旋转。\n"
         "    作用：模拟不同角度拍摄，提高旋转鲁棒性。"),
        ("ColorJitter(brightness=0.2, contrast=0.2)",
         "颜色抖动：随机调整亮度和对比度。\n"
         "    作用：模拟不同光照条件（注：MNIST 是灰度图，效果有限）。"),
        ("RandomAffine(degrees=10, translate=(0.1, 0.1))",
         "随机仿射变换：在指定角度和平移范围内随机变换。\n"
         "    作用：综合模拟旋转和平移的组合变换。"),
        ("RandomErasing(p=0.5, scale=(0.02, 0.15))",
         "随机擦除：在图像上随机遮挡一小块区域。\n"
         "    作用：强迫模型关注全局特征，避免过度依赖局部区域。"),
        ("Normalize((0.1307,), (0.3081,))",
         "标准化：将像素值减去均值除以标准差。\n"
         "    作用：使数据分布与预训练模型匹配，加速收敛。"),
    ]

    for name, desc in augmentation_info:
        print(f"  {name}")
        print(f"    {desc}")
        print()


# ======================== 增强流水线构建 ========================

def get_transforms(use_augmentation=True):
    """
    构建数据预处理流水线。

    参数:
        use_augmentation: 是否启用数据增强

    返回:
        训练变换、测试变换
    """
    # 测试集不做增强，只做必要的预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if use_augmentation:
        # 训练集：增强 + 预处理
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),        # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
            transforms.RandomRotation(15),                # 随机旋转
            transforms.ToTensor(),                        # 转为张量
            transforms.Normalize((0.1307,), (0.3081,)),  # 标准化
            transforms.RandomErasing(p=0.3),              # 随机擦除（需在张量上操作）
        ])
    else:
        # 训练集：不做增强
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    return train_transform, test_transform


# ======================== 简单分类模型 ========================

class SimpleCNN(nn.Module):
    """用于 MNIST 分类的简单 CNN"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ======================== 训练和评估 ========================

def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        correct += (output.argmax(1) == target).sum().item()
        total += data.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    """评估模型准确率"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    return correct / total


# ======================== 对比实验 ========================

def run_comparison():
    """对比有增强和无增强的训练效果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 3
    lr = 1e-3

    results = {}

    for aug_flag, aug_name in [(False, "无增强"), (True, "有增强")]:
        print(f"\n{'='*40}")
        print(f"实验: {aug_name}")
        print(f"{'='*40}")

        train_transform, test_transform = get_transforms(use_augmentation=aug_flag)

        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            print(f"  Epoch {epoch}: 训练损失={train_loss:.4f}, "
                  f"训练准确率={train_acc*100:.2f}%, 测试准确率={test_acc*100:.2f}%")

        results[aug_name] = test_acc

    print(f"\n{'='*40}")
    print("对比结果:")
    for name, acc in results.items():
        print(f"  {name}: 测试准确率 {acc*100:.2f}%")
    print("注: 数据增强在更大数据集和更多 epoch 下效果更显著")


# ======================== 主流程 ========================

def main():
    print("=" * 50)
    print("数据增强 — 计算机视觉中的数据扩充")
    print("=" * 50)

    # 展示各种增强技术
    demonstrate_transforms()

    # 打印增强流水线
    train_transform, test_transform = get_transforms(use_augmentation=True)
    print("--- 训练增强流水线 ---")
    print(f"  {train_transform}\n")
    print("--- 测试预处理流水线 ---")
    print(f"  {test_transform}\n")

    # 对比实验
    run_comparison()

    # 补充说明
    print("\n--- 数据增强最佳实践 ---")
    print("  1. 增强应与任务匹配：医学图像不适合水平翻转（左右不对称）")
    print("  2. 增强强度要适中：过度增强会引入噪声，降低数据质量")
    print("  3. 测试时不做增强：评估应使用标准预处理，确保可复现")
    print("  4. 增强与正则化结合：配合 Dropout、权重衰减效果更好")
    print("  5. 高级方法：AutoAugment、RandAugment 可自动搜索最优增强策略")


if __name__ == "__main__":
    main()
