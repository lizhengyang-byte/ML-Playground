"""
迁移学习 — 使用预训练模型进行微调和特征提取

核心思路：
  迁移学习利用在大规模数据集（如 ImageNet）上预训练的模型，
  将其学到的通用视觉特征迁移到新任务。两种主要策略：

  1. 特征提取（Feature Extraction）：冻结预训练层，只训练新增的分类头
  2. 微调（Fine-Tuning）：解冻部分或全部预训练层，用小学习率联合训练

  本示例在 CIFAR-10 数据集上演示两种策略的对比。
  CIFAR-10 图像为 32×32，需调整预训练模型的输入尺寸。

  适用场景：
  - 目标数据集较小（<1000 样本）→ 特征提取更稳定
  - 目标数据集较大（>10000 样本）→ 微调通常效果更好
  - 新任务与 ImageNet 相似 → 迁移效果好
  - 新任务差异大 → 需要解冻更多层或使用更小的学习率
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset


# ======================== 策略一：特征提取 ========================

def build_feature_extraction_model(num_classes=10):
    """
    特征提取：冻结预训练 ResNet18 的所有层，只替换最后的全连接层。

    原理：
    - 预训练层的权重完全冻结，不参与梯度更新
    - 只有新增的分类头会被训练
    - 速度快，适合小数据集

    返回:
        model: 模型
        params_to_train: 需要训练的参数
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后的全连接层（ResNet18 原始输出是 1000 类）
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    # 只有新加的层需要训练
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())

    return model, params_to_train


# ======================== 策略二：微调 ========================

def build_finetune_model(num_classes=10, freeze_until=None):
    """
    微调：解冻部分或全部预训练层，用较小学习率训练。

    参数:
        num_classes: 分类数
        freeze_until: 冻结到第几层（None=不解冻，int=冻结前 N 层）

    原理：
    - 解冻预训练层，允许它们根据新任务调整权重
    - 使用很小的学习率（如 1e-4），避免破坏预训练特征
    - 通常分类头用更大学习率，预训练层用更小学习率

    返回:
        model: 模型
        params_group: 参数组（不同学习率）
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    if freeze_until is not None:
        # 冻结前 N 层
        children = list(model.children())
        for i, child in enumerate(children):
            if i < freeze_until:
                for param in child.parameters():
                    param.requires_grad = False

    # 将参数分为两组：预训练层（小学习率）和分类头（大学习率）
    pretrained_params = [p for p in model.parameters() if p.requires_grad and p.shape[0] != num_classes]
    classifier_params = list(model.fc.parameters())

    params_group = [
        {"params": pretrained_params, "lr": 1e-4},
        {"params": classifier_params, "lr": 1e-3},
    ]

    return model, params_group


# ======================== 数据加载 ========================

def get_dataloaders(batch_size=64, subset_size=None):
    """
    加载 CIFAR-10 数据集。

    参数:
        batch_size: 批量大小
        subset_size: 使用子集的大小（None=使用全部数据，用于加速演示）
    """
    transform_train = transforms.Compose([
        transforms.Resize(64),  # CIFAR-10 原始 32×32，放大到 64 适配 ResNet
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    # 可选：使用子集加速演示
    if subset_size is not None:
        train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        test_dataset = Subset(test_dataset, range(min(subset_size // 5, len(test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


# ======================== 训练和评估 ========================

def train_one_epoch(model, loader, criterion, optimizer, device):
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


# ======================== 主流程 ========================

def main():
    print("=" * 50)
    print("迁移学习 — 特征提取 vs 微调")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 超参数
    batch_size = 32
    epochs = 3
    subset_size = 2000  # 使用小数据集加速演示

    # 加载数据
    print(f"\n加载 CIFAR-10 数据集（使用 {subset_size} 样本子集）...")
    train_loader, test_loader = get_dataloaders(batch_size, subset_size)
    print(f"训练集大小: {len(train_loader.dataset)}  测试集大小: {len(test_loader.dataset)}")
    print(f"CIFAR-10 类别: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck")

    results = {}

    # --- 策略一：特征提取 ---
    print(f"\n{'='*50}")
    print("策略一: 特征提取 (Feature Extraction)")
    print("  - 冻结所有预训练层")
    print("  - 只训练新的分类头")
    print(f"{'='*50}")

    model_fe, params_fe = build_feature_extraction_model(num_classes=10)
    model_fe = model_fe.to(device)

    # 统计可训练参数
    trainable_fe = sum(p.numel() for p in params_fe)
    total_fe = sum(p.numel() for p in model_fe.parameters())
    print(f"  总参数: {total_fe:,}  可训练参数: {trainable_fe:,} ({trainable_fe/total_fe*100:.1f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_fe, lr=1e-3)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model_fe, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model_fe, test_loader, device)
        print(f"  Epoch {epoch}: 损失={train_loss:.4f}, 训练准确率={train_acc*100:.2f}%, 测试准确率={test_acc*100:.2f}%")

    results["特征提取"] = test_acc

    # --- 策略二：微调 ---
    print(f"\n{'='*50}")
    print("策略二: 微调 (Fine-Tuning)")
    print("  - 冻结 ResNet 的前 6 层（conv1 + layer1-3）")
    print("  - 解冻 layer4 和分类头")
    print("  - 分类头学习率 1e-3，预训练层学习率 1e-4")
    print(f"{'='*50}")

    model_ft, params_ft = build_finetune_model(num_classes=10, freeze_until=6)
    model_ft = model_ft.to(device)

    trainable_ft = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    total_ft = sum(p.numel() for p in model_ft.parameters())
    print(f"  总参数: {total_ft:,}  可训练参数: {trainable_ft:,} ({trainable_ft/total_ft*100:.1f}%)")

    optimizer_ft = optim.Adam(params_ft, lr=1e-3)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model_ft, train_loader, criterion, optimizer_ft, device)
        test_acc = evaluate(model_ft, test_loader, device)
        print(f"  Epoch {epoch}: 损失={train_loss:.4f}, 训练准确率={train_acc*100:.2f}%, 测试准确率={test_acc*100:.2f}%")

    results["微调"] = test_acc

    # --- 从零训练作为基线 ---
    print(f"\n{'='*50}")
    print("基线: 从零训练 (Training from Scratch)")
    print("  - 不使用预训练权重")
    print(f"{'='*50}")

    model_scratch = models.resnet18(weights=None)
    model_scratch.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 10))
    model_scratch = model_scratch.to(device)

    optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model_scratch, train_loader, criterion, optimizer_scratch, device)
        test_acc = evaluate(model_scratch, test_loader, device)
        print(f"  Epoch {epoch}: 损失={train_loss:.4f}, 训练准确率={train_acc*100:.2f}%, 测试准确率={test_acc*100:.2f}%")

    results["从零训练"] = test_acc

    # --- 结果对比 ---
    print(f"\n{'='*50}")
    print("三种策略对比:")
    print(f"{'='*50}")
    for name, acc in results.items():
        print(f"  {name:　<8s}: 测试准确率 {acc*100:.2f}%")

    # --- 迁移学习指导 ---
    print("\n--- 迁移学习策略选择指南 ---")
    print("  场景                          推荐策略")
    print("  数据少 (<1000) + 任务相似      → 特征提取")
    print("  数据少 (<1000) + 任务差异大    → 微调（冻结浅层）")
    print("  数据多 (>10000) + 任务相似     → 微调（解冻深层）")
    print("  数据多 (>10000) + 任务差异大   → 微调（解冻全部）")
    print("\n--- 常用预训练模型 ---")
    print("  ResNet: 经典残差网络，适合大多数视觉任务")
    print("  VGG: 结构简单，特征提取能力强，参数量大")
    print("  EfficientNet: 效率与精度平衡，适合移动端")
    print("  ViT: Vision Transformer，适合大规模数据集")


if __name__ == "__main__":
    main()
