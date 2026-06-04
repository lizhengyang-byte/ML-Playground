"""
图像分割 — 语义分割概念与预训练模型演示

核心思路：
  图像分割是像素级的分类任务，为图像中每个像素分配一个类别标签。
  语义分割不区分同类别的不同实例（与实例分割的区别）。

  本示例展示：
  1. 使用预训练 DeepLabV3 模型进行语义分割推理
  2. 手动实现简化的 U-Net 结构（编码器-解码器架构）
  3. 分割任务的核心概念（上采样、跳跃连接、Dice Loss）

DeepLabV3 使用空洞卷积（Atrous/Dilated Convolution）扩大感受野，
在不降低分辨率的情况下捕获多尺度上下文信息。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image


# ======================== DeepLabV3 预训练模型 ========================

def load_deeplabv3():
    """加载预训练的 DeepLabV3 语义分割模型"""
    model = deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    model.eval()
    return model


# ======================== 合成演示图像 ========================

def create_demo_image():
    """创建一张简单的合成图像用于演示分割流程"""
    img = Image.new("RGB", (256, 256), color=(135, 206, 235))  # 天蓝色背景
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # 模拟地面（绿色区域）
    draw.rectangle([0, 180, 256, 256], fill=(34, 139, 34))
    # 模拟建筑物（灰色矩形）
    draw.rectangle([80, 100, 180, 180], fill=(128, 128, 128))
    # 模拟树木（深绿色圆形）
    draw.ellipse([30, 120, 80, 180], fill=(0, 100, 0))
    draw.ellipse([180, 110, 230, 170], fill=(0, 100, 0))
    return img


def run_deeplabv3(model, image, transform):
    """
    使用 DeepLabV3 对图像进行语义分割。

    参数:
        model: 预训练分割模型
        image: PIL Image
        transform: 图像变换

    返回:
        分割掩码（每个像素的类别索引）
    """
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    # DeepLabV3 输出一个字典，'out' 键包含分割结果
    # 形状: (1, 21, H, W) — 21 个类别（Pascal VOC）
    logits = output["out"]
    # 取每个像素概率最大的类别
    seg_mask = logits.argmax(dim=1).squeeze(0)  # (H, W)
    return seg_mask, logits


# ======================== 简化 U-Net 实现 ========================

class DoubleConv(nn.Module):
    """U-Net 的基本卷积块：Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MiniUNet(nn.Module):
    """
    简化版 U-Net 语义分割网络。

    U-Net 架构核心思想：
    - 编码器（下采样）：逐步缩小空间尺寸，提取高层语义特征
    - 解码器（上采样）：逐步恢复空间尺寸，生成像素级预测
    - 跳跃连接：将编码器特征拼接到解码器，保留细节信息

    结构：
    编码器: [64] → [128] → [256]
    瓶颈层: [512]
    解码器: [256] → [128] → [64]
    输出: 1×1 卷积映射到类别数
    """

    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # 编码器（下采样路径）
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        # 瓶颈层
        self.bottleneck = DoubleConv(256, 512)

        # 解码器（上采样路径）
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)  # 512 = 256(上采样) + 256(跳跃连接)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)  # 256 = 128 + 128

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)   # 128 = 64 + 64

        # 输出层
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # 池化层
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)           # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)

        # 瓶颈层
        b = self.bottleneck(self.pool(e3))  # (B, 512, H/8, W/8)

        # 解码器 + 跳跃连接
        d3 = self.up3(b)                        # (B, 256, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)          # (B, 512, H/4, W/4)
        d3 = self.dec3(d3)                       # (B, 256, H/4, W/4)

        d2 = self.up2(d3)                        # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)          # (B, 256, H/2, W/2)
        d2 = self.dec2(d2)                       # (B, 128, H/2, W/2)

        d1 = self.up1(d2)                        # (B, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)          # (B, 128, H, W)
        d1 = self.dec1(d1)                       # (B, 64, H, W)

        return self.out_conv(d1)  # (B, num_classes, H, W)


# ======================== 损失函数 ========================

class DiceLoss(nn.Module):
    """
    Dice Loss — 图像分割中常用的损失函数。

    Dice 系数衡量预测区域与真实区域的重叠度：
    Dice = 2 * |A ∩ B| / (|A| + |B|)

    相比交叉熵损失，Dice Loss 对类别不平衡更鲁棒，
    因为它直接优化区域重叠度而非逐像素分类。
    """

    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        参数:
            predictions: (B, C, H, W) 模型输出
            targets: (B, H, W) 类别标签
        """
        # 将标签转为 one-hot 编码
        targets_one_hot = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # 展平
        pred_flat = predictions.view(-1)
        target_flat = targets_one_hot.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1.0 - dice


# ======================== 主流程 ========================

def main():
    print("=" * 50)
    print("图像分割 — 语义分割演示")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Part 1: DeepLabV3 预训练模型 ---
    print("\n--- Part 1: DeepLabV3 预训练语义分割 ---")
    print("加载预训练 DeepLabV3 模型...")
    model = load_deeplabv3()
    transform = transforms.Compose([transforms.ToTensor()])

    image = create_demo_image()
    print(f"图像尺寸: {image.size}")

    seg_mask, logits = run_deeplabv3(model, image, transform)
    print(f"分割掩码形状: {seg_mask.shape}")
    print(f"输出 logits 形状: {logits.shape}")

    # 统计各类别像素数
    unique_classes = seg_mask.unique()
    print(f"检测到的类别数: {len(unique_classes)}")
    # Pascal VOC 类别名称
    voc_classes = [
        "背景", "飞机", "自行车", "鸟", "船", "瓶子", "公交车", "汽车",
        "猫", "椅子", "牛", "餐桌", "狗", "马", "摩托车", "人",
        "盆栽", "羊", "沙发", "火车", "显示器",
    ]
    for cls in unique_classes:
        cls_idx = cls.item()
        cls_name = voc_classes[cls_idx] if cls_idx < len(voc_classes) else f"类别{cls_idx}"
        pixel_count = (seg_mask == cls).sum().item()
        print(f"  {cls_name} (索引 {cls_idx}): {pixel_count} 像素")

    # --- Part 2: Mini U-Net ---
    print("\n--- Part 2: 简化 U-Net 结构 ---")
    mini_unet = MiniUNet(in_channels=1, num_classes=10).to(device)
    total_params = sum(p.numel() for p in mini_unet.parameters())
    print(f"Mini U-Net 参数量: {total_params:,}")

    # 前向传播测试
    dummy_input = torch.randn(2, 1, 28, 28).to(device)
    output = mini_unet(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # --- Part 3: Dice Loss 演示 ---
    print("\n--- Part 3: Dice Loss 演示 ---")
    dice_loss_fn = DiceLoss(num_classes=10)
    dummy_pred = torch.randn(2, 10, 28, 28).to(device)
    dummy_target = torch.randint(0, 10, (2, 28, 28)).to(device)
    loss = dice_loss_fn(dummy_pred, dummy_target)
    print(f"Dice Loss 值: {loss.item():.4f}")

    # --- 核心概念 ---
    print("\n--- 图像分割核心概念 ---")
    print("  1. 语义分割: 为每个像素分配类别标签，不区分同类实例")
    print("  2. 实例分割: 在语义分割基础上区分同一类别的不同实例")
    print("  3. 全景分割: 语义分割 + 实例分割的统一")
    print("  4. 编码器-解码器: U-Net、DeepLabV3 等主流架构的基础")
    print("  5. 跳跃连接: 编码器特征直接传递给解码器，保留空间细节")
    print("  6. 空洞卷积: 扩大感受野而不降低分辨率（DeepLabV3）")
    print("  7. 上采样: 转置卷积或插值恢复空间尺寸")
    print("  8. Dice Loss: 基于区域重叠度的损失，对类别不平衡鲁棒")


if __name__ == "__main__":
    main()
