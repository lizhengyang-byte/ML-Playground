"""
目标检测 — 使用 torchvision 预训练 Faster R-CNN 进行目标检测

核心思路：
  Faster R-CNN 是经典的两阶段目标检测器：
  1. 区域提议网络（RPN）生成候选区域
  2. 对每个候选区域进行分类和边界框回归

  torchvision 提供了预训练的检测模型，可以直接加载使用。
  本示例展示：
  - 加载预训练 Faster R-CNN 模型
  - 用合成图像演示检测流程（边界框、类别、置信度）
  - 解释目标检测的核心概念（IoU、NMS、Anchor）

注意：实际使用时需传入真实图像。此处用随机张量演示 API 流程，
并生成一张简单的合成图像来展示完整检测管线。
"""

import torch
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from PIL import Image, ImageDraw


# ======================== 模型加载 ========================

def load_model():
    """加载预训练的 Faster R-CNN 模型（ResNet50-FPN 骨干）"""
    model = fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_V2_ResNet50_FPN_Weights.DEFAULT)
    model.eval()  # 切换到推理模式
    return model


# ======================== 图像预处理 ========================

def get_transform():
    """目标检测的图像预处理：转为张量并归一化"""
    return transforms.Compose([
        transforms.ToTensor(),
    ])


# ======================== 合成演示图像 ========================

def create_demo_image():
    """
    创建一张简单的合成图像用于演示检测流程。
    图像包含一个蓝色矩形（模拟物体）在白色背景上。
    """
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # 画一个蓝色矩形，模拟待检测物体
    draw.rectangle([50, 50, 150, 150], fill=(0, 0, 255))
    # 画一个红色圆形
    draw.ellipse([180, 180, 280, 280], fill=(255, 0, 0))
    return img


# ======================== 检测推理 ========================

def detect(model, image, transform, threshold=0.5):
    """
    对单张图像执行目标检测。

    参数:
        model: 预训练检测模型
        image: PIL Image
        transform: 图像变换
        threshold: 置信度阈值

    返回:
        detections: 检测结果列表，每项包含 boxes, labels, scores
    """
    # 预处理
    img_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度

    # 推理（不计算梯度）
    with torch.no_grad():
        predictions = model(img_tensor)

    # 提取结果
    pred = predictions[0]
    # COCO 数据集的类别名称（91 个类别，索引 0 是背景）
    coco_names = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush",
    ]

    # 过滤低置信度检测
    high_conf = pred["scores"] >= threshold
    boxes = pred["boxes"][high_conf]
    labels = pred["labels"][high_conf]
    scores = pred["scores"][high_conf]

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        label_name = coco_names[label.item()] if label.item() < len(coco_names) else f"class_{label.item()}"
        detections.append({
            "box": box.tolist(),
            "label": label_name,
            "score": score.item(),
        })
    return detections


# ======================== IoU 计算 ========================

def compute_iou(box1, box2):
    """
    计算两个边界框的 IoU（Intersection over Union）。

    IoU = 交集面积 / 并集面积
    是目标检测中评估定位精度的核心指标，也用于 NMS 去重。

    参数:
        box1, box2: [x1, y1, x2, y2] 格式的边界框
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# ======================== 非极大值抑制（NMS） ========================

def nms(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制（Non-Maximum Suppression）。

    目标检测中，同一个物体可能产生多个重叠的检测框。
    NMS 按置信度排序，保留高分框，抑制与其 IoU 过高的低分框。

    参数:
        boxes: 边界框列表 [[x1,y1,x2,y2], ...]
        scores: 对应置信度列表
        iou_threshold: IoU 超过此阈值则抑制
    返回:
        保留的框索引
    """
    if len(boxes) == 0:
        return []

    # 按置信度降序排序
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while sorted_indices:
        current = sorted_indices.pop(0)
        keep.append(current)
        # 移除与当前框 IoU 过高的框
        sorted_indices = [
            i for i in sorted_indices
            if compute_iou(boxes[current], boxes[i]) < iou_threshold
        ]

    return keep


# ======================== 主流程 ========================

def main():
    print("=" * 50)
    print("目标检测 — Faster R-CNN 演示")
    print("=" * 50)

    # 加载模型
    print("\n加载预训练 Faster R-CNN 模型...")
    model = load_model()
    transform = get_transform()
    print("模型加载完成")

    # 创建合成图像
    image = create_demo_image()
    print(f"\n合成图像尺寸: {image.size}")

    # 执行检测
    print("\n执行目标检测（合成图像，预期无法检测到真实物体）...")
    detections = detect(model, image, transform, threshold=0.1)

    if detections:
        print(f"检测到 {len(detections)} 个候选区域:")
        for i, det in enumerate(detections):
            print(f"  [{i+1}] 类别: {det['label']}, 置信度: {det['score']:.4f}, "
                  f"边界框: {[round(v, 1) for v in det['box']]}")
    else:
        print("未检测到超过阈值的目标（合成图像预期结果）")

    # 演示 IoU 计算
    print("\n--- IoU 计算演示 ---")
    box_a = [50, 50, 150, 150]
    box_b = [100, 100, 200, 200]
    box_c = [0, 0, 50, 50]
    iou_ab = compute_iou(box_a, box_b)
    iou_ac = compute_iou(box_a, box_c)
    print(f"  框 A 与框 B 的 IoU: {iou_ab:.4f}（部分重叠）")
    print(f"  框 A 与框 C 的 IoU: {iou_ac:.4f}（不重叠）")

    # 演示 NMS
    print("\n--- NMS 演示 ---")
    demo_boxes = [[50, 50, 150, 150], [55, 55, 155, 155], [200, 200, 300, 300]]
    demo_scores = [0.95, 0.90, 0.80]
    kept = nms(demo_boxes, demo_scores, iou_threshold=0.5)
    print(f"  输入 {len(demo_boxes)} 个框，NMS 后保留 {len(kept)} 个: {kept}")

    # 打印模型信息
    print("\n--- 模型信息 ---")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Faster R-CNN (ResNet50-FPN) 参数量: {total_params:,}")

    print("\n--- 目标检测核心概念 ---")
    print("  1. 锚框 (Anchor): 在特征图上预定义不同尺度和比例的候选框")
    print("  2. RPN: 区域提议网络，预测锚框是否包含物体并微调位置")
    print("  3. RoI Pooling: 将不同大小的提议区域映射为固定大小的特征")
    print("  4. IoU: 交并比，衡量预测框与真实框的重叠程度")
    print("  5. NMS: 非极大值抑制，去除冗余的重叠检测框")


if __name__ == "__main__":
    main()
