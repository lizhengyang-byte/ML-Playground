"""
模型导出 (Model Export)

将训练好的模型导出为标准格式，便于部署和跨平台推理：
1. ONNX 格式导出（sklearn）
2. ONNX 格式导出（PyTorch）
3. TorchScript 导出
4. 格式对比与选择建议
"""

import os
import numpy as np

# ============================================================
# 1. ONNX 格式导出（sklearn）
# ============================================================
print("=" * 60)
print("1. ONNX 格式导出（sklearn）")
print("=" * 60)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 训练sklearn模型
pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
pipe.fit(X_train, y_train)
sk_acc = accuracy_score(y_test, pipe.predict(X_test))
print(f"sklearn模型测试准确率: {sk_acc:.4f}")

# 尝试导出ONNX
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    # 定义输入类型
    initial_type = [('float_input', FloatTensorType([None, 4]))]

    # 转换
    onnx_model = convert_sklearn(pipe, initial_types=initial_type)

    # 保存
    onnx_path = 'sklearn_model.onnx'
    with open(onnx_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    file_size = os.path.getsize(onnx_path)
    print(f"ONNX模型已保存: {onnx_path} ({file_size} 字节)")

    # 使用onnxruntime推理
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        # 推理
        X_test_float = X_test.astype(np.float32)
        outputs = session.run(None, {input_name: X_test_float})
        onnx_pred = np.argmax(outputs[0], axis=1)
        onnx_acc = accuracy_score(y_test, onnx_pred)
        print(f"ONNX推理准确率: {onnx_acc:.4f}")
        print(f"与sklearn结果一致: {np.array_equal(pipe.predict(X_test), onnx_pred)}")
    except ImportError:
        print("onnxruntime 未安装, 跳过推理验证")
        print("安装: pip install onnxruntime")

except ImportError:
    print("skl2onnx 未安装, 跳过sklearn ONNX导出")
    print("安装: pip install skl2onnx")

# ============================================================
# 2. ONNX 格式导出（PyTorch）
# ============================================================
print("\n" + "=" * 60)
print("2. ONNX 格式导出（PyTorch）")
print("=" * 60)

try:
    import torch
    import torch.nn as nn

    # 定义模型
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, x):
            return self.net(x)

    # 训练
    torch.manual_seed(42)
    model = SimpleClassifier()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        torch_pred = model(X_test_t).argmax(dim=1)
        torch_acc = (torch_pred == y_test_t).float().mean().item()
    print(f"PyTorch模型测试准确率: {torch_acc:.4f}")

    # 导出ONNX
    onnx_pt_path = 'pytorch_model.onnx'
    dummy_input = torch.randn(1, 4)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_pt_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    file_size = os.path.getsize(onnx_pt_path)
    print(f"ONNX模型已保存: {onnx_pt_path} ({file_size} 字节)")

    # ONNX推理验证
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_pt_path)
        input_name = session.get_inputs()[0].name

        X_test_np = X_test.astype(np.float32)
        outputs = session.run(None, {input_name: X_test_np})
        onnx_pred = np.argmax(outputs[0], axis=1)
        onnx_acc = accuracy_score(y_test, onnx_pred)
        print(f"ONNX推理准确率: {onnx_acc:.4f}")
        print(f"与PyTorch结果一致: {np.array_equal(torch_pred.numpy(), onnx_pred)}")
    except ImportError:
        print("onnxruntime 未安装, 跳过推理验证")

except ImportError:
    print("PyTorch 未安装, 跳过")

# ============================================================
# 3. TorchScript 导出
# ============================================================
print("\n" + "=" * 60)
print("3. TorchScript 导出")
print("=" * 60)

try:
    import torch
    import torch.nn as nn

    # 方法1: torch.jit.trace (跟踪)
    print("方法1: torch.jit.trace")
    model.eval()
    example_input = torch.randn(1, 4)

    traced_model = torch.jit.trace(model, example_input)
    traced_path = 'model_traced.pt'
    traced_model.save(traced_path)
    print(f"  Traced模型已保存: {traced_path} ({os.path.getsize(traced_path)} 字节)")

    # 加载并推理
    loaded_traced = torch.jit.load(traced_path)
    with torch.no_grad():
        traced_pred = loaded_traced(X_test_t).argmax(dim=1)
        traced_acc = (traced_pred == y_test_t).float().mean().item()
    print(f"  Traced推理准确率: {traced_acc:.4f}")

    # 方法2: torch.jit.script (脚本化)
    print("\n方法2: torch.jit.script")
    scripted_model = torch.jit.script(model)
    scripted_path = 'model_scripted.pt'
    scripted_model.save(scripted_path)
    print(f"  Scripted模型已保存: {scripted_path} ({os.path.getsize(scripted_path)} 字节)")

    # 加载并推理
    loaded_scripted = torch.jit.load(scripted_path)
    with torch.no_grad():
        scripted_pred = loaded_scripted(X_test_t).argmax(dim=1)
        scripted_acc = (scripted_pred == y_test_t).float().mean().item()
    print(f"  Scripted推理准确率: {scripted_acc:.4f}")

    # 对比三种方式
    print(f"\n  三种方式准确率对比:")
    print(f"    PyTorch原始: {torch_acc:.4f}")
    print(f"    Traced:     {traced_acc:.4f}")
    print(f"    Scripted:   {scripted_acc:.4f}")

except ImportError:
    print("PyTorch 未安装, 跳过")

# ============================================================
# 4. Trace vs Script 对比
# ============================================================
print("\n" + "=" * 60)
print("4. Trace vs Script 对比")
print("=" * 60)

print("""
torch.jit.trace:
  原理: 用示例输入执行一次, 记录操作序列
  优点: 简单, 对大多数模型有效
  缺点: 不支持数据依赖的控制流 (if/for)
  适用: 固定结构的前向传播模型

torch.jit.script:
  原理: 直接解析Python/TorchScript代码
  优点: 支持控制流, 更灵活
  缺点: 不支持所有Python特性, 调试较难
  适用: 有动态控制流的模型

选择建议:
  1. 简单前向模型 → trace (更简单)
  2. 有if/for的模型 → script (更灵活)
  3. 部署到C++ → 两种都支持
  4. 移动端部署 → 优先trace
""")

# ============================================================
# 5. 导出格式对比
# ============================================================
print("=" * 60)
print("5. 导出格式对比与选择建议")
print("=" * 60)

print("""
格式        来源        适用场景              优点                    缺点
----        ----        --------              ----                    ----
ONNX        多框架      跨平台部署            标准格式,多运行时支持    需要额外转换工具
TorchScript PyTorch     PyTorch生态内部署     与PyTorch深度集成       仅限PyTorch
pickle      sklearn     sklearn模型保存       简单直接                安全性注意
joblib      sklearn     sklearn模型保存       高效压缩                仅限sklearn

实际选择:
  1. sklearn模型 → joblib (最简单) 或 ONNX (跨平台)
  2. PyTorch模型 → TorchScript (PyTorch生态) 或 ONNX (跨平台)
  3. 跨框架/跨语言 → ONNX
  4. 移动端/边缘设备 → ONNX + ONNX Runtime Mobile
  5. 生产环境 → ONNX (统一推理引擎)

部署流程:
  训练框架 → 导出 → 优化(量化/剪枝) → 部署
  PyTorch → TorchScript/ONNX → ONNX Runtime → Docker/云服务
  sklearn → joblib/ONNX → Flask/FastAPI → Docker/云服务
""")

# 清理
for f in ['sklearn_model.onnx', 'pytorch_model.onnx',
          'model_traced.pt', 'model_scripted.pt']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已清理: {f}")
