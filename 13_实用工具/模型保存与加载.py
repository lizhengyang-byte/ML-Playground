"""
模型保存与加载 (Model Save/Load)

模型持久化是实际应用的关键步骤：
1. joblib - sklearn模型的推荐保存方式
2. pickle - Python通用序列化
3. torch.save/load - PyTorch模型保存
"""

import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# 准备数据和训练模型
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 训练一个模型
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"模型训练完成, 测试准确率: {accuracy:.4f}")

# ============================================================
# 1. joblib 保存与加载 (sklearn推荐)
# ============================================================
import joblib

print("\n" + "=" * 60)
print("1. joblib 保存与加载")
print("=" * 60)

model_path = 'model.joblib'

# 保存
joblib.dump(model, model_path)
file_size = os.path.getsize(model_path)
print(f"模型已保存: {model_path} ({file_size} 字节)")

# 加载
model_loaded = joblib.load(model_path)
y_pred_loaded = model_loaded.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"模型已加载, 测试准确率: {accuracy_loaded:.4f}")
print(f"加载后模型类型: {type(model_loaded)}")

# 验证结果一致
y_pred_original = model.predict(X_test)
print(f"加载前后预测结果一致: {np.array_equal(y_pred_original, y_pred_loaded)}")

# ============================================================
# 2. pickle 保存与加载
# ============================================================
import pickle

print("\n" + "=" * 60)
print("2. pickle 保存与加载")
print("=" * 60)

pkl_path = 'model.pkl'

# 保存
with open(pkl_path, 'wb') as f:
    pickle.dump(model, f)
file_size_pkl = os.path.getsize(pkl_path)
print(f"模型已保存: {pkl_path} ({file_size_pkl} 字节)")

# 加载
with open(pkl_path, 'rb') as f:
    model_pkl = pickle.load(f)

y_pred_pkl = model_pkl.predict(X_test)
accuracy_pkl = accuracy_score(y_test, y_pred_pkl)
print(f"模型已加载, 测试准确率: {accuracy_pkl:.4f}")
print(f"加载前后预测结果一致: {np.array_equal(y_pred_original, y_pred_pkl)}")

# 保存多个对象
multi_path = 'model_and_scaler.pkl'
scaler = StandardScaler()
scaler.fit(X_train)

with open(multi_path, 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'feature_names': iris.feature_names}, f)

with open(multi_path, 'rb') as f:
    bundle = pickle.load(f)

print(f"\n保存多个对象:")
print(f"  键: {list(bundle.keys())}")
print(f"  模型类型: {type(bundle['model'])}")
print(f"  特征名: {bundle['feature_names']}")

# ============================================================
# 3. PyTorch 模型保存与加载
# ============================================================
print("\n" + "=" * 60)
print("3. PyTorch 模型保存与加载")
print("=" * 60)

try:
    import torch
    import torch.nn as nn

    # 定义一个简单的模型
    class SimpleNet(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, output_dim)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    # 训练模型
    torch.manual_seed(42)
    torch_model = SimpleNet(4, 3)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = torch_model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    # 测试准确率
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = torch_model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        acc_torch = (predicted.numpy() == y_test).mean()
    print(f"PyTorch模型训练完成, 测试准确率: {acc_torch:.4f}")

    # 方法1: torch.save 保存 state_dict (推荐)
    state_dict_path = 'pytorch_model_state.pt'
    torch.save(torch_model.state_dict(), state_dict_path)
    print(f"\nstate_dict已保存: {state_dict_path} ({os.path.getsize(state_dict_path)} 字节)")

    # 加载 state_dict
    torch_model_loaded = SimpleNet(4, 3)
    torch_model_loaded.load_state_dict(torch.load(state_dict_path, weights_only=True))
    torch_model_loaded.eval()

    with torch.no_grad():
        outputs = torch_model_loaded(X_test_tensor)
        _, predicted_loaded = torch.max(outputs, 1)
        acc_loaded = (predicted_loaded.numpy() == y_test).mean()
    print(f"state_dict加载后准确率: {acc_loaded:.4f}")

    # 方法2: torch.save 保存整个模型
    full_model_path = 'pytorch_model_full.pt'
    torch.save(torch_model, full_model_path)
    print(f"\n完整模型已保存: {full_model_path} ({os.path.getsize(full_model_path)} 字节)")

    # 加载完整模型
    torch_model_full = torch.load(full_model_path, weights_only=False)
    torch_model_full.eval()

    with torch.no_grad():
        outputs = torch_model_full(X_test_tensor)
        _, predicted_full = torch.max(outputs, 1)
        acc_full = (predicted_full.numpy() == y_test).mean()
    print(f"完整模型加载后准确率: {acc_full:.4f}")

    # 方法3: 保存 checkpoint (训练过程中保存)
    checkpoint_path = 'pytorch_checkpoint.pt'
    checkpoint = {
        'epoch': 100,
        'model_state_dict': torch_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'accuracy': acc_torch,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\nCheckpoint已保存: {checkpoint_path} ({os.path.getsize(checkpoint_path)} 字节)")

    # 加载 checkpoint
    checkpoint_loaded = torch.load(checkpoint_path, weights_only=False)
    print(f"Checkpoint内容: epoch={checkpoint_loaded['epoch']}, "
          f"loss={checkpoint_loaded['loss']:.4f}, accuracy={checkpoint_loaded['accuracy']:.4f}")

except ImportError as e:
    print(f"PyTorch未安装或不可用: {e}")

# ============================================================
# 4. 对比与建议
# ============================================================
print("\n" + "=" * 60)
print("4. 保存方式对比与建议")
print("=" * 60)

print(f"""
方式          适用场景                  优点                    注意事项
----          --------                  ----                    ----------
joblib        sklearn模型               高效压缩, sklearn官方推荐   需安装joblib
pickle        Python对象通用            无需额外库                安全性注意,跨版本兼容
torch.save    PyTorch模型               支持state_dict/checkpoint  指定weights_only参数

最佳实践:
  1. sklearn模型: 使用 joblib.dump/load
  2. PyTorch模型: 使用 state_dict方式, 保存/加载时保持模型定义一致
  3. 大型模型: joblib的compress参数可压缩文件大小
  4. 生产环境: 固定随机种子, 记录依赖版本, 保存完整pipeline
""")

# 清理临时文件
for f in ['model.joblib', 'model.pkl', 'model_and_scaler.pkl',
          'pytorch_model_state.pt', 'pytorch_model_full.pt', 'pytorch_checkpoint.pt']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已清理: {f}")
