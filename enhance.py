import os
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '教学文档')
m = {}

m['08_自然语言处理/文本预处理.md'] = """
## 数学原理

### TF-IDF 权重
TF-IDF(t,d) = (count(t,d)/len(d)) * log(N/DF(t))

### 分词的数学视角
中文分词可以建模为序列标注问题：P(y|x) = prod P(y_t|x, y_{1:t-1})
CRF/HMM 是常用的分词模型。
"""

m['13_实用工具/可视化.md'] = """
## 数学原理

### 核密度估计（KDE）
f_hat(x) = (1/nh) sum K((x-x_i)/h)
用于绘制平滑的直方图（如 seaborn.kdeplot）。

### 热力图
heatmap[i][j] = value(i,j)，用于可视化混淆矩阵、相关系数矩阵等。
"""

m['13_实用工具/实验日志与追踪.md'] = """
## 数学原理

### 实验追踪的指标记录
记录训练/验证 loss 曲线、评估指标随 epoch 的变化：
loss_train(epoch), loss_val(epoch), metric(epoch)
用于诊断过拟合（train_loss下降但val_loss上升）。
"""

m['13_实用工具/机器学习管道.md'] = """
## 数学原理

### Pipeline 的数学保证
Pipeline 确保训练和推理时执行完全相同的变换：
X_test_transformed = f_transform(X_test; theta_train)
theta_train 只从训练集学习，避免数据泄露。
"""

m['13_实用工具/模型可解释性.md'] = """
## 数学原理

### SHAP 值（Shapley 值）
phi_j = sum_{S subset N\\{j}} (|S|!(|N|-|S|-1)!/|N|!) * [f(S union {j}) - f(S)]
满足：效率性 sum(phi_j) = f(x) - E[f(X)]，对称性，可加性。

### Permutation Importance
PI_j = E[L(y, f(X))] - E[L(y, f(X_{perm_j}))]
打乱特征 j 后的性能下降。
"""

m['13_实用工具/模型导出.md'] = """
## 数学原理

### ONNX 图表示
计算图 G = (V, E)，V 为算子节点（Conv, MatMul, Relu 等），E 为数据流。
每个节点：y = op(x_1, ..., x_n; params)

### 量化
float32 -> int8: x_q = round(x/scale) + zero_point
scale = (x_max - x_min) / 255
"""

m['13_实用工具/随机种子与可复现性.md'] = """
## 数学原理

### 伪随机数生成器（PRNG）
x_{n+1} = (a * x_n + c) mod m（线性同余法）
种子 seed = x_0，固定种子保证序列可复现。

### 随机性来源
1. 数据划分（train_test_split 的 random_state）
2. 模型初始化（权重随机初始化）
3. 训练过程（Dropout, mini-batch shuffle）
"""

count = 0
for rel_path, math_text in m.items():
    full_path = os.path.join(base, rel_path)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            existing = f.read()
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(existing + math_text)
        count += 1
        print(f'Enhanced: {rel_path}')
print(f'Done: {count} files')