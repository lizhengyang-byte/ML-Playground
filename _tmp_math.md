## 数学原理

### 1. 实验追踪的结构化表示

每个实验可表示为元组：

$$\text{Exp}_i = (\lambda_i, D_i, \theta_i, M_i, t_i)$$

- $\lambda_i$：超参数集合
- $D_i$：数据集标识
- $\theta_i$：模型参数
- $M_i$：评估指标集合
- $t_i$：时间戳

### 2. 指标的统计显著性

对比两个模型时，单次运行的结果不可靠。应使用多次运行的统计检验：

$$\bar{s} = \frac{1}{K}\sum_{k=1}^{K} s_k, \quad \text{SE} = \frac{\sigma_s}{\sqrt{K}}$$

95% 置信区间：$\bar{s} \pm 1.96 \cdot \text{SE}$

配对 t 检验比较两个模型：

$$t = \frac{\bar{s}_A - \bar{s}_B}{\text{SE}_{diff}}, \quad df = K - 1$$

### 3. JSON 格式的实验记录

结构化存储便于程序化分析：

$$\text{record} = \{\text{params}: \lambda, \text{metrics}: M, \text{timestamp}: t\}$$

支持后续的排序、过滤、聚合分析。

### 4. 超参数搜索的实验对比

| 实验 | 参数 $\lambda$ | 指标 $M$ |
|------|---------------|----------|
| Exp1 | $\{lr=0.01, d=3\}$ | $\{acc=0.85, f1=0.83\}$ |
| Exp2 | $\{lr=0.001, d=5\}$ | $\{acc=0.89, f1=0.87\}$ |

通过排序找到最优实验：$\lambda^* = \arg\max_i M_i(\text{acc})$

### 5. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `logging.info(f"acc={acc:.4f}")` | 记录指标到日志 |
| `json.dump(record, f)` | 序列化 $\text{Exp}_i \to \text{JSON}$ |
| `pd.DataFrame(records)` | 实验记录表，行=实验，列=指标 |
| `df.sort_values("accuracy", ascending=False)` | 按 $M_i$ 排序找最优 |
| `df.groupby("lr")["acc"].mean()` | 按参数分组统计平均指标 |
