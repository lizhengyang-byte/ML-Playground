# ML-Playground

机器学习学习与参考项目 — 一个覆盖机器学习全领域的代码示例集合，每个脚本独立可运行，包含完整的数据准备、模型训练、结果输出和要点总结。

## 环境要求

- Python 3.10+
- 推荐使用虚拟环境（项目内已配置 `.venv`）

```bash
# 激活虚拟环境
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

## 运行方式

每个 `.py` 文件都是独立可运行的脚本，直接执行即可：

```bash
python "02_监督学习/分类/逻辑回归.py"
python "06_深度学习/卷积神经网络_CNN.py"
python "13_实用工具/模型保存与加载.py"
```

## 项目结构

```
machine_learning/
├── 01_数据预处理/          # 缺失值处理、特征缩放、编码转换、数据清洗、数据分割
├── 02_监督学习/
│   ├── 分类/               # KNN、SVM、决策树、朴素贝叶斯、逻辑回归、随机森林、LDA、感知机
│   └── 回归/               # 线性回归、岭回归、Lasso、ElasticNet、SVR、KNN、决策树、随机森林、多项式、贝叶斯岭
├── 03_无监督学习/
│   ├── 聚类/               # KMeans、DBSCAN、层次聚类、GMM、均值漂移、谱聚类
│   ├── 降维/               # PCA、SVD、t-SNE、UMAP、自编码器降维
│   └── 关联规则/           # Apriori、FP-Growth
├── 04_半监督学习/          # 标签传播、标签扩展、自训练
├── 05_强化学习/            # Q-Learning、SARSA、DQN、REINFORCE、Actor-Critic、PPO
├── 06_深度学习/            # MLP、CNN、RNN、LSTM、GRU、Transformer、自编码器、VAE、GAN、ResNet
├── 07_集成学习/            # Bagging、Boosting、AdaBoost、Stacking、Voting、XGBoost、LightGBM、CatBoost
├── 08_自然语言处理/        # 词袋模型、TF-IDF、Word2Vec、情感分析、文本分类、NER、文本生成、文本预处理
├── 09_计算机视觉/          # 图像分类、目标检测、图像分割、数据增强、迁移学习
├── 10_时间序列/            # ARIMA、SARIMA、Prophet、指数平滑、LSTM时序预测
├── 11_特征工程/            # 特征选择、特征提取、特征重要性、特征交叉
├── 12_模型评估与选择/      # 交叉验证、评估指标、混淆矩阵、ROC曲线、学习曲线、超参数调优
├── 13_实用工具/            # 数据加载、可视化、模型保存/加载/导出、Pipeline、可解释性、日志追踪、早停回调、随机种子
├── CLAUDE.md               # 开发规范
├── requirements.txt        # 依赖清单
└── 需求文档.md              # 项目需求说明
```

## 各目录概览

### 01 数据预处理

| 文件 | 说明 |
|------|------|
| 缺失值处理.py | 均值/中位数/众数填充、删除、插值 |
| 特征缩放.py | 标准化、归一化、RobustScaler、MaxAbsScaler |
| 编码转换.py | 独热编码、标签编码、序数编码、目标编码 |
| 数据清洗.py | 异常值检测与处理、重复值、数据类型转换 |
| 数据分割.py | 训练/测试/验证集划分、分层抽样 |

### 02 监督学习

**分类**：KNN、SVM、决策树、朴素贝叶斯、逻辑回归、随机森林、LDA、感知机

**回归**：线性回归、岭回归、Lasso、ElasticNet、SVR、KNN回归、决策树、随机森林、多项式回归、贝叶斯岭回归

### 03 无监督学习

**聚类**：KMeans、DBSCAN、层次聚类、GMM、均值漂移、谱聚类

**降维**：PCA、SVD、t-SNE、UMAP、自编码器降维

**关联规则**：Apriori、FP-Growth

### 04 半监督学习

标签传播、标签扩展、自训练

### 05 强化学习

Q-Learning、SARSA、DQN、REINFORCE、Actor-Critic、PPO

### 06 深度学习

全连接网络(MLP)、卷积神经网络(CNN)、循环神经网络(RNN)、LSTM、GRU、Transformer、自编码器、变分自编码器(VAE)、生成对抗网络(GAN)、残差网络(ResNet)

### 07 集成学习

Bagging、Boosting、AdaBoost、Stacking、投票法、XGBoost、LightGBM、CatBoost

### 08 自然语言处理

词袋模型、TF-IDF、Word2Vec、情感分析、文本分类、命名实体识别、文本生成、文本预处理

### 09 计算机视觉

图像分类、目标检测、图像分割、数据增强、迁移学习

### 10 时间序列

ARIMA、SARIMA、Prophet、指数平滑、LSTM时序预测

### 11 特征工程

特征选择、特征提取、特征重要性、特征交叉

### 12 模型评估与选择

交叉验证、评估指标、混淆矩阵、ROC曲线、学习曲线、超参数调优

### 13 实用工具

| 文件 | 说明 |
|------|------|
| 数据加载.py | sklearn内置数据集、pandas读CSV、numpy读文本 |
| 可视化.py | 文本直方图、相关性矩阵、箱线图、交叉频率表 |
| 模型保存与加载.py | joblib、pickle、PyTorch checkpoint |
| 机器学习管道.py | Pipeline、ColumnTransformer、自定义转换器、网格搜索 |
| 模型可解释性.py | 特征重要性、排列重要性、部分依赖、LIME、SHAP |
| 实验日志与追踪.py | logging、结构化日志、JSON实验记录、训练过程追踪 |
| 早停与回调.py | EarlyStopping、回调机制、sklearn早停 |
| 模型导出.py | ONNX导出、TorchScript、格式对比 |
| 随机种子与可复现性.py | 多框架种子设置、全局封装、验证方法 |

## 技术栈

| 领域 | 主要库 |
|------|--------|
| 经典机器学习 | scikit-learn |
| 深度学习 | PyTorch |
| 强化学习 | gymnasium + PyTorch |
| 自然语言处理 | transformers、gensim、jieba |
| 时间序列 | statsmodels、prophet |
| 数据处理 | numpy、pandas |
| 集成学习 | xgboost、lightgbm、catboost |
| 可视化 | matplotlib、seaborn（已安装但本项目仅输出文本） |

## 代码规范

- 每个脚本独立可运行，无需外部数据文件
- 使用 `sklearn.datasets` 内置数据或合成数据
- 中文注释说明关键逻辑
- 输出文本结果，不保存图片文件
- 标注常见陷阱和参数敏感点
