"""
机器学习管道 (ML Pipeline)

Pipeline 将预处理和模型训练串联为一个对象，避免数据泄露：
1. Pipeline 基础用法
2. ColumnTransformer 处理混合类型特征
3. 自定义转换器
4. 管道与网格搜索
5. 完整的端到端管道
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# 1. Pipeline 基础用法
# ============================================================
print("=" * 60)
print("1. Pipeline 基础用法")
print("=" * 60)

from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 创建管道: 标准化 → 随机森林
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
])

pipe.fit(X_train, y_train)
acc = accuracy_score(y_test, pipe.predict(X_test))
print(f"管道训练完成, 测试准确率: {acc:.4f}")

# 访问管道步骤
print(f"管道步骤: {[name for name, _ in pipe.steps]}")
print(f"缩放器均值: {pipe.named_steps['scaler'].mean_[:3].round(4)}")

# make_pipeline 简写
pipe2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
pipe2.fit(X_train, y_train)
print(f"make_pipeline 准确率: {accuracy_score(y_test, pipe2.predict(X_test)):.4f}")

# ============================================================
# 2. ColumnTransformer 处理混合类型
# ============================================================
print("\n" + "=" * 60)
print("2. ColumnTransformer 处理混合类型特征")
print("=" * 60)

# 构造混合类型数据
np.random.seed(42)
n = 200
df = pd.DataFrame({
    '年龄': np.random.randint(18, 65, n),
    '收入': np.random.normal(50000, 15000, n).astype(int),
    '学历': np.random.choice(['本科', '硕士', '博士'], n),
    '城市': np.random.choice(['北京', '上海', '广州', '深圳'], n),
    '工作年限': np.random.randint(0, 40, n),
})

# 加入缺失值
mask = np.random.random(n) < 0.05
df.loc[mask, '收入'] = np.nan

# 模拟标签
y = (df['收入'].fillna(df['收入'].median()) > 50000).astype(int).values

print(f"数据形状: {df.shape}")
print(f"缺失值:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# 定义数值和分类特征
numeric_features = ['年龄', '收入', '工作年限']
categorical_features = ['学历', '城市']

# 数值特征管道: 填充缺失值 → 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类特征管道: 填充缺失值 → 独热编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='未知')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 组合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 完整管道
full_pipe = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=50, random_state=42))
full_pipe.fit(df, y)

# 交叉验证
scores = cross_val_score(full_pipe, df, y, cv=5, scoring='accuracy')
print(f"\n5折交叉验证: {scores.mean():.4f} ± {scores.std():.4f}")

# 预测新数据
new_data = pd.DataFrame({
    '年龄': [30],
    '收入': [60000],
    '学历': ['硕士'],
    '城市': ['北京'],
    '工作年限': [5]
})
pred = full_pipe.predict(new_data)
print(f"新样本预测: {'高收入' if pred[0] == 1 else '低收入'}")

# ============================================================
# 3. 自定义转换器
# ============================================================
print("\n" + "=" * 60)
print("3. 自定义转换器")
print("=" * 60)

class OutlierClipper(BaseEstimator, TransformerMixin):
    """将异常值裁剪到指定分位数范围"""
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_clipped

class FeatureCreator(BaseEstimator, TransformerMixin):
    """创建交互特征"""
    def __init__(self, method='multiply'):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.method == 'multiply' and X.shape[1] >= 2:
            new_feat = (X[:, 0] * X[:, 1]).reshape(-1, 1)
            return np.hstack([X, new_feat])
        return X

# 使用自定义转换器
pipe_custom = make_pipeline(
    OutlierClipper(lower_percentile=1, upper_percentile=99),
    StandardScaler(),
    LogisticRegression(max_iter=200)
)

pipe_custom.fit(X_train, y_train)
acc_custom = accuracy_score(y_test, pipe_custom.predict(X_test))
print(f"自定义转换器管道准确率: {acc_custom:.4f}")

# ============================================================
# 4. 管道与网格搜索
# ============================================================
print("\n" + "=" * 60)
print("4. 管道与网格搜索")
print("=" * 60)

pipe_search = make_pipeline(
    StandardScaler(),
    SelectKBest(f_classif),
    LogisticRegression(max_iter=500)
)

# 参数名格式: 步骤名__参数名
param_grid = {
    'selectkbest__k': [2, 3, 4],
    'logisticregression__C': [0.1, 1.0, 10.0]
}

gs = GridSearchCV(pipe_search, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)

print(f"最佳参数: {gs.best_params_}")
print(f"最佳CV准确率: {gs.best_score_:.4f}")
print(f"测试准确率: {gs.best_estimator_.score(X_test, y_test):.4f}")

# ============================================================
# 5. 完整的端到端管道
# ============================================================
print("\n" + "=" * 60)
print("5. 完整端到端管道")
print("=" * 60)

# 构造更复杂的数据
np.random.seed(42)
n = 300
df_full = pd.DataFrame({
    '数值1': np.random.randn(n),
    '数值2': np.random.randn(n) * 2 + 1,
    '类别A': np.random.choice(['X', 'Y', 'Z'], n),
    '类别B': np.random.choice(['甲', '乙'], n),
})

# 加入缺失值
df_full.loc[np.random.choice(n, 10, replace=False), '数值1'] = np.nan
df_full.loc[np.random.choice(n, 5, replace=False), '类别A'] = np.nan

y_full = ((df_full['数值1'].fillna(0) + df_full['数值2']) > 1).astype(int)

# 构建完整管道
preprocessor_full = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('clip', OutlierClipper()),
        ('scaler', StandardScaler())
    ]), ['数值1', '数值2']),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='缺失')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), ['类别A', '类别B'])
])

final_pipe = make_pipeline(preprocessor_full, RandomForestClassifier(n_estimators=100, random_state=42))

scores_full = cross_val_score(final_pipe, df_full, y_full, cv=5, scoring='accuracy')
print(f"完整管道5折CV: {scores_full.mean():.4f} ± {scores_full.std():.4f}")

# ============================================================
# 6. Pipeline 优点总结
# ============================================================
print("\n" + "=" * 60)
print("6. Pipeline 优点总结")
print("=" * 60)

print("""
Pipeline 核心价值:
  1. 防止数据泄露: 预处理只在训练集上fit, 对测试集只transform
  2. 代码简洁: 一个对象包含完整的数据处理+模型
  3. 方便部署: 保存一个joblib文件即可
  4. 网格搜索: 统一调参, 包括预处理参数

使用建议:
  1. 所有sklearn项目都应使用Pipeline
  2. ColumnTransformer处理混合类型数据
  3. 自定义转换器继承BaseEstimator+TransformerMixin
  4. 参数搜索用 双下划线 格式: 步骤名__参数名
""")
