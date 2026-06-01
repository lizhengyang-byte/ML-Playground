"""
早停与回调 (Early Stopping & Callbacks)

防止过拟合、监控训练过程的核心技术：
1. PyTorch 自定义早停
2. PyTorch 回调机制
3. sklearn 中的早停
4. 完整训练流程示例
"""

import numpy as np
import copy

# ============================================================
# 1. PyTorch 自定义早停
# ============================================================
print("=" * 60)
print("1. PyTorch 自定义早停")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch 未安装，跳过")

if HAS_TORCH:
    class Callback:
        """回调基类"""
        def on_train_begin(self, logs=None): pass
        def on_train_end(self, logs=None): pass
        def on_epoch_begin(self, epoch, logs=None): pass
        def on_epoch_end(self, epoch, logs=None): pass
        def on_batch_begin(self, batch, logs=None): pass
        def on_batch_end(self, batch, logs=None): pass

    class EarlyStopping(Callback):
        """早停机制: 当验证指标不再改善时停止训练"""
        def __init__(self, patience=10, min_delta=0, mode='min', verbose=True):
            """
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善量
            mode: 'min'(损失) 或 'max'(准确率)
            verbose: 是否打印信息
            """
            self.patience = patience
            self.min_delta = min_delta
            self.mode = mode
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.best_model_state = None

        def __call__(self, score, model):
            if self.best_score is None:
                self.best_score = score
                self.best_model_state = copy.deepcopy(model.state_dict())
            elif self._is_improvement(score):
                self.best_score = score
                self.best_model_state = copy.deepcopy(model.state_dict())
                self.counter = 0
                if self.verbose:
                    print(f"    [早停] 指标改善: {score:.6f} (计数器重置)")
            else:
                self.counter += 1
                if self.verbose:
                    print(f"    [早停] 未改善 ({self.counter}/{self.patience})")
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"    [早停] 触发! 恢复最佳模型 (best={self.best_score:.6f})")

        def _is_improvement(self, score):
            if self.mode == 'min':
                return score < self.best_score - self.min_delta
            else:
                return score > self.best_score + self.min_delta

        def restore_best_model(self, model):
            """恢复最佳模型参数"""
            if self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)

    print("EarlyStopping 类定义完成")
    print(f"  参数: patience=10, min_delta=0, mode='min'")

    # 演示早停
    print("\n演示早停机制:")
    es = EarlyStopping(patience=5, verbose=True)
    fake_scores = [2.0, 1.8, 1.6, 1.5, 1.55, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57]

    model_dummy = nn.Linear(1, 1)
    for epoch, score in enumerate(fake_scores):
        print(f"  Epoch {epoch+1}: score={score:.4f}")
        es(score, model_dummy)
        if es.early_stop:
            print(f"  => 早停触发于 Epoch {epoch+1}")
            break

# ============================================================
# 2. PyTorch 回调机制
# ============================================================
print("\n" + "=" * 60)
print("2. PyTorch 回调机制")
print("=" * 60)

if HAS_TORCH:
    class MetricsLogger(Callback):
        """记录训练指标"""
        def __init__(self):
            self.history = {'epoch': [], 'loss': [], 'val_loss': []}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.history['epoch'].append(epoch)
            self.history['loss'].append(logs.get('loss', 0))
            self.history['val_loss'].append(logs.get('val_loss', 0))

    class LRScheduler(Callback):
        """学习率调度"""
        def __init__(self, optimizer, factor=0.5, patience=5):
            self.optimizer = optimizer
            self.factor = factor
            self.patience = patience
            self.best_val_loss = float('inf')
            self.wait = 0

        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.factor
                    print(f"    [LR调度] 学习率降低至 {self.optimizer.param_groups[0]['lr']:.6f}")
                    self.wait = 0

    class ModelCheckpoint(Callback):
        """模型检查点"""
        def __init__(self, filepath='best_model.pt'):
            self.filepath = filepath
            self.best_val_loss = float('inf')

        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get('val_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # 实际使用中会保存模型
                print(f"    [检查点] 验证损失改善, 模型已保存")

    class CallbackList:
        """回调管理器"""
        def __init__(self, callbacks=None):
            self.callbacks = callbacks or []

        def on_train_begin(self, logs=None):
            for cb in self.callbacks:
                cb.on_train_begin(logs)

        def on_train_end(self, logs=None):
            for cb in self.callbacks:
                cb.on_train_end(logs)

        def on_epoch_begin(self, epoch, logs=None):
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch, logs)

        def on_epoch_end(self, epoch, logs=None):
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)

    print("回调基类和常用回调定义完成:")
    print("  MetricsLogger: 记录训练指标")
    print("  LRScheduler: 学习率调度")
    print("  ModelCheckpoint: 模型检查点")
    print("  CallbackList: 回调管理器")

# ============================================================
# 3. 完整训练流程示例
# ============================================================
print("\n" + "=" * 60)
print("3. 完整训练流程（带早停+回调）")
print("=" * 60)

if HAS_TORCH:
    # 准备数据
    np.random.seed(42)
    torch.manual_seed(42)

    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

    X_train_t = torch.FloatTensor(X[:400])
    y_train_t = torch.LongTensor(y[:400])
    X_val_t = torch.FloatTensor(X[400:])
    y_val_t = torch.LongTensor(y[400:])

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 定义模型
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 配置回调
    early_stopping = EarlyStopping(patience=10, verbose=True)
    metrics_logger = MetricsLogger()
    lr_scheduler = LRScheduler(optimizer, factor=0.5, patience=8)
    checkpoint = ModelCheckpoint('best.pt')

    callbacks = CallbackList([early_stopping, metrics_logger, lr_scheduler, checkpoint])

    # 训练
    n_epochs = 100
    print(f"\n开始训练 (最多{n_epochs}个epoch):")
    callbacks.on_train_begin()

    for epoch in range(n_epochs):
        callbacks.on_epoch_begin(epoch)

        # 训练
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            val_loss = criterion(val_output, y_val_t).item()
            val_pred = val_output.argmax(dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()

        # 回调
        logs = {'loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc}
        callbacks.on_epoch_end(epoch, logs)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:>3}: loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if early_stopping.early_stop:
            print(f"\n训练在 Epoch {epoch+1} 提前停止")
            break

    callbacks.on_train_end()

    # 恢复最佳模型并评估
    early_stopping.restore_best_model(model)
    model.eval()
    with torch.no_grad():
        final_pred = model(X_val_t).argmax(dim=1)
        final_acc = (final_pred == y_val_t).float().mean().item()
    print(f"\n最佳模型验证准确率: {final_acc:.4f}")

    # 清理
    import os
    if os.path.exists('best.pt'):
        os.remove('best.pt')

# ============================================================
# 4. sklearn 中的早停
# ============================================================
print("\n" + "=" * 60)
print("4. sklearn 中的早停")
print("=" * 60)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# SGDClassifier 的 early_stopping
sgd = SGDClassifier(
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=10,  # 10个epoch无改善则停止
    validation_fraction=0.1,
    random_state=42
)
sgd.fit(X_train, y_train)
print(f"SGD (early_stopping=True): 准确率={accuracy_score(y_test, sgd.predict(X_test)):.4f}")
print(f"  实际迭代次数: {sgd.n_iter_ if isinstance(sgd.n_iter_, int) else sgd.n_iter_[0]}")

# GradientBoosting 的 n_iter_no_change
gb = GradientBoostingClassifier(
    n_estimators=200,
    n_iter_no_change=10,
    validation_fraction=0.1,
    random_state=42
)
gb.fit(X_train, y_train)
print(f"\nGBM (n_iter_no_change=10): 准确率={accuracy_score(y_test, gb.predict(X_test)):.4f}")
print(f"  实际树数量: {gb.n_estimators_}")

# ============================================================
# 5. 早停参数选择建议
# ============================================================
print("\n" + "=" * 60)
print("5. 早停参数选择建议")
print("=" * 60)

print("""
patience 选择:
  - 小数据集( <1k): patience=5~10
  - 中等数据集(1k~100k): patience=10~20
  - 大数据集( >100k): patience=20~50
  - 训练噪声大: 增大patience

min_delta 选择:
  - 默认0即可
  - 如果指标波动大, 设为指标范围的0.1%~1%

mode 选择:
  - 损失类指标: mode='min' (越小越好)
  - 准确率类指标: mode='max' (越大越好)

最佳实践:
  1. 早停是正则化手段之一, 可与其他方法组合
  2. 保存最佳模型而非最后模型
  3. 监控验证集指标, 非训练集
  4. patience太小会过早停止, 太大会浪费时间
""")
