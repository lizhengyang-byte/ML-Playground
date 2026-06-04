"""
实验日志与追踪 (Experiment Logging & Tracking)

记录实验参数、指标和过程，便于对比和复现：
1. Python logging 模块基础
2. 结构化实验日志
3. JSON 实验记录
4. 对比实验结果
5. 轻量级实验追踪器
"""

import os
import json
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# 1. Python logging 模块基础
# ============================================================
print("=" * 60)
print("1. Python logging 模块基础")
print("=" * 60)

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('ml_experiment')

logger.info("实验开始")
logger.warning("学习率设置较高, 可能不稳定")
logger.error("这不是真正的错误, 只是演示")

# 日志级别
print(f"\n日志级别从低到高: DEBUG < INFO < WARNING < ERROR < CRITICAL")
print(f"设置 level=logging.INFO 时, DEBUG 不会输出")

# ============================================================
# 2. 结构化实验日志
# ============================================================
print("\n" + "=" * 60)
print("2. 结构化实验日志")
print("=" * 60)

class ExperimentLogger:
    """简单的实验日志记录器"""
    def __init__(self, experiment_name):
        self.name = experiment_name
        self.start_time = time.time()
        self.params = {}
        self.metrics = {}
        self.notes = []

        self.logger = logging.getLogger(f'ml.{experiment_name}')
        self.logger.info(f"=== 实验 [{experiment_name}] 开始 ===")

    def log_params(self, **kwargs):
        """记录超参数"""
        self.params.update(kwargs)
        param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"参数: {param_str}")

    def log_metric(self, name, value, step=None):
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({'step': step, 'value': value})

    def log_note(self, note):
        """记录备注"""
        self.notes.append(note)
        self.logger.info(f"备注: {note}")

    def summary(self):
        """输出实验摘要"""
        elapsed = time.time() - self.start_time
        print(f"\n  实验 [{self.name}] 摘要:")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  参数: {self.params}")
        for name, values in self.metrics.items():
            final_val = values[-1]['value'] if values else None
            print(f"  {name}: {final_val}")
        if self.notes:
            print(f"  备注: {self.notes}")

# 使用
exp = ExperimentLogger("iris_rf_baseline")
exp.log_params(n_estimators=100, max_depth=None, random_state=42)

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

exp.log_metric('accuracy_mean', scores.mean())
exp.log_metric('accuracy_std', scores.std())
exp.log_note("baseline模型, 未调参")
exp.summary()

# ============================================================
# 3. JSON 实验记录
# ============================================================
print("\n" + "=" * 60)
print("3. JSON 实验记录")
print("=" * 60)

class ExperimentTracker:
    """基于JSON的实验追踪器"""
    def __init__(self, log_dir='experiments'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_experiment = None

    def start_experiment(self, name, params=None):
        """开始新实验"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"{name}_{timestamp}"
        self.current_experiment = {
            'id': exp_id,
            'name': name,
            'timestamp': timestamp,
            'params': params or {},
            'metrics': {},
            'status': 'running'
        }
        print(f"  开始实验: {exp_id}")
        return exp_id

    def log_metrics(self, **metrics):
        """记录指标"""
        if self.current_experiment:
            for k, v in metrics.items():
                self.current_experiment['metrics'][k] = v

    def finish_experiment(self, status='completed'):
        """结束并保存实验"""
        if self.current_experiment:
            self.current_experiment['status'] = status
            self.current_experiment['duration'] = time.time()

            filepath = self.log_dir / f"{self.current_experiment['id']}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.current_experiment, f, indent=2, ensure_ascii=False)
            print(f"  实验已保存: {filepath}")
            self.current_experiment = None

    def load_experiment(self, filepath):
        """加载实验记录"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compare_experiments(self, exp_files):
        """对比多个实验"""
        experiments = []
        for f in exp_files:
            data = self.load_experiment(f)
            experiments.append(data)

        print(f"\n  实验对比 ({len(experiments)}个):")
        print(f"  {'实验名':<25} {'状态':<10} {'指标'}")
        print(f"  {'-'*25} {'-'*10} {'-'*30}")
        for exp in experiments:
            metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in exp.get('metrics', {}).items())
            print(f"  {exp['name']:<25} {exp['status']:<10} {metrics_str}")

# 使用
tracker = ExperimentTracker('my_experiments')

# 实验1
tracker.start_experiment('rf_baseline', {'n_estimators': 100})
tracker.log_metrics(accuracy=0.95, f1=0.94)
tracker.finish_experiment()

# 实验2
tracker.start_experiment('rf_tuned', {'n_estimators': 200, 'max_depth': 5})
tracker.log_metrics(accuracy=0.97, f1=0.96)
tracker.finish_experiment()

# 对比
import glob
exp_files = sorted(glob.glob('my_experiments/*.json'))
if exp_files:
    tracker.compare_experiments(exp_files)

# ============================================================
# 4. 训练过程追踪
# ============================================================
print("\n" + "=" * 60)
print("4. 训练过程追踪")
print("=" * 60)

# 模拟训练过程的日志
np.random.seed(42)

class TrainingLogger:
    """记录训练过程中的损失和指标变化"""
    def __init__(self):
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

    def log_epoch(self, epoch, train_loss, val_loss, val_acc):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

    def print_progress(self, epoch, total_epochs):
        """打印训练进度"""
        bar_width = 30
        progress = (epoch + 1) / total_epochs
        filled = int(bar_width * progress)
        bar = '#' * filled + '-' * (bar_width - filled)

        train_loss = self.history['train_loss'][-1]
        val_loss = self.history['val_loss'][-1]
        val_acc = self.history['val_acc'][-1]

        print(f"  Epoch {epoch+1:>3}/{total_epochs} |{bar}| "
              f"loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    def best_epoch(self):
        """找到最佳epoch"""
        best_idx = np.argmin(self.history['val_loss'])
        return {
            'epoch': self.history['epoch'][best_idx],
            'train_loss': self.history['train_loss'][best_idx],
            'val_loss': self.history['val_loss'][best_idx],
            'val_acc': self.history['val_acc'][best_idx]
        }

# 模拟训练
train_log = TrainingLogger()
base_loss = 2.0
for epoch in range(20):
    train_loss = base_loss * np.exp(-0.15 * epoch) + np.random.normal(0, 0.02)
    val_loss = base_loss * np.exp(-0.12 * epoch) + np.random.normal(0, 0.03)
    val_acc = min(0.5 + 0.025 * epoch + np.random.normal(0, 0.01), 0.99)
    train_log.log_epoch(epoch, train_loss, val_loss, val_acc)
    if (epoch + 1) % 5 == 0:
        train_log.print_progress(epoch, 20)

best = train_log.best_epoch()
print(f"\n  最佳epoch: {best['epoch']+1}, val_loss={best['val_loss']:.4f}, val_acc={best['val_acc']:.4f}")

# ============================================================
# 5. 实验管理最佳实践
# ============================================================
print("\n" + "=" * 60)
print("5. 实验管理最佳实践")
print("=" * 60)

print("""
实验记录应包含:
  1. 超参数: 所有影响结果的配置
  2. 数据信息: 数据集名称、大小、划分方式
  3. 环境信息: Python版本、库版本、硬件(CPU/GPU)
  4. 指标: 训练/验证/测试集上的各项指标
  5. 时间: 开始时间、结束时间、耗时
  6. 随机种子: 保证可复现
  7. 代码版本: git commit hash (如有)

工具推荐 (从轻到重):
  1. JSON/CSV手动记录 → 适合小项目、学习
  2. TensorBoard → PyTorch/TF官方支持, 可视化好
  3. MLflow → 开源, 功能全面, 支持模型管理
  4. Weights & Biases → 云服务, 协作方便

命名规范:
  实验名 = 模型_数据集_关键参数
  例如: rf_iris_n100 / bert_sst2_lr2e-5
""")

# 清理
import shutil
if os.path.exists('my_experiments'):
    shutil.rmtree('my_experiments')
    print("已清理临时实验目录")
