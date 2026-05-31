# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Machine learning learning/reference project — a comprehensive code skeleton covering ML algorithms organized by category. The goal (per `需求文档.md`) is to fill in each empty `.py` file with working implementations and explanations.

## Language

All file names, directory names, comments, docstrings, and documentation are in **Chinese (Simplified)**. Maintain this convention.

## Environment

- **Python 3.14.4** (CPython), managed by **uv 0.11.7**
- Virtual environment at `.venv/` — activate before running anything
- Pre-installed packages: numpy, scipy, scikit-learn, matplotlib, pandas, seaborn, statsmodels, pytorch, transformers, and others
- No `pyproject.toml` or `setup.py` — dependency management is via `requirements.txt` (currently empty)

## Running Code

```bash
# Activate environment
.venv\Scripts\activate        # Windows PowerShell
source .venv/bin/activate     # Linux/Mac

# Run a single algorithm file
python "02_监督学习/分类/逻辑回归.py"
```

No build system, linter, or test suite is configured.

## Project Structure

13 numbered directories, each a self-contained topic. Every `.py` file is currently empty and needs implementation.

| Directory | Category |
|-----------|----------|
| `01_数据预处理` | Data preprocessing (cleaning, scaling, encoding, splitting) |
| `02_监督学习` | Supervised learning — `分类/` (classification) and `回归/` (regression) |
| `03_无监督学习` | Unsupervised — `聚类/` (clustering), `降维/` (dim. reduction), `关联规则/` |
| `04_半监督学习` | Semi-supervised learning |
| `05_强化学习` | Reinforcement learning |
| `06_深度学习` | Deep learning (MLP, CNN, RNN, LSTM, GRU, Transformer, GAN, VAE, ResNet) |
| `07_集成学习` | Ensemble methods (Bagging, Boosting, Stacking, XGBoost, LightGBM, CatBoost) |
| `08_自然语言处理` | NLP (BoW, TF-IDF, Word2Vec, sentiment, NER, text generation) |
| `09_计算机视觉` | Computer vision (classification, detection, segmentation, augmentation) |
| `10_时间序列` | Time series (ARIMA, SARIMA, Prophet, LSTM, exponential smoothing) |
| `11_特征工程` | Feature engineering (selection, extraction, importance, crossing) |
| `12_模型评估与选择` | Model evaluation (cross-validation, metrics, ROC, confusion matrix, tuning) |
| `13_实用工具` | Utilities (data loading, visualization, model save/load) |

## Implementation Conventions

- Each file should be **self-contained and runnable** — include imports, sample data generation/loading, model training, and visualization/output in one script
- Use **scikit-learn** as the primary library for classical ML algorithms; use **PyTorch** for deep learning
- Include Chinese comments/docstrings explaining the algorithm's core idea, key parameters, and when to use it
- Prefer `matplotlib` / `seaborn` for visualization output
- Each `__init__.py` should remain empty (namespace package)
