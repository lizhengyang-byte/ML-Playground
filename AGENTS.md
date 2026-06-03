# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Purpose

**ML-Playground** — a comprehensive ML learning/reference project with 100+ self-contained `.py` scripts covering algorithms organized by category. Each file is a standalone, runnable demonstration with synthetic data, model training, evaluation, and Chinese explanatory notes.

## Language

All file names, directory names, comments, docstrings, and documentation are in **Chinese (Simplified)**. Maintain this convention.

## Environment

- **Python 3.14.4** (CPython), managed by **uv 0.11.7**
- Virtual environment at `.venv/` — activate before running anything
- Pre-installed packages: numpy, scipy, scikit-learn, matplotlib, pandas, seaborn, statsmodels, pytorch, transformers, gensim, gymnasium, xgboost, lightgbm, catboost, shap, and others
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

13 numbered directories, each a self-contained topic. Every non-init `.py` file is a complete, runnable script.

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
| `13_实用工具` | Utilities (data loading, visualization, model save/load, export, pipeline, interpretability, logging, early stopping, reproducibility) |

## Implementation Conventions

- Each file is **self-contained and runnable** — imports, sample data generation, model training, evaluation, and text output in one script
- **No visualization code** — files must not use `matplotlib`, `seaborn`, or any plotting library. All output is text only (printed metrics, parameters, predictions)
- **Library selection by domain:**
  - Classical ML (supervised/unsupervised/semi-supervised/ensemble/feature engineering/evaluation) → `scikit-learn`
  - Deep learning (CNN/RNN/LSTM/Transformer/GAN/VAE) → `PyTorch` + `torchvision`
  - Reinforcement learning → `gymnasium` (environments) + `PyTorch` (policy networks)
  - NLP → `transformers` (pretrained models), `gensim` (Word2Vec), `jieba` (Chinese tokenization)
  - Time series → `statsmodels` (ARIMA/SARIMA), `prophet`, `PyTorch` (LSTM)
  - Data preprocessing → `numpy`, `pandas`
- Include **Chinese comments** explaining the algorithm's core idea, key parameters, and when to use it
- Mark common pitfalls, parameter sensitivity, and data requirements
- Each `__init__.py` remains empty (namespace package)

## File Conventions

- File names use Chinese characters with underscores for readability (e.g., `卷积神经网络_CNN.py`)
- Each file starts with a docstring describing the algorithm
- Sections are separated by decorated comment blocks (`# ====...====`)
- Output uses `print()` for all results
- Optional dependencies use `try/except ImportError` with graceful fallback
