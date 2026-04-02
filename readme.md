# 📈 NIFTY 50 Market Direction Predictor (Fine-tuned LLM)

A professional-grade implementation for predicting market movement (Up/Down) of the NIFTY 50 index using a fine-tuned **DistilGPT2** language model. This project leverages **PEFT/LoRA** for efficient training on CPU-only environments and incorporates **MLflow** for robust experiment tracking and model versioning.

---

## 🚀 Overview

This repository demonstrates how to transform financial time-series data into a natural language sequence problem, allowing a causal language model to recognize market patterns and predict the next day's direction based on Open, High, and Low price points.

### ✨ Key Features
- **Data Engineering**: Automated pipeline fetching data via `yfinance` with noise-tolerant rounding and prompt formatting.
- **Efficient Fine-tuning**: Uses **LoRA (Low-Rank Adaptation)** via Hugging Face's `PEFT` library to train with just ~1.2MB of trainable parameters.
- **Low Resource Optimization**: Fully optimized for **CPU-only** training (tested on 16GB RAM) using `SFTTrainer`.
- **MLflow Tracking**: Integrated logging for precision, recall, F1-score, and model artifacts using a local SQLite backend.
- **Robust Inference**: Specialized inference script with pattern-matching extraction to eliminate LLM "hallucinations."

---

## 🛠️ Tech Stack

- **Model Architecture**: `distilbert/distilgpt2`
- **Libraries**: `transformers`, `peft`, `trl`, `mlflow`, `yfinance`, `pandas`
- **Optimization**: `LoRA` (r=16, alpha=32)
- **Environment**: Python 3.11+ (CPU-optimized)

---

## 🧪 Evaluation Metrics
| Metric | Score |
| ------ | ------ |
| Precision | 0.583 |
| F1-Score | 0.7 |
| Recall | 0.87 |

---

## 📥 Installation

```bash
# Clone the repository
# (Assuming you've already cloned)

# Install dependencies
pip install -r requirements.txt
```

---

## 🏗️ Workflow

### 1. Data Preparation
Downloads recent NIFTY 50 data and formats it into the training structure: `Date: ..., Open: ..., Output: Up/Down`.
```bash
python data_prep.py
```

### 2. Model Fine-Tuning
Trains the model on the generated `train.txt`. This process takes ~15-30 minutes on a standard CPU.
```bash
python fine_tune.py
```

### 3. Real-time Inference
Get a prediction for a specific set of market values.
```bash
python inference.py
```

---

## 📊 Sample Prompt Format
The model is trained on a "Causal Chain" format:
> **Data:** Date: 2026-03-27, Open: 23173.55, High: 23186.1, Low: 22819.6  
> **Prediction:** Up

---

## ⚠️ Disclaimer
*This project is for educational and research purposes only. Stock market predictions are inherently uncertain. Do not use this tool for financial trading without professional advice.*

---
