# NLP_SPAM: SMS and Email Spam Classification Pipeline

A complete machine learning pipeline for spam detection across SMS and email domains, including data processing, feature engineering, model training, evaluation, and reproducible experiments.

## Overview

This repository provides a modular ML pipeline to classify messages as spam or ham across two domains (SMS and email). It includes end‑to‑end components for preprocessing, feature engineering, model training, evaluation, and scripts to run experiments consistently. The pipeline highlights domain differences, transfer learning challenges, and production‑oriented recommendations.

**Key Features:**
- **Unified Pipeline:** Consistent processing for SMS and email with domain‑aware options
- **Feature Engineering:** Token patterns, normalization, stop‑word removal, and bag‑of‑words
- **Multiple Experiments:** SMS‑only, Email‑only, Combined, and Cross‑domain transfer
- **Evaluation Metrics:** Accuracy, Precision, Recall with reproducible runs
- **Scripts + Tests:** Ready‑to‑run scripts and unit tests for core modules

## Dataset Description

**SMS Spam Collection**
- Original size: 5,574 messages; after deduplication: 5,169
- Class distribution: imbalanced (ham‑dominant)
- Characteristics: short, informal, abbreviations

**Email Spam Collection**
- Original size: 5,728 messages; after deduplication: 5,695
- Class distribution: imbalanced (ham‑dominant)
- Characteristics: longer, structured, formal language

Datasets are provided in [data/email_spam.csv](data/email_spam.csv) and [data/sms_spam.csv](data/sms_spam.csv).

## Project Structure

```
nlp_spam/
├── data/                      # Raw datasets
│   ├── email_spam.csv
│   └── sms_spam.csv
├── src/                       # Source code
│   ├── pipeline/              # Core ML pipeline components
│   │   ├── data_processor.py  # Data loading and preprocessing
│   │   ├── feature_engineer.py# Feature engineering helpers
│   │   ├── model_trainer.py   # Model training and creation
│   │   └── evaluator.py       # Model evaluation utilities
│   └── utils/                 # Utility functions
│       ├── config.py          # Configuration constants
│       ├── logger.py          # Logging utilities
│       └── utils.py           # General helpers
├── scripts/                   # Executable scripts
│   ├── run_pipeline.py        # Main pipeline execution
│   └── run_tests.py           # Test runner
├── tests/                     # Unit tests for modules
├── notebooks/                 # Jupyter notebooks (analysis)
│   ├── spam_notebook_final.ipynb
│   └── spam_todo.ipynb
├── README.md                  # Project documentation (this file)
└── pyproject.toml             # Dependencies and packaging
```

Core modules: [src/pipeline/data_processor.py](src/pipeline/data_processor.py), [src/pipeline/feature_engineer.py](src/pipeline/feature_engineer.py), [src/pipeline/model_trainer.py](src/pipeline/model_trainer.py), [src/pipeline/evaluator.py](src/pipeline/evaluator.py).

## Installation

```bash
# From the repo root
pip install -e .

# Optional: run tests to verify installation
python scripts/run_tests.py
```

Dependencies are managed via [pyproject.toml](pyproject.toml) and include pandas, scikit‑learn, nltk, numpy, matplotlib.

## Usage

### Basic Pipeline Runs

```bash
# From the repo root
python scripts/run_pipeline.py --dataset sms --model logistic
python scripts/run_pipeline.py --dataset email --model logistic

# Combined dataset
python scripts/run_pipeline.py --dataset combined --model logistic
```

### Advanced Configuration

```bash
# Adjust vectorizer options and feature limit
python scripts/run_pipeline.py --dataset sms --model logistic \
  --max-features 5000 --token-pattern "([A-Za-z]{4,}|!+)" --remove-stopwords

# Enable class balancing (oversampling)
python scripts/run_pipeline.py --dataset email --balance

# Cross-domain evaluation: train on SMS, test on Email
python scripts/run_pipeline.py --train-dataset sms --test-dataset email --model logistic

# Verbose logging
python scripts/run_pipeline.py --dataset sms --model logistic --log-level debug
```

Note: Flags above reflect typical options used by the pipeline. See [scripts/run_pipeline.py](scripts/run_pipeline.py) for the exact CLI.

### Notebooks

Interactive analyses are available in [spam_notebook_final.ipynb](spam_notebook_final.ipynb) and [spam_todo.ipynb](spam_todo.ipynb).

## Methodology

### Text Preprocessing
- **Number normalization:** Replace numeric sequences with a generic token (e.g., "numero")
- **Tokenization pattern:** Words with 4+ letters and exclamation marks (pattern: ([A-Za-z]{4,}|!+))
- **Stop words removal:** English stop words filtered out
- **Vectorization:** Bag‑of‑words via CountVectorizer with configurable `max_features`

### Model Training
- **Algorithm:** Logistic Regression (LBFGS, max_iter=1000)
- **Feature representation:** Token frequency counts
- **Class balancing:** Optional oversampling of the minority class on training splits
- **Train/test split:** 80/20 with stratification per domain

### Evaluation
- **Primary metrics:** Accuracy, Precision, Recall
- **Experiment modes:** SMS‑only, Email‑only, Combined dataset, Cross‑domain transfer
- **Analysis focus:** Precision‑Recall trade‑offs and domain transfer robustness

## Model Performance (Summary)

Representative outcomes from pipeline experiments:
- **Combined (SMS+Email):** Accuracy ≈ 95.99%, Precision ≈ 94.27%, Recall ≈ 86.71%
- **SMS‑only:** Accuracy ≈ 96.62%, Precision ≈ 98.97%, Recall ≈ 73.85%
- **Email‑only:** Accuracy ≈ 99.02%, Precision ≈ 99.26%, Recall ≈ 98.78%
- **Cross‑domain (Train SMS → Test Email):** Accuracy ≈ 70.48%, Precision ≈ 63.65%, Recall ≈ 95.53%

Key observations:
- Email classification is consistently easier due to longer, structured texts.
- SMS models demand careful tuning to maintain recall while preserving high precision.
- Cross‑domain transfer shows substantial degradation; domain‑specific models are preferred.

## Recommendations

### Deployment
- **Domain detection:** Route messages to domain‑specific models (SMS vs Email)
- **Precision priority:** Favor high precision for user‑facing applications to minimize false positives
- **Retraining cadence:** Periodically retrain to track evolving spam patterns
- **Feedback loop:** Add human review for borderline predictions near threshold

### Configurations
- **SMS:** Token pattern ([A-Za-z]{4,}|!+), `max_features=5000`, number normalization, stop‑word removal
- **Email:** Same configuration typically yields near‑perfect results
- **Multi‑domain:** Prefer separate models; combined training acceptable when unification is required

## Running Tests

```bash
python scripts/run_tests.py
```

Unit tests cover core components:
- [tests/test_data_processor.py](tests/test_data_processor.py)
- [tests/test_feature_engineer.py](tests/test_feature_engineer.py)
- [tests/test_model_trainer.py](tests/test_model_trainer.py)
- [tests/test_evaluator.py](tests/test_evaluator.py)

## Authors

- Hajar Salame
- Fatima Ezzahra Alaoui Mrani
- Ghita Ajarra
- Mateo Giraldo
- Anderson Adarve

**Date:** December 2025
