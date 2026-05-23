# Air Quality ML Pipeline

A machine learning pipeline for air quality prediction using time-series data with geographic features.

## Overview

Air quality is a critical environmental and health issue. PM2.5 (particulate matter ≤ 2.5 micrometers) is particularly dangerous as these tiny particles can penetrate deep into lungs and bloodstream. Using authentic measurement data from African cities, we'll train several models to predict PM2.5 concentrations.
## Dataset Description

**Data Volume**

- Training data: 8,071 observations
- Test data: 2,783 observations
- Total features: 80 (including target)

**Key variables**

Our dataset contains three main categories of features:

1. Direct Pollutant Measurements

Gases that correlate with combustion, industrial activity, or atmospheric chemistry processes generating PM2.5:

   - nitrogendioxide_tropospheric_no2_column_number_density
   - carbonmonoxide_co_column_number_density
   - formaldehyde_tropospheric_hcho_column_number_density
   - sulphurdioxide_so2_column_number_density

2. Aerosol Indicators

Features directly related to particulate matter:

   - uvaerosolindex_absorbing_aerosol_index
   - uvaerosollayerheight_aerosol_height

3. Temporal & Spatial Features

   - Cities: Lagos, Nairobi, Bujumbura, Kampala
   - Geographic coordinates: site_latitude, site_longitude
   - Date range: 2023-01-01 to 2024-02-26

**Data Quality Issues**


1. Missing values (training data):

- 7 features are missing from more than 90% of the observations.
- 48 features are missing from 30% to 65% of the observations.
- 15 features are missing from less than 2% of the observations.
10 features, including temporal and spatial features and the target, are never missing.

The excessively missing features (>70%) were dropped due to insufficient information. For other missing values, we leveraged temporal dependencies where adjacent time points are correlated, enabling city-specific imputation.



2. Spatial Coverage Imbalance

- Kampala : 61% of the data
- Nairobi : 16% of the data
- Lagos : 9% of the data
- Bujumbura : 1,3% of the data

There is a risk that the model would be biased toward Kampala's patterns and with higher uncertainty for other underrepresented cities.

## Project Structure


```
AIR-QUALITY-FGH/
├── data/
│   ├── test.cvs                # test data
│   └── train.csv               # training data
├── mlruns/
├── notebooks/
│   └── air_quality.ipynb       # The exploratory notebook code
├── scripts/
│   ├── crypto_utils.py
│   ├── run_pipeline.py         # ML Pipeline
│   └── run_tests.py            # Test runner
├── src/
│   ├── pipeline/
│   │   ├── data_processor.py   # Data preprocessing module
│   │   ├── evaluator.py        # Model evaluation module
│   │   ├── feature_engineer.py # Feature engineering module
│   │   └── model_trainer.py    # Model training module
│   └── utils/
│       └── config.py           # Configuration
├── tests/                      # Pipeline tests
├── pyproject.toml              # Project configuration
└── README.md                   # Project documentation
```




## Installation

```bash
# Extract the project files
cd air_quality

# Install dependencies and package with uv
uv sync --extra dev

# Verify installation
uv run python scripts/run_tests.py --quick
```

## Usage

### Basic Pipeline

```bash
# Run basic pipeline
uv run python scripts/run_pipeline.py

# Different feature selection methods
uv run python scripts/run_pipeline.py --model linear --method rfe --n-features 15
```

### Advanced Models

The pipeline supports three model types:
- Linear Regression: Baseline model for understanding linear relationships
- XGBoost: Industry-standard gradient boosting with optimized performance
- LightGBM: Fast gradient boosting using histogram-based algorithms

Both XGBoost and LightGBM are ensemble methods that:
- Iteratively improve predictions through boosting
- Provide robust performance on tabular data
- Offer feature importance rankings
MLflow Experiment Tracking

### MLflow Experiment Tracking

Experiments were initially run locally in VSCode and results manually recorded. The ability to compare runs systematically was limited.
MLflow helped centralizing the storage and the tracking.  

Screenshots from MLflow are on the folder MLFLOW on gitlab.

## Key Findings

- XGBoost offers best balance of performance and speed
- LightGBM achieves marginal RMSE improvement at 16x computational cost
- Linear model, despite being fastest, is completely unusable
- None of the models achieve production-ready performance

## Model Performance



| Model | Method | Features | RMSE | R² | Std Dev | Optimized | Time |
|-------|--------|----------|------|-----|---------|-----------|------|
| Linear | SelectKBest | 15 | 34.99 | **-1.976** | 12.67 | Non | 17s |
| XGBoost | SelectKBest | 15 | 27.64 | 0.090 | 14.93 | Oui | 1m 45s |
| XGBoost | RFE | 15 | 27.26 | **0.121** | 15.10 | Oui | 2m 10s |
| LightGBM | SelectKBest | 15 | 27.67 | 0.083 | 14.84 | Oui | 31m 22s |
| LightGBM | RFE | 15 | **27.22** | 0.114 | 14.98 | Oui | 34m 7s |
| LightGBM | RFE | 20 | 27.44 | 0.089 | 14.57 | Oui | — |




Best Overall: XGBoost + RFE (15 features) - R² = 0.121, RMSE = 27.26


| Approach | Prioritizes | Spatial/Temporal Features |
|----------|-------------|---------------------------|
| SelectKBest | CO, NO2, solar angles, location, year |  Yes (location, year) |
| RFE | Major pollutants (SO2, CO, NO2, O3), cloud albedo |  None selected |



SelectKBest finds location and year highly relevant (F-scores: 546 and 373), but RFE excludes them. 

Overall Performance Assessment
After extensive testing across 3 models, 2 feature selection methods, and multiple configurations:
- Best Configuration

    Model: XGBoost + RFE
    R²: 0.121 (explains only 12.1% of PM2.5 variation)
    RMSE: 27.26 ± 15.10 µg/m³
Interpretation: Model is barely better than predicting the average PM2.5 for every observation

- Critical Limitations

**Poor Predictive Power**

Even the best model (R² = 0.121) fails to explain 88% of variance
Cannot distinguish between WHO air quality categories:

Good (0-12 µg/m³)
Moderate (12-35 µg/m³)
Unhealthy (35-55 µg/m³)

RMSE of 27 µg/m³ spans multiple health categories


**Geographic Overfitting**

Kampala bias: Model learns patterns from 61% of data
Performance by city:

Kampala: R² = 0.090-0.158 (acceptable)
Nairobi: R² ≈ 0.000 (useless)
Lagos: R² = 0.012-0.022 (near useless)
Bujumbura: R² = 0.170-0.345 (best, despite only 123 samples. )


## Methodology

1. Data Preprocessing

- Loading the data
- Missing value handling:
   Forward/backward fill for temporal continuity
   Dropping features with >70% missing values
- Geographic folding: Creating city-based validation splits to prevent data leakage

2. Feature Engineering

- Temporal features: Extraction of day, month, hour, day of week
- Geographic features: Combined latitude-longitude encoding
- Categorical encoding: One-hot encoding for location variables
- Feature selection:
  SelectKBest: Statistical method using F-scores
  RFE (Recursive Feature Elimination): Model-based iterative selection

3. Model Selection Rationale

- Models tested : linar regression, LightGBM, XGboost
     Linear Regression as a baseline to understand if relationships are linear, XGBoost and LightGBM because they handle non-linear relationships and are robust to overfitting

4. Evaluation Methodology

Metrics Used:

   - R² Score: Percentage of variance explained (higher is better, max = 1.0)
   - RMSE: Root Mean Squared Error in PM2.5 units (lower is better)
   - Cross-fold stability: Standard deviation of RMSE across geographic folds

Validation Strategy:

   - Geographic cross-validation using city-based folds
   - Prevents data leakage between training and validation
   - Tests model generalization across different locations
   - The stability of the model accross the folds was taken into account

5. Hyperparameter Tuning

   - Method: GridSearchCV with GroupKFold
   - Grouping: By city to avoid data leakage
   - Parameters tuned: Learning rate, max depth, number of estimators, regularization terms

## Conclusion

This project demonstrates the critical importance of **domain knowledge** and **appropriate data** in machine learning. While advanced algorithms (XGBoost, LightGBM) improved over linear baselines, they cannot compensate for:

1. **Fundamental data limitations** (missing meteorology, traffic, temporal features)
2. **Geographic bias** (61% Kampala overrepresentation)
3. **Inappropriate problem framing** (regression vs health-category classification)

The surprisingly strong performance on Bujumbura (R² = 0.345 despite only 123 samples) suggests that **local patterns matter more than data volume**. This validates the recommendation for city-specific models.

**Potential improvements**:
1. Implement stratified oversampling to balance cities
2. Redesign as classification problem for actionable health warnings
3. Pilot city-specific models starting with Kampala and Bujumbura


---
## Authors

- **ALAOUI-MRANI Fatima-ezzahra**: Notebook, Evaluator, MLOPS, Evaluation and business impact.
- **AJARRA Ghita**: Notebook, Model trainer, MLOPS, Evaluation and business impact.
- **SALAME Hajar**: Notebook, Feature engineering, MLOPS, Evaluation and business impact.

