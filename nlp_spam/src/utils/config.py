"""
Configuration for NLP Spam Detection Pipeline.


This module contains all configuration constants used throughout
the pipeline. Students don't need to modify this file.
"""


from pathlib import Path


# =============================================================================
# PROJECT PATHS
# =============================================================================


# Base project directory (automatically detected)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"


# =============================================================================
# DATA CONFIGURATION
# =============================================================================


# File names
EMAIL_FILE = "email_spam.csv"
SMS_FILE = "sms_spam.csv"


# Column names
MESSAGE ="message"
LABEL = "label"


# =============================================================================
# PREPROCESSING CONFIGURATION
TRAIN_TEST_SPLIT_SIZE =0.2
RANDOM_STATE = 3
# =============================================================================


# Tokenization and text preprocessing
TOKEN_REGEX = r"([A-Za-z]{4,}|!+)" # words>3 lettres + points d'exclamation
LOWERCASE = False
stop_words="english"            # "english" or custom list if needed
NB_FEATURES = 5000          # vocabulary size

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================


# Cross-validation splits
N_SPLITS = 5

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================


# Available model types
MODEL_TYPES = ["logistic regression", "xgboost", "lightgbm"]


# Default hyperparameter grids for optimization
# Default hyperparameter grids for quick testing
DEFAULT_PARAM_GRIDS = {
    "xgboost": {
        # Vectorizer tuning
        'vectorizer__max_features': [1000, 3000],          
        'vectorizer__ngram_range': [(1, 1)],               

        # XGBoost tuning
        'classifier__n_estimators': [50, 100],            # moins de boosting rounds
        'classifier__max_depth': [3, 5],                  # moins de profondeur
        'classifier__learning_rate': [0.1],               # valeur unique
        'classifier__subsample': [1.0],                   # valeur unique
        'classifier__colsample_bytree': [0.8]             # valeur unique
    },

    "lightgbm": {
        # Vectorizer tuning
        'vectorizer__max_features': [1000, 3000],
        'vectorizer__ngram_range': [(1, 1)],

        # LightGBM tuning
        'classifier__n_estimators': [50, 100],
        'classifier__num_leaves': [31],
        'classifier__max_depth': [10],
        'classifier__learning_rate': [0.1],
        'classifier__subsample': [1.0],
        'classifier__colsample_bytree': [0.8]
    }
}



# Random state for reproducibility
RANDOM_STATE = 42


# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================


MLFLOW_EXPERIMENT_NAME = "nlp_spam_pipeline"
MLFLOW_TRACKING_URI = "./mlruns"


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================


METRICS = ["accuracy", "precision", "recall", "f1_score"]


def get_data_file_path(filename):
    """Get full path to a data file."""
    return DATA_PATH / filename
