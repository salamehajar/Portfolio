"""
Model training module for Air Quality ML Pipeline.

This module handles model training, comparison and evaluation.
Students need to complete the TODO sections.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# TODO Import additional models (Workshop 3)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# TODO Import MLflow (Workshop 4)
import mlflow
from utils.config import MODEL_TYPES, RANDOM_STATE, TARGET_COL, CITY_COL
from .evaluator import Evaluator
from utils.logger import get_logger



class ModelTrainer:
    """
    Model trainer for air quality prediction.
    
    Handles training of different regression models, model comparison,
    and model persistence.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.trained_models = {}
        self.evaluator = Evaluator()
        self.best_model = None
        self.best_model_name = None
    
    def create_model(self, model_type, **params):
        """
        Create a model instance of the specified type.
        
        Args:
            model_type: Type of model ('linear', ...)
            **params: Model-specific parameters
            
        Returns:
            Initialized model instance
        """
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Available: {MODEL_TYPES}")
        
        # Create model instance based on model_type
        if model_type == 'linear':
            model = LinearRegression(**params)
        # TODO Add XGBoost and LightGBM model creation (Workshop 3)
        # XGBoost
        elif model_type == 'xgboost':
            try:
                model = XGBRegressor(
                objective='reg:squarederror',
                random_state=RANDOM_STATE,
                **params
            )      
            except Exception as e:
                raise ImportError("XGBoost is not installed or failed to initialize.") from e

    # LightGBM
        elif model_type == 'lightgbm':
            try:
                 model = LGBMRegressor(
                objective='regression',
                verbose=-1,
                random_state=RANDOM_STATE,
                **params
            )
            except Exception as e:
                raise ImportError("LightGBM is not installed or failed to initialize.") from e

        else:
            raise ValueError(f"Unknown model type: {model_type}. Available: {MODEL_TYPES}")

        return model
    
    def train_single_model(self, X, y, model_type='linear', **model_params):
        """
        Train a single model.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to train
            **model_params: Model-specific parameters
            
        Returns:
            Trained model instance
        """
        logger = get_logger()
        logger.substep(f"Training {model_type.title()} Model")
        model = self.create_model(model_type, **model_params)

        # TODO Add MLflow parameter logging (Workshop 4)
        if mlflow.active_run():
            mlflow.log_params({
                "model_type": model_type,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
                **model_params  # log all additional model hyperparameters
    })
        # TODO Train the model
        model.fit(X, y)
        # TODO Store trained model
        self.trained_models[model_type] = model
        # TODO Calculate training score
        train_score = model.score(X, y)
        # Logging
        with logger.indent():
            logger.model_info(f"Training R² score: {train_score:.4f}")
        
        logger.success(f"{model_type.title()} model training completed")

        return model
    
    
    def predict(self, model, X):
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        logger = get_logger()
        
        # TODO Make predictions using model
        predictions = model.predict(X)
        # Logging
        logger.success(f"Generated {len(predictions)} predictions")
        
        return predictions
