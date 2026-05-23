"""
Model evaluation module for Air Quality ML Pipeline.

This module provides core evaluation functionality used by the package.
For detailed evaluation with visualizations, see utils.evaluation_utils.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

#Import GridSearchCV for hyperparameter optimization (Workshop 3)
from sklearn.model_selection import GridSearchCV
# TODO Import MLflow (Workshop 4)
import mlflow
from utils.config import N_SPLITS, RANDOM_STATE
from utils.logger import get_logger, LogLevel


class Evaluator:
    """
    Core evaluator for air quality prediction models.
    
    This class handles basic model evaluation including cross-validation
    and metrics calculation used by the package components.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def calculate_metrics(self, y_true, y_pred):
        """
        The function evaluates how well a regression model performs by comparing 
        true values with predicted values
        
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary with calculated metrics
        """
        # TODO Calculate comprehensive regression metrics
        
        # Create metrics dictionary
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred) #Measures how much of the variation in the target the model explains
        mae = mean_absolute_error(y_true, y_pred)

   
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics
    
    def cross_validate_model(self, model, X, y, groups=None):
        """
        Perform cross-validation using GroupKFold.
        
        This method uses GroupKFold to ensure entire cities are either in training 
        OR validation, never both, preventing data leakage.
        
        Args:
            model: Scikit-learn model to evaluate
            X: Feature matrix
            y: Target variable
            groups: Grouping variable for GroupKFold (e.g., cities)
            
        Returns:
            Dictionary with cross-validation results
        """
        logger = get_logger()
        logger.info(f"Cross-validating {model.__class__.__name__}...", LogLevel.NORMAL)
        
        # TODO Set up GroupKFold cross-validation
        # If groups provided, use GroupKFold with N_SPLITS and N_SPLITS
        # Else if no groups provided, use KFold with N_SPLITS, shuffle=True and RANDOM_STATE

        if groups is not None:
           cv = GroupKFold(N_SPLITS)
           cv_strategy = "GroupKFold"
           split_generator = cv.split(X, y, groups)
        else:
           cv = KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
           cv_strategy = "KFold"
           split_generator = cv.split(X, y)

        fold_results = []
        # TODO Perform cross-validation enumerating folds
            # Split data
        for fold_idx, (train_idx, val_idx) in enumerate(split_generator, start=1):

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics and append to fold_results
            metrics = self.calculate_metrics(y_val, y_pred)
            fold_results.append(metrics)
            
            # Logging
            logger.info(
            f"Fold {fold_idx}: RMSE={metrics['rmse']:.3f}, "
            f"MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}",
            LogLevel.VERBOSE
              )
        
        # Aggregate results
        cv_results = {}
        # Enumerate metrics and calculate mean/std across folds
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        # TODO Add MLflow cross-validation metrics logging (Workshop 4) 
        try:      
            # Log cross-validation results (metrics only - must be numeric)
            cv_metrics = {f"cv_{k}": float(v) for k, v in cv_results.items()}
            # Add additional CV metadata (metrics only - must be numeric)
            mlflow.log_metrics(cv_metrics)
            # Log strategy as parameter (strings allowed in parameters)
            mlflow.log_param("cv_strategy", cv_strategy)
            mlflow.log_param("cv_n_splits", N_SPLITS)
            mlflow.log_param("cv_n_samples", len(X))
        except Exception as e:
            logger.warning(f"MLflow logging for CV failed: {e}")
        # Logging
        if logger.level >= LogLevel.NORMAL:
            print(f"  Average: RMSE={cv_results['rmse_mean']:.3f}±{cv_results['rmse_std']:.3f}")
        
        return cv_results
    
    
    def hyperparameter_optimization_cv(self, model, param_grid, X, y, groups=None):
        """
        hyperparameters = settings to choose before training a machine-learning model
        Perform hyperparameter optimization using GridSearchCV with geographic cross-validation.
        
        This method combines GridSearchCV with GroupKFold to ensure that entire cities
        are either in training OR validation during hyperparameter search, preventing data leakage.
        
        Args:
            model: Scikit-learn model to optimize
            param_grid: Dictionary of hyperparameters to search
            X: Feature matrix
            y: Target variable
            groups: Grouping variable for GroupKFold (e.g., cities)
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        logger = get_logger()
        logger.info(f"Optimizing hyperparameters for {model.__class__.__name__}...", LogLevel.NORMAL)
        
        # TODO Add hyperparameter optimization with geographic cross-validation (Workshop 3)
        # Set up cross-validation strategy
        if groups is not None:
            cv = GroupKFold(n_splits=N_SPLITS)
            cv_strategy = "GroupKFold"
        else:
            cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            cv_strategy = "KFold"
        
        # Configure GridSearchCV with geographic cross-validation
        gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",  # minimize RMSE
        cv=cv,
        n_jobs=-1,
        verbose=2 if logger.level >= LogLevel.VERBOSE else 0
        )
        
        # Fit GridSearchCV
        if groups is not None:
           gs.fit(X, y, groups=groups)
        else:
           gs.fit(X, y)
        
        # Extract results (GridSearchCV returns negative RMSE, convert to positive)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        best_score = -gs.best_score_
        
        #new changes
        # Logging
        logger.success(f"Best RMSE: {best_score:.3f}")
        if logger.level >= LogLevel.NORMAL:
            print(f"  Best parameters: {best_params}")
        
        return best_model, best_params, best_score

