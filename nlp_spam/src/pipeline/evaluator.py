"""
Model evaluation module for the NLP Spam ML Pipeline.

This module provides core evaluation functionality used by the package.
For detailed evaluation with visualizations, see utils.evaluation_utils.
"""

import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

from utils.config import N_SPLITS, RANDOM_STATE
from utils.logger import get_logger, LogLevel

class Evaluator:
    """
    Core evaluator for spam detection models.
    
    This class handles basic model evaluation including cross-validation
    and metrics calculation used by the package components.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.logger = get_logger()

    def calculate_metrics(self, y_true, y_pred, pos_label=1):
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            pos_label: The label of the positive class (default: 1 for Spam)
            
        Returns:
            Dictionary with calculated metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
            'recall': recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        }
        
        return metrics

    def cross_validate_model(self, model, X, y):
        """
        Perform cross-validation using StratifiedKFold.
        
        Args:
            model: Scikit-learn Pipeline or Model to evaluate
            X: Feature matrix (text)
            y: Target variable (labels)
            
        Returns:
            Dictionary with aggregated cross-validation results
        """
        self.logger.info(f"Cross-validating {model.__class__.__name__}...")
        
        # 1. Set up Cross-Validation Strategy (Strictly StratifiedKFold)
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_strategy = "StratifiedKFold"
        split_generator = cv.split(X, y)

        fold_results = []
        
        # 2. Perform CV Loop
        for fold_idx, (train_idx, val_idx) in enumerate(split_generator, start=1):
            
            # Split data safely
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred)
            fold_results.append(metrics)
            
            self.logger.info(
                f"Fold {fold_idx}: Acc={metrics['accuracy']:.3f}, "
                f"Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}"
            )
        
        # 3. Aggregate Results
        cv_results = {}
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        # 4. MLflow Logging
        try:      
            cv_metrics = {f"cv_{k}": float(v) for k, v in cv_results.items()}
            mlflow.log_metrics(cv_metrics)
            mlflow.log_param("cv_strategy", cv_strategy)
            mlflow.log_param("cv_n_splits", N_SPLITS)
            mlflow.log_param("cv_n_samples", len(X))
        except Exception as e:
            self.logger.warning(f"MLflow logging for CV failed: {e}")
            
        self.logger.info(
            f"Average: Accuracy={cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}"
        )
        
        return cv_results

    def hyperparameter_optimization_cv(self, model, param_grid, X, y):
        """
        Perform hyperparameter optimization using GridSearchCV with StratifiedKFold.
        """
        self.logger.info(f"Optimizing hyperparameters for {model.__class__.__name__}...")
        
        # 1. Set up Cross-Validation Strategy
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_strategy = "StratifiedKFold"
        
        # 2. Configure GridSearchCV
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        
        # 3. Fit GridSearchCV
        gs.fit(X, y)
        
        # 4. Extract results
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        best_score = gs.best_score_
        
        # 5. Logging
        self.logger.info(f"Best Accuracy: {best_score:.3f}")
        
        if self.logger.level <= LogLevel.NORMAL:
             print(f"  Best parameters: {best_params}")
            
        try:
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            mlflow.log_metric("best_cv_accuracy", best_score)
            mlflow.log_param("cv_strategy", cv_strategy)
            mlflow.log_param("cv_n_splits", N_SPLITS)
        except Exception as e:
            self.logger.warning(f"MLflow logging for GridSearch failed: {e}")
        
        return best_model, best_params, best_score