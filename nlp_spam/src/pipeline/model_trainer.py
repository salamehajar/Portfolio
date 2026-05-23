"""
Model Trainer Module for NLP Spam Detection Pipeline.

This module handles the training of machine learning models for spam detection.
Based on the logistic regression approach developed in the exploratory notebook.

Classes:
    ModelTrainer: Handles model training and prediction
"""

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from typing import Any, Optional
import numpy as np
import pickle
from pathlib import Path
import mlflow
import mlflow.sklearn
from utils.logger import get_logger

class ModelTrainer:
    """
    Trains and manages machine learning models for spam detection.
    
    This class encapsulates the model training logic using Logistic Regression,
    as determined during the exploratory analysis phase. It supports training,
    prediction, and model persistence.
    
    Attributes
    ----------
    max_iter : int
        Maximum number of iterations for the logistic regression solver
    random_state : int
        Random seed for reproducibility
    model : LogisticRegression
        The trained logistic regression model (None before training)
    solver : str
        Solver algorithm for optimization (default: 'lbfgs')
    
    Examples
    --------
     from sklearn.feature_extraction.text import CountVectorizer
     vectorizer = CountVectorizer(max_features=100)
     X_train = vectorizer.fit_transform(train_messages)
     trainer = ModelTrainer(max_iter=1000, random_state=42)
     trainer.train(X_train, train_labels)
     predictions = trainer.predict(X_test)
    """
    
    def __init__(
        self, 
        model_type: str = "logistic regression",
        max_iter: int = 1000, 
        random_state: int = 42,
        solver: str = 'lbfgs'
    ):
        
        """
        Initialize the ModelTrainer.
        
        Parameters
        ----------
        max_iter : int, default=1000
            Maximum number of iterations for convergence.
            Higher values may improve convergence on complex problems
            but increase training time.
        random_state : int, default=42
            Random seed for reproducibility of results across runs
        solver : str, default='lbfgs'
            Algorithm to use for optimization. Options include:
            - 'lbfgs': Good for small to medium datasets
            - 'liblinear': Good for small datasets
            - 'saga': Good for very large datasets
        """
        self.logger = get_logger()
        self.model_type = model_type.lower()
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        self.model: Optional[Any] = None
        
    def train(self, X_train, y_train) -> 'ModelTrainer':
        """
        
        Train the logistic regression model on the provided data.
        
        Fits a binary logistic regression classifier to distinguish between
        spam (positive class) and legitimate messages (negative class).
        The model learns weights for each feature to maximize prediction accuracy.
        
        Parameters
        ----------
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training feature matrix. Typically the output of a vectorizer
            (e.g., CountVectorizer or TfidfVectorizer)
        y_train : array-like, shape (n_samples,)
            Target labels for training data (0 for ham, 1 for spam)
            
        Returns
        -------
        self : ModelTrainer
            Returns self to allow method chaining
            
        Raises
        ------
        ValueError
            If X_train or y_train are empty or have mismatched dimensions
            
        Notes
        -----
        The model uses L2 regularization by default to prevent overfitting.
        Training time scales with the number of features and iterations.
        
        Examples
        --------
         trainer = ModelTrainer(max_iter=1000)
         trainer.train(X_train, y_train)
         print(f"Model trained with {len(trainer.model.coef_[0])} features")
        """
        
        if X_train is None or y_train is None:
            raise ValueError("Training data and labels cannot be None")
        
        # Check for empty data
        if len(y_train) == 0:
            raise ValueError("Training data is empty")
        
        # Start MLflow run if not active
        if mlflow.active_run() is None:
            mlflow.start_run()

    
        
        # Initialize model based on type
        if self.model_type == "logistic regression":
            self.logger.info("Training Logistic Regression...")
            self.model = LogisticRegression(
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver=self.solver
            )
        elif self.model_type == "xgboost":
            self.logger.info("Training XGBoost...")
            self.model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0   
            )
        elif self.model_type == "lightgbm":
            self.logger.info("Training LightGBM...")
            self.model = LGBMClassifier(
                objective='binary',
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test) -> np.ndarray:
        """
        
        Make predictions on test data.
        
        Uses the trained model to predict binary class labels for new messages.
        
        Parameters
        ----------
        X_test : array-like or sparse matrix, shape (n_samples, n_features)
            Test feature matrix with the same feature dimensionality as training data
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Predicted class labels (0 for ham, 1 for spam)
            
        Raises
        ------
        ValueError
            If the model has not been trained yet or if X_test is None
        
        Examples
        --------
        predictions = trainer.predict(X_test)
        spam_count = np.sum(predictions == 1)
        print(f"Detected {spam_count} spam messages")
        """
        
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        if X_test is None:
            raise ValueError("Test data cannot be None")
        
        predictions = self.model.predict(X_test)
        return predictions
    
    def predict_proba(self, X_test) -> np.ndarray:
        """
        
        Predict class probabilities for test data.
        
        Returns probability estimates for each class, useful for threshold tuning
        and confidence-based decision making.
        
        Parameters
        ----------
        X_test : array-like or sparse matrix, shape (n_samples, n_features)
            Test feature matrix
            
        Returns
        -------
        probabilities : np.ndarray, shape (n_samples, 2)
            Predicted probabilities for each class.
            probabilities[:, 0] = P(ham)
            probabilities[:, 1] = P(spam)
            
        Raises
        ------
        ValueError
            If the model has not been trained yet or if X_test is None
            
        Examples
        --------
        probs = trainer.predict_proba(X_test)
        spam_probs = probs[:, 1]
        high_confidence_spam = X_test[spam_probs > 0.9]
        """
        
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        if X_test is None:
            raise ValueError("Test data cannot be None")
        
        probabilities = self.model.predict_proba(X_test)
        return probabilities
    
    def get_feature_importance(self) -> np.ndarray:
        
        """
        Get the learned feature coefficients (weights).
        
        Returns the model's learned weights for each feature, which indicate
        the importance and direction of influence for spam classification.
        Positive weights push toward spam, negative weights toward ham.
        
        Returns
        -------
        coefficients : np.ndarray, shape (n_features,)
            Learned coefficient values for each feature
            
        Raises
        ------
        ValueError
            If the model has not been trained yet
        
        Examples
        --------
         coeffs = trainer.get_feature_importance()
        top_spam_features = np.argsort(coeffs)[-10:]  # Top 10 spam indicators
        """
        
        
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if self.model_type == "logistic regression":
            return self.model.coef_[0]
        elif self.model_type in ["xgboost", "lightgbm"]:
            return self.model.feature_importances_
        else:
            raise ValueError(f"Feature importance not supported for {self.model_type}")
    
    def save_model(self, filepath: Path) -> None:
        """
        Save the trained model to disk.
        
        Persists the trained model using pickle serialization for later use
        in production or further analysis.
        
        Parameters
        ----------
        filepath : Path or str
            Path where the model should be saved (including filename)
            
        Raises
        ------
        ValueError
            If the model has not been trained yet
        IOError
            If there are issues writing to the specified path
            
        Examples
        --------
        >>> trainer.save_model(Path("models/spam_classifier.pkl"))
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath: Path) -> 'ModelTrainer':
        """
        Load a trained model from disk.
        
        Loads a previously saved model, allowing you to make predictions
        without retraining.
        
        Parameters
        ----------
        filepath : Path or str
            Path to the saved model file
            
        Returns
        -------
        self : ModelTrainer
            Returns self to allow method chaining
            
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        IOError
            If there are issues reading the file
            
        Examples
        --------
        >>> trainer = ModelTrainer()
        >>> trainer.load_model(Path("models/spam_classifier.pkl"))
        >>> predictions = trainer.predict(X_test)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        return self
    
    def is_trained(self) -> bool:
        """
        Check if the model has been trained.
        
        Returns
        -------
        bool
            True if the model has been trained, False otherwise
            
        Examples
        --------
        >>> trainer = ModelTrainer()
        >>> print(trainer.is_trained())  # False
        >>> trainer.train(X_train, y_train)
        >>> print(trainer.is_trained())  # True
        """
        return self.model is not None
