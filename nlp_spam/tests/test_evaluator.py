"""
Technical validation tests for NLP Evaluator class.

These tests validate the model evaluation functionality including
metrics calculation, cross-validation, and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from src.pipeline.evaluator import Evaluator

class TestEvaluator:
    """Test suite for Evaluator technical validation (NLP version)."""
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        evaluator = Evaluator()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1]) 
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        for metric in ['accuracy', 'precision', 'recall']:
            assert isinstance(metrics[metric], (int, float))
            assert 0.0 <= metrics[metric] <= 1.0

    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        evaluator = Evaluator()
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = y_true.copy()
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0

    def test_cross_validate_model_structure(self):
        """Test cross-validation structure and outputs."""
        evaluator = Evaluator()
        
        X = pd.Series(["free money", "hello friend", "win cash", "meeting"] * 5) 
        y = pd.Series([1, 0, 1, 0] * 5)
        
        model = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', LogisticRegression())
        ])
        
        cv_results = evaluator.cross_validate_model(model, X, y)
        
        assert isinstance(cv_results, dict)
        expected_metrics = ['accuracy_mean', 'accuracy_std', 'recall_mean']
        for metric in expected_metrics:
            assert metric in cv_results

    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization integration."""
        evaluator = Evaluator()
        
        # Setup Data
        X = pd.Series([
            "win money", "hello friend", "cash prize", "meeting today",
            "free offer", "how are you", "click link", "lunch time"
        ] * 4) # 32 samples to ensure CV splits work
        y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0] * 4)
        
        # Setup Pipeline
        model = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', LogisticRegression(solver='liblinear')) 
        ])
        
        # Tiny grid for speed
        param_grid = {
            'clf__C': [0.1, 1.0],      
            'vect__max_features': [10] 
        }
        
        # Run Optimization
        best_model, best_params, best_score = evaluator.hyperparameter_optimization_cv(
            model, param_grid, X, y
        )
        
        # Assertions
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)
        assert 'clf__C' in best_params
        assert 0.0 <= best_score <= 1.0

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = Evaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'calculate_metrics')

def run_evaluator_tests():
    import sys
    result = pytest.main([__file__, "-v", "--tb=short"])
    if result == 0:
        print("✅ All NLP Evaluator tests passed!")
        return True
    else:
        print("❌ Some NLP Evaluator tests failed!")
        return False

if __name__ == "__main__":
    run_evaluator_tests()