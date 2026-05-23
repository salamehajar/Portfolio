"""
Unit tests for the ModelTrainer class.

Tests cover model training, prediction, persistence, and error handling
to ensure robustness of the spam detection pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import tempfile
import shutil
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.model_trainer import ModelTrainer


@pytest.fixture
def sample_text_data():
    """
    Create sample text data for testing.
    
    Returns a realistic spam/ham dataset with balanced classes
    and clear distinguishing features.
    """
    messages = [
        "Win a free iPhone now! Call immediately!",
        "Hi, how are you doing today?",
        "URGENT: You've won $1000000! Click here!",
        "Let's meet for coffee tomorrow",
        "Free money! Limited time offer!!!",
        "Can you send me the report?",
        "Congratulations! You won the lottery!",
        "Thanks for your email yesterday",
        "Act now! Special discount available!",
        "See you at the meeting"
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham
    
    return messages, labels


@pytest.fixture
def vectorized_data(sample_text_data):
    """
    Create vectorized features from sample text data.
    
    Uses CountVectorizer to convert text into numerical features
    suitable for model training.
    """
    messages, labels = sample_text_data
    
    vectorizer = CountVectorizer(max_features=50)
    X = vectorizer.fit_transform(messages)
    y = np.array(labels)
    
    return X, y, vectorizer


@pytest.fixture
def trained_model(vectorized_data):
    """
    Create a pre-trained model for testing prediction methods.
    """
    X, y, vectorizer = vectorized_data
    trainer = ModelTrainer(max_iter=100, random_state=42)
    trainer.train(X, y)
    
    return trainer, X, y, vectorizer


@pytest.fixture
def temp_model_dir():
    """
    Create a temporary directory for model persistence tests.
    
    Automatically cleaned up after tests complete.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestModelTrainerInitialization:
    """Tests for ModelTrainer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test that ModelTrainer initializes with default parameters."""
        trainer = ModelTrainer()
        
        assert trainer.max_iter == 1000
        assert trainer.random_state == 42
        assert trainer.solver == 'lbfgs'
        assert trainer.model is None
        assert not trainer.is_trained()
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        trainer = ModelTrainer(max_iter=500, random_state=123, solver='liblinear')
        
        assert trainer.max_iter == 500
        assert trainer.random_state == 123
        assert trainer.solver == 'liblinear'
        assert trainer.model is None
    
    def test_is_trained_before_training(self):
        """Test that is_trained returns False before training."""
        trainer = ModelTrainer()
        assert trainer.is_trained() is False


class TestModelTraining:
    """Tests for model training functionality."""
    
    def test_train_success(self, vectorized_data):
        """Test successful model training."""
        X, y, _ = vectorized_data
        trainer = ModelTrainer(max_iter=100, random_state=42)
        
        result = trainer.train(X, y)
        
        # Check method chaining
        assert result is trainer
        # Check model is trained
        assert trainer.model is not None
        assert trainer.is_trained()
        # Check model type
        assert isinstance(trainer.model, LogisticRegression)
    
    def test_train_with_none_data(self):
        """Test that training with None data raises ValueError."""
        trainer = ModelTrainer()
        
        with pytest.raises(ValueError, match="Training data and labels cannot be None"):
            trainer.train(None, None)
    
    def test_train_with_empty_data(self, vectorized_data):
        """Test that training with empty data raises ValueError."""
        X, _, _ = vectorized_data
        trainer = ModelTrainer()
        
        with pytest.raises(ValueError, match="Training data is empty"):
            trainer.train(X, np.array([]))
    
    def test_train_creates_model_with_correct_parameters(self, vectorized_data):
        """Test that trained model has correct configuration."""
        X, y, _ = vectorized_data
        trainer = ModelTrainer(max_iter=200, random_state=99)
        trainer.train(X, y)
        
        assert trainer.model.max_iter == 200
        assert trainer.model.random_state == 99
    
    def test_train_reproducibility(self, vectorized_data):
        """Test that training with same random_state produces identical results."""
        X, y, _ = vectorized_data
        
        trainer1 = ModelTrainer(max_iter=100, random_state=42)
        trainer1.train(X, y)
        
        trainer2 = ModelTrainer(max_iter=100, random_state=42)
        trainer2.train(X, y)
        
        # Compare model coefficients
        np.testing.assert_array_almost_equal(
            trainer1.model.coef_,
            trainer2.model.coef_
        )


class TestModelPrediction:
    """Tests for model prediction functionality."""
    
    def test_predict_success(self, trained_model):
        """Test successful prediction on test data."""
        trainer, X, _, _ = trained_model
        
        predictions = trainer.predict(X)
        
        assert predictions is not None
        assert len(predictions) == X.shape[0]
        assert predictions.dtype in [np.int32, np.int64]
        # Check predictions are binary (0 or 1)
        assert set(np.unique(predictions)).issubset({0, 1})
    
    def test_predict_without_training(self, vectorized_data):
        """Test that prediction without training raises ValueError."""
        X, _, _ = vectorized_data
        trainer = ModelTrainer()
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            trainer.predict(X)
    
    def test_predict_with_none_data(self, trained_model):
        """Test that prediction with None data raises ValueError."""
        trainer, _, _, _ = trained_model
        
        with pytest.raises(ValueError, match="Test data cannot be None"):
            trainer.predict(None)
    
    def test_predict_proba_success(self, trained_model):
        """Test successful probability prediction."""
        trainer, X, _, _ = trained_model
        
        probabilities = trainer.predict_proba(X)
        
        assert probabilities is not None
        assert probabilities.shape == (X.shape[0], 2)
        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(
            probabilities.sum(axis=1),
            np.ones(X.shape[0])
        )
        # Check probabilities are in [0, 1]
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
    
    def test_predict_proba_without_training(self, vectorized_data):
        """Test that predict_proba without training raises ValueError."""
        X, _, _ = vectorized_data
        trainer = ModelTrainer()
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            trainer.predict_proba(X)
    
    def test_predict_proba_with_none_data(self, trained_model):
        """Test that predict_proba with None data raises ValueError."""
        trainer, _, _, _ = trained_model
        
        with pytest.raises(ValueError, match="Test data cannot be None"):
            trainer.predict_proba(None)
    
    def test_predictions_match_probabilities(self, trained_model):
        """Test that hard predictions match argmax of probabilities."""
        trainer, X, _, _ = trained_model
        
        predictions = trainer.predict(X)
        probabilities = trainer.predict_proba(X)
        predictions_from_proba = np.argmax(probabilities, axis=1)
        
        np.testing.assert_array_equal(predictions, predictions_from_proba)


class TestFeatureImportance:
    """Tests for feature importance extraction."""
    
    def test_get_feature_importance_success(self, trained_model):
        """Test successful extraction of feature coefficients."""
        trainer, X, _, _ = trained_model
        
        coefficients = trainer.get_feature_importance()
        
        assert coefficients is not None
        assert len(coefficients) == X.shape[1]
        assert coefficients.dtype == np.float64
    
    def test_get_feature_importance_without_training(self):
        """Test that getting feature importance without training raises ValueError."""
        trainer = ModelTrainer()
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            trainer.get_feature_importance()
    
    def test_feature_importance_interpretation(self, trained_model, sample_text_data):
        """Test that feature importance has reasonable values."""
        trainer, _, _, vectorizer = trained_model
        
        coefficients = trainer.get_feature_importance()
        feature_names = vectorizer.get_feature_names_out()
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, coefficients))
        
        # Spam-related words should have positive coefficients
        spam_words = ['free', 'win', 'urgent', 'call']
        spam_coeffs = [feature_importance.get(word, 0) for word in spam_words]
        
        # At least some spam words should have positive weights
        assert any(coeff > 0 for coeff in spam_coeffs if coeff != 0)


class TestModelPersistence:
    """Tests for model saving and loading functionality."""
    
    def test_save_model_success(self, trained_model, temp_model_dir):
        """Test successful model saving."""
        trainer, _, _, _ = trained_model
        model_path = temp_model_dir / "test_model.pkl"
        
        trainer.save_model(model_path)
        
        assert model_path.exists()
        assert model_path.stat().st_size > 0
    
    def test_save_model_creates_directory(self, trained_model, temp_model_dir):
        """Test that save_model creates parent directories if needed."""
        trainer, _, _, _ = trained_model
        model_path = temp_model_dir / "nested" / "dir" / "model.pkl"
        
        trainer.save_model(model_path)
        
        assert model_path.exists()
        assert model_path.parent.exists()
    
    def test_save_model_without_training(self, temp_model_dir):
        """Test that saving untrained model raises ValueError."""
        trainer = ModelTrainer()
        model_path = temp_model_dir / "model.pkl"
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            trainer.save_model(model_path)
    
    def test_load_model_success(self, trained_model, temp_model_dir):
        """Test successful model loading."""
        trainer, X, _, _ = trained_model
        model_path = temp_model_dir / "test_model.pkl"
        
        # Save model
        trainer.save_model(model_path)
        
        # Create new trainer and load model
        new_trainer = ModelTrainer()
        result = new_trainer.load_model(model_path)
        
        # Check method chaining
        assert result is new_trainer
        # Check model is loaded
        assert new_trainer.is_trained()
        # Check predictions match
        original_predictions = trainer.predict(X)
        loaded_predictions = new_trainer.predict(X)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_load_nonexistent_model(self, temp_model_dir):
        """Test that loading nonexistent model raises FileNotFoundError."""
        trainer = ModelTrainer()
        model_path = temp_model_dir / "nonexistent_model.pkl"
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            trainer.load_model(model_path)
    
    def test_save_and_load_preserves_predictions(self, trained_model, temp_model_dir):
        """Test that saved and loaded model produces identical predictions."""
        trainer, X, _, _ = trained_model
        model_path = temp_model_dir / "model.pkl"
        
        # Get predictions from original model
        original_predictions = trainer.predict(X)
        original_probabilities = trainer.predict_proba(X)
        
        # Save and load model
        trainer.save_model(model_path)
        new_trainer = ModelTrainer()
        new_trainer.load_model(model_path)
        
        # Get predictions from loaded model
        loaded_predictions = new_trainer.predict(X)
        loaded_probabilities = new_trainer.predict_proba(X)
        
        # Compare predictions
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
        np.testing.assert_array_almost_equal(original_probabilities, loaded_probabilities)


class TestModelPerformance:
    """Integration tests for model performance on spam detection."""
    
    def test_model_learns_from_data(self, vectorized_data):
        """Test that model achieves reasonable accuracy on training data."""
        X, y, _ = vectorized_data
        trainer = ModelTrainer(max_iter=100, random_state=42)
        trainer.train(X, y)
        
        predictions = trainer.predict(X)
        accuracy = np.mean(predictions == y)
        
        # Model should achieve at least 70% accuracy on simple training data
        assert accuracy >= 0.7
    
    def test_model_predicts_both_classes(self, trained_model):
        """Test that model can predict both spam and ham."""
        trainer, X, y, _ = trained_model
        
        predictions = trainer.predict(X)
        unique_predictions = set(predictions)
        
        # Model should predict both classes (0 and 1)
        assert 0 in unique_predictions
        assert 1 in unique_predictions
    
    def test_consistent_predictions(self, trained_model):
        """Test that model produces consistent predictions on same data."""
        trainer, X, _, _ = trained_model
        
        predictions1 = trainer.predict(X)
        predictions2 = trainer.predict(X)
        
        np.testing.assert_array_equal(predictions1, predictions2)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_sample_prediction(self, trained_model):
        """Test prediction on a single sample."""
        trainer, X, _, _ = trained_model
        
        single_sample = X[0:1]
        prediction = trainer.predict(single_sample)
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
    
    def test_large_max_iter(self, vectorized_data):
        """Test training with very large max_iter."""
        X, y, _ = vectorized_data
        trainer = ModelTrainer(max_iter=10000, random_state=42)
        
        # Should complete without errors
        trainer.train(X, y)
        assert trainer.is_trained()
    
    def test_different_solvers(self, vectorized_data):
        """Test that different solvers work correctly."""
        X, y, _ = vectorized_data
        solvers = ['lbfgs', 'liblinear', 'saga']
        
        for solver in solvers:
            trainer = ModelTrainer(max_iter=100, random_state=42, solver=solver)
            trainer.train(X, y)
            
            assert trainer.is_trained()
            predictions = trainer.predict(X)
            assert len(predictions) == len(y)


class TestModelRetraining:
    """Tests for retraining models."""
    
    def test_retrain_replaces_model(self, vectorized_data):
        """Test that retraining replaces the existing model."""
        X, y, _ = vectorized_data
        trainer = ModelTrainer(max_iter=100, random_state=42)
        
        # Train first time
        trainer.train(X, y)
        first_predictions = trainer.predict(X)
        
        # Train second time with different random state
        trainer.random_state = 99
        trainer.train(X, y)
        second_predictions = trainer.predict(X)
        
        # Both should work, though predictions may differ due to random state
        assert len(first_predictions) == len(second_predictions)
        assert trainer.is_trained()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
