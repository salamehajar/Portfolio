"""
NLP Spam Detection Pipeline Package

This package contains the core NLP pipeline components for production-ready spam detection:
- DataProcessor: Dataset loading and preprocessing
- TextPreprocessor: Tokenization, normalization, stop word handling, vocabulary sizing
- ModelTrainer: Model training and evaluation
- Evaluator: Metrics computation and visualization

Usage:
    from pipeline import DataProcessor, TextPreprocessor, ModelTrainer, Evaluator
"""

__version__ = "1.0.0"

# Import main classes for easy access
# Note: Uncomment these as each module is implemented by the team
from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .evaluator import Evaluator

__all__ = [
    'DataProcessor',
    'FeatureEngineer', 
    'ModelTrainer',
    'Evaluator'
]
