"""
NLP Spam Detection Package

Core components:
- DataProcessor: Load and preprocess text data
- FeatureEngineer: Extract and select features
- ModelTrainer: Train and compare models
- Evaluator: Evaluate models

Utilities:
- config: Constants and settings
- utils: General helper functions
- evaluation_utils: Detailed evaluation with plots
"""

__version__ = "1.0.0"
__author__ = "NLP ML Workshop - NLP Spam Detection"

# Uncomment these imports once all pipeline modules are implemented

from pipeline import DataProcessor, FeatureEngineer, ModelTrainer, Evaluator

__all__ = [
    'DataProcessor',
    'FeatureEngineer', 
    'ModelTrainer',
    'Evaluator'
]
