"""
NLP Spam Utilities Package

This package contains utility functions and configurations for the NLP Spam Detection project:
- config: Configuration constants and settings
- utils: General utility functions
- evaluation_utils: Detailed evaluation functions with visualizations

Usage:
    from utils.config import *
    from utils.utils import print_step_header
    from utils.evaluation_utils import evaluate_model_detailed
"""

__version__ = "1.0.0"

# Import commonly used items
from .config import *
from .utils import *
from .logger import get_logger, set_log_level, print_step_header, print_results_summary

__all__ = [
    # Config constants
    #à faire
    # Utility functions
    'get_logger',
    'set_log_level',
    'print_step_header',
    'print_results_summary'
]
