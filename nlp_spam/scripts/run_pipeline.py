
"""
NLP Spam Detection Pipeline with Inline MLflow Integration

This pipeline includes MLflow logging directly in the main workflow,
handling Data Loading, Feature Engineering, Model Training, and Evaluation.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Pipeline Components
from pipeline.data_processor import DataProcessor
from pipeline.feature_engineer import FeatureEngineer
from pipeline.evaluator import Evaluator

# Config & Utils
from utils.config import (
    DEFAULT_PARAM_GRIDS, RANDOM_STATE, NB_FEATURES, 
    LOWERCASE, stop_words
)
from utils.logger import get_logger, set_log_level, log_level_from_string, LogLevel
from utils.utils import format_time_elapsed

# MLflow & Sklearn
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Optional Tree Models
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    HAS_TREE_MODELS = True
except ImportError:
    HAS_TREE_MODELS = False


def get_model_instance(model_name, random_state):
    """Factory function to create the classifier instance."""
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=random_state, max_iter=1000)
    
    
    if model_name == "xgboost":
        if not HAS_TREE_MODELS: raise ValueError("XGBoost not installed.")
        return XGBClassifier(random_state=random_state, eval_metric='logloss')
    
    if model_name == "lightgbm":
        if not HAS_TREE_MODELS: raise ValueError("LightGBM not installed.")
        return LGBMClassifier(random_state=random_state, force_col_wise=True)
        
    raise ValueError(f"Model '{model_name}' not supported.")


def run_pipeline(args):
    """
    Run the complete NLP spam prediction pipeline with inline MLflow integration.
    """
    start_time = time.time()
    logger = get_logger()

    # -------------------------------------------------------------------------
    # 1. MLflow Setup
    # -------------------------------------------------------------------------
    if args.mlflow:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        # Configure MLflow experiment
        mlflow.set_experiment("NLP_Spam_Detection")
        
        # Create descriptive run name
        run_name = f"{args.model}_{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        
        # Set Tags for filtering in UI
        mlflow.set_tag("pipeline", "NLP_Spam")
        mlflow.set_tag("dataset_scenario", args.dataset)
        mlflow.set_tag("model_type", args.model)
        mlflow.set_tag("optimization_enabled", args.optimize)
        
        # Log configuration parameters
        mlflow.log_params(vars(args))

    try:
        # Pipeline header
        logger.header("NLP SPAM DETECTION PIPELINE")
        with logger.indent():
            logger.info(f"Model: {args.model}")
            logger.info(f"Dataset Scenario: {args.dataset}")
            logger.info(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
            logger.info(f"MLflow tracking: {'Enabled' if args.mlflow else 'Disabled'}")

        # Initialize DataProcessor
        processor = DataProcessor()

        # ---------------------------------------------------------------------
        # Step 1: Data Loading & Split
        # ---------------------------------------------------------------------
        logger.step("Data Loading and Preparation", 1)
        with logger.timer("Data loading"):
            # Load raw data (SMS and Email)
            sms_df, email_df = processor.load_data()
            
            # Clean duplicates
            sms_df = processor.drop_duplicates(sms_df)
            email_df = processor.drop_duplicates(email_df)

        # Extract messages and labels
        sms_msg, sms_lbl = processor.extract_msgs_labels(sms_df)
        email_msg, email_lbl = processor.extract_msgs_labels(email_df)

        # Apply splitting logic based on dataset argument
        logger.info(f"Applying splitting strategy: {args.dataset}")
        
        if args.dataset == "sms":
            X_train, X_test, y_train, y_test = processor.train_eval_same_data(sms_msg, sms_lbl)
        elif args.dataset == "email":
            X_train, X_test, y_train, y_test = processor.train_eval_same_data(email_msg, email_lbl)
        elif args.dataset == "cross":
            # Train on SMS, Test on Email
            X_train, y_train, X_test, y_test = processor.train_data1_eval_data2(
                sms_msg, sms_lbl, email_msg, email_lbl
            )
        elif args.dataset == "combined":
            X_train, X_test, y_train, y_test = processor.train_eval_combined(sms_df, email_df)
        else:
            raise ValueError(f"Unknown dataset scenario: {args.dataset}")

        # Balance the training data
        with logger.timer("Balancing training data"):
            X_train, y_train = processor.balance(X_train, y_train)

        # Log Data Statistics
        logger.data_info(f"Training Samples: {len(X_train)}")
        logger.data_info(f"Test Samples:     {len(X_test)}")

        if args.mlflow:
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

        # ---------------------------------------------------------------------
        # Step 2: Pipeline Construction (Feature Eng + Model)
        # ---------------------------------------------------------------------
        logger.step("Pipeline Construction", 2)
        
        # Initialize Feature Engineer
        fe = FeatureEngineer(
            max_features=NB_FEATURES,
            token_pattern=args.token_pattern,
            lowercase=LOWERCASE,
            stop_words=stop_words,
            #log_to_mlflow=False  # We log manually in this script
        )
        
        # Initialize Classifier
        clf = get_model_instance(args.model, RANDOM_STATE)
        
        # Create Scikit-Learn Pipeline
        # This ensures vectorization happens correctly inside CV folds
        model_pipeline = Pipeline([
            ('vectorizer', fe),
            ('classifier', clf)
        ])

        # ---------------------------------------------------------------------
        # Step 3: Cross-Validation / Optimization
        # ---------------------------------------------------------------------
        logger.step("Model Training & Optimization", 3)
        
        evaluator = Evaluator()
        
        # Prepare param grid if optimization is enabled
        param_grid = DEFAULT_PARAM_GRIDS.get(args.model, {})
        
        if args.optimize and param_grid:
            logger.info("Performing Hyperparameter Optimization (GridSearchCV)...")
            
            with logger.timer("Hyperparameter optimization"):
                best_model, best_params, best_score = evaluator.hyperparameter_optimization_cv(
                    model=model_pipeline,
                    param_grid=param_grid,
                    X=X_train,
                    y=y_train
                )
            
            logger.success(f"Best CV Accuracy: {best_score:.4f}")
            logger.info(f"Best Parameters: {best_params}")
            
            # Update pipeline to the best found model
            model_pipeline = best_model
            
            if args.mlflow:
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_accuracy", best_score)
        
        else:
            # Standard Cross-Validation
            logger.info("Running Standard Cross-Validation...")
            with logger.timer("Cross-validation"):
                cv_results = evaluator.cross_validate_model(
                    model=model_pipeline,
                    X=X_train,
                    y=y_train
                )
            
            # Fit final model on full training data
            model_pipeline.fit(X_train, y_train)
            
            if args.mlflow:
                mlflow.log_metric("cv_accuracy_mean", cv_results['accuracy_mean'])
                mlflow.log_metric("cv_accuracy_std", cv_results['accuracy_std'])

        # ---------------------------------------------------------------------
        # Step 4: Final Evaluation
        # ---------------------------------------------------------------------
        logger.step("Final Evaluation on Test Set", 4)
        
        # Predict on Test Data
        y_pred = model_pipeline.predict(X_test)
        
        # Calculate Metrics
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        
        # Log metrics to console
        logger.results_summary({
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}"
        })

        if args.mlflow:
            mlflow.log_metrics(metrics)

        # ---------------------------------------------------------------------
        # Step 5: Model Artifact Logging
        # ---------------------------------------------------------------------
        if args.mlflow:
            logger.step("Logging Artifacts to MLflow", 5)
            
            # FIX: Convert Series to DataFrame for MLflow compatibility
            # MLflow expects structured input (DataFrame/Array), not a raw Series
            input_example = X_test.head(5).to_frame(name="message")
            
            # Generate predictions for signature (using the dataframe input)
            # Note: The pipeline expects the raw text column, so we might need to 
            # pass the series or ensure the pipeline handles the dataframe.
            # Ideally, we pass the raw text series to predict, but log the dataframe.
            prediction_example = model_pipeline.predict(X_test.head(5))
            
            # Infer signature
            signature = infer_signature(input_example, prediction_example)
            
            # Log the full pipeline
            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=f"NLP_{args.model}_{args.dataset}"
            )
            logger.success("Model logged to MLflow")

        # ---------------------------------------------------------------------
        # Completion
        # ---------------------------------------------------------------------
        end_time = time.time()
        execution_time = end_time - start_time
        logger.pipeline_complete(execution_time)
        
        if args.mlflow:
            mlflow.log_metric("execution_time_seconds", execution_time)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise e
    
    finally:
        if args.mlflow:
            mlflow.end_run()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run NLP Spam Detection Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset', type=str, default='combined',
        choices=['sms', 'email', 'cross', 'combined'],
        help='Dataset scenario to run'
    )
    
    parser.add_argument(
        '--model', type=str, default='logistic_regression',
        choices=['logistic_regression', 'xgboost', 'lightgbm'],
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--token-pattern', type=str, default=r"([A-Za-z]{4,}|!+)",
        help='Regex pattern for tokenization'
    )
    
    parser.add_argument(
        '--optimize', action='store_true',
        help='Enable hyperparameter optimization (GridSearch)'
    )
    
    parser.add_argument(
        '--log-level', type=str, default='normal',
        choices=['silent', 'normal', 'verbose'],
        help='Logging verbosity level'
    )
    
    parser.add_argument(
        '--mlflow', action='store_true',
        help='Enable MLflow tracking'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        
        # Configure logging
        if args.log_level == 'verbose':
            log_level = LogLevel.VERBOSE
        elif args.log_level == 'silent':
            log_level = LogLevel.SILENT
        else:
            log_level = LogLevel.NORMAL
            
        set_log_level(log_level)
        
        run_pipeline(args)
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
