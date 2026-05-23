#!/usr/bin/env python3
"""
Simple Air Quality ML Pipeline with Inline MLflow Integration

This pipeline includes MLflow logging directly in the main workflow without
utility functions, making it easy for students to understand.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import DataProcessor, FeatureEngineer, ModelTrainer
from pipeline.evaluator import Evaluator
from utils.config import MODEL_TYPES

from utils.logger import get_logger, set_log_level, log_level_from_string, LogLevel
from utils.utils import format_time_elapsed

# TODO Import parameter grids for optimization (Workshop 3)
from utils.config import DEFAULT_PARAM_GRIDS

# TODO Import MLflow (Workshop 4)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def run_pipeline(args):
    """
    Run the complete air quality prediction pipeline with inline MLflow integration.
    """
    start_time = time.time()
    logger = get_logger()

    # TODO Add MLflow setup and run start (Workshop 4)
    if args.mlflow:
        # Configuration MLflow simple
        mlflow.set_experiment("Air_Quality_Experiment")
        # Create descriptive run name
        run_name = f"{args.model}_{args.method}_{time.strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        
        # Set tags for Dataset and Model columns in MLflow UI
            # Dataset tags (for Dataset column)
        mlflow.set_tag("dataset_name", "Air_Quality")
        mlflow.set_tag("dataset_samples", 0)  # will update after data loading
        mlflow.set_tag("dataset_features", 0)  # will update after data loading
            # Model tags (for Model column)
        mlflow.set_tag("model_type", args.model)

            # Pipeline tags
        mlflow.set_tag("pipeline", "Air_Quality")
        mlflow.set_tag("feature_selection_method", args.method)
        mlflow.set_tag("optimization_enabled", args.optimize)
        # Log pipeline configuration parameters
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("feature_selection_method", args.method)
        mlflow.log_param("n_features", args.n_features)
        mlflow.log_param("optimization_enabled", args.optimize)
    try:
        # Pipeline header with configuration
        logger.header("AIR QUALITY ML PIPELINE")
        with logger.indent():
            logger.info(f"Model: {args.model}")
            logger.info(f"Features: {args.n_features}")
            logger.info(f"Selection method: {args.method}")
            logger.info(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
            logger.info(f"MLflow tracking: {'Enabled' if args.mlflow else 'Disabled'}")
            if args.mlflow:
                logger.info("Final model will be retrained on all data and registered in MLflow")

        # Initialize components
        processor = DataProcessor()
        engineer = FeatureEngineer()
        trainer = ModelTrainer()

        # Step 1: Data Loading and Preprocessing
        logger.step("Data Loading and Preprocessing", 1)
        with logger.timer("Data loading and preprocessing"):
            train_data, test_data = processor.load_and_preprocess()

        # TODO Add MLflow dataset logging (Workshop 4)
        if args.mlflow:
            # Log dataset inline (no separate function)
            train_csv_path = Path("data/train.csv")
            mlflow.log_artifact(str(train_csv_path), artifact_path="dataset")
            test_csv_path = Path("data/test.csv")
            mlflow.log_artifact(str(test_csv_path), artifact_path="dataset")

            # Log dataset metrics
            mlflow.log_param("train_samples", train_data.shape[0])
            mlflow.log_param("train_features", train_data.shape[1])
            mlflow.log_param("test_samples", test_data.shape[0])
            mlflow.log_param("test_features", test_data.shape[1])
        # Step 2: Feature Engineering
        logger.step("Feature Engineering", 2)
        with logger.timer("Feature engineering"):
            train_features, test_features = engineer.extract_all_features(train_data, test_data)

        with logger.indent():
            logger.data_info(f"Original features: {train_data.shape[1]}")
            logger.feature_info(f"Features after engineering: {train_features.shape[1]}")

        # Step 3: Feature Selection
        logger.step("Feature Selection", 3)
        with logger.timer("Feature selection"):
            selected_features = engineer.select_best_features(
                train_features, 
                method=args.method, 
                n_features=args.n_features
            )

        logger.feature_info(f"Selected {len(selected_features)} features")

        # TODO Add MLflow feature selection logging (Workshop 4)
        if args.mlflow:
            # Note: engineer.select_best_features() already logs MLflow parameters
            # We just log additional pipeline-specific info with different parameter names
            mlflow.log_param("n_selected_features", len(selected_features))
            mlflow.log_param("selected_features", ", ".join(selected_features))
        # Step 4: Cross-Validation Evaluation
        logger.step("Cross-Validation Evaluation", 4)

        # Initialize evaluator
        evaluator = Evaluator()

        # TODO Create model for cross-validation
        model = trainer.create_model(args.model)
          
        # TODO Prepare data X, y, and groups for cross-validation
        X = train_features[selected_features]
        y = train_data['pm2_5']
        groups = train_data['city']

        if not args.optimize:
            with logger.timer("Cross-validation"):
                # TODO Standard cross-validation using Evaluator
                cv_results = evaluator.cross_validate_model(
                model=model,
                X=X,
                y=y,
                groups=groups
               )
        # TODO Add hyperparameter optimization logic (Workshop 3)
        else :
            # Get parameter grid for the model
            param_grid = DEFAULT_PARAM_GRIDS.get(args.model, {})
            # If no grid is defined, use default parameters
            if not param_grid:
                logger.warning(f"No parameter grid defined for {args.model}, using default parameters")
                with logger.timer("Cross-validation"):
                    cv_results = evaluator.cross_validate_model(
                        model=model,
                        X=X,
                        y=y,
                        groups=groups
                    )
            else :
            # If grid is defined, perform optimization  
                logger.info(f"Performing hyperparameter optimization")
                logger.info(f"Parameter grid: {param_grid}")      
                
                with logger.timer("Hyperparameter optimization"):
                    # Perform hyperparameter optimization
                    best_model, best_params, best_score = evaluator.hyperparameter_optimization_cv(
                        model=model,
                        X=X,
                        y=y,
                        groups=groups,
                        param_grid=param_grid
                    )
                    if hasattr(best_model, 'best_params_'):
                        logger.info(f"Best parameters: {best_model.best_params_}")
                    else:
                        # For direct model (not GridSearchCV)
                        logger.info(f"Best parameters: {best_model.get_params()}")

                    # Add MLflow hyperparameter optimization logging (Workshop 4)

                    if args.mlflow:
                        mlflow.log_params(best_params)
                        mlflow.log_metric("best_cv_rmse", best_score)
                    # Use optimized model for final evaluation
                    model = best_model

                    # Quick evaluation to get full cv_results format
                    with logger.timer("Final cross-validation with optimized model"):
                        cv_results = evaluator.cross_validate_model(
                            model=model,
                            X=X,
                            y=y,
                            groups=groups
                        )
                    
                    logger.info(f"Optimized RMSE: {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f}")
                    logger.info(f"Optimized R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")


        # Extract results for compatibility
        mean_rmse = cv_results['rmse_mean']
        std_rmse = cv_results['rmse_std']
        mean_r2 = cv_results['r2_mean']
        std_r2 = cv_results['r2_std']

        # TODO Add MLflow model and results logging (Workshop 4)
        if args.mlflow:
            # Log cross-validation results
            mlflow.log_metric("cv_rmse_mean", mean_rmse)
            mlflow.log_metric("cv_rmse_std", std_rmse)
            mlflow.log_metric("cv_r2_mean", mean_r2)
            mlflow.log_metric("cv_r2_std", std_r2)
            # Log the trained model (for Model column)
                    # Step 1: Prepare clean data for MLflow (avoid warnings)
                    # Remove rows with missing values and convert to float64
            X_clean = X.dropna().astype(np.float64)
            y_clean = y.loc[X_clean.index].astype(np.float64)

                    # Step 2: Create MLflow model signature using clean data
                    # The signature describes input/output format for the model
            signature = infer_signature(X_clean, model.predict(X_clean))
                    # Step 3: Create descriptive model name (appears in Model column)
            model_name_mlflow = f"{args.model}_cv_model"
                    # Step 4: Prepare input example for MLflow documentation
                    # This shows users what kind of data the model expects
            input_example = X_clean.head(3)
                    # Log model to MLflow
            mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name_mlflow,
            signature=signature,
            input_example=input_example
            )
                    # Register model in MLflow Model Registry
                            
        # TODO Step 5: Save Model (Workshop 4)
        if args.mlflow:
            # Retrain final model on all training data (best practice)
            model.fit(X, y)

                # Log the final model to MLflow
                # Prepare clean data for MLflow model signature
            X_clean_final = X.dropna().astype(np.float64)
            y_clean_final = y.loc[X_clean_final.index].astype(np.float64)
                # Create model name for MLflow
            final_model_name = f"{args.model}_final_model"
                # Log final model to MLflow
            signature_final = infer_signature(X_clean_final, model.predict(X_clean_final))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="final_model",
                registered_model_name=final_model_name,
                signature=signature_final,
                input_example=X_clean_final.head(3)
                )
                # Register model in MLflow Model Registry
                
                
                
        # Prepare results for summary
        model_name = args.model
        cv_results_dict = {
            'rmse_mean': mean_rmse,
            'rmse_std': std_rmse,
            'r2_mean': mean_r2,
            'r2_std': std_r2
        }

        results = {
            'model_type': args.model,
            'cv_results': cv_results_dict,
            'selected_features': selected_features
        }

        # Step 6: Results Summary
        logger.step("Results Summary", 6)

        end_time = time.time()
        execution_time = format_time_elapsed(start_time, end_time)
        execution_time_sec = end_time - start_time

        summary = {
            'Model': model_name,
            'RMSE': f"{cv_results_dict.get('rmse_mean', 'N/A'):.3f}",
            'R²': f"{cv_results_dict.get('r2_mean', 'N/A'):.3f}",
            'Features': len(selected_features),
            'Selection Method': args.method,
            'Optimized': args.optimize,
            'Execution Time': execution_time
        }

        logger.results_summary(summary)

        # TODO Add MLflow final results logging (Workshop 4)
        if args.mlflow:
            mlflow.log_param("n_features_selected", len(selected_features))
            mlflow.log_param("feature_selection_method", args.method)
            mlflow.log_param("optimization_enabled", args.optimize)
            mlflow.log_metric("pipeline_execution_time", execution_time_sec)

        logger.pipeline_complete(end_time - start_time)

        return {
            'results': results,
            'summary': summary,
            'selected_features': selected_features,
            'execution_time': execution_time
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    
    finally:
        # TODO End MLflow run (Workshop 4)
        pass
        if args.mlflow:
            mlflow.end_run()

    # TODO End MLflow run (Workshop 4)
    if args.mlflow:
        mlflow.end_run()

def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Run Air Quality ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', type=str, default='linear',
        choices=MODEL_TYPES,
        help='Model type to train'
    )
    
    parser.add_argument(
        '--n-features', type=int, default=15,
        help='Number of features to select'
    )
    
    parser.add_argument(
        '--method', type=str, default='selectkbest',
        choices=['selectkbest', 'rfe'],
        help='Feature selection method'
    )
    
    parser.add_argument(
        '--optimize', action='store_true',
        help='Enable hyperparameter optimization using GridSearchCV'
    )
    
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare multiple models instead of training single model'
    )
     
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output (deprecated, use --log-level verbose)'
    )
    
    parser.add_argument(
        '--log-level', type=str, default='normal',
        choices=['silent', 'normal', 'verbose'],
        help='Logging level: silent (no output), normal (main steps), verbose (all details)'
    )

    # TODO Add MLflow tracking argument --mlflow (Workshop 4)
    parser.add_argument(
        '--mlflow', action='store_true',
        help='Enable MLflow tracking and model logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Configure logging level
        if args.verbose:
            # Support legacy --verbose flag
            log_level = LogLevel.VERBOSE
        else:
            log_level = log_level_from_string(args.log_level)
        
        set_log_level(log_level)
        
        # Run pipeline
        run_pipeline(args)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
