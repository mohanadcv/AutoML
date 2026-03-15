"""
Main Pipeline - AutoML System Orchestrator

This is the BRAIN that connects everything together.
Supports both CLI (for developers) and programmatic use.

Complete Workflow:
1. Load Data
2. Validate Data
3. Select Target (user input)
4. Detect Task Type (with override option)
5. Preprocess Data
6. Split Data (train/val/test)
7. Train Models (user selects which models)
8. Hyperparameter Tuning (optional)
9. Evaluate on Test Set
10. Save Best Model

"""
# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import argparse
import logging
import joblib
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# Import all modules
from Config.config import Config
from src.data_processing.loader import DataLoader
from src.data_processing.validator import DataValidator
from src.data_processing.task_detector import TaskDetector
from src.data_processing.preprocessing_pipeline import PreprocessingPipeline
from src.data_processing.splitting import DataSplitter
from src.models.registry import ModelRegistry
from src.models.trainers.classification import ClassificationTrainer
from src.models.trainers.regression import RegressionTrainer
from src.models.hyperparameter_tuning_setup import HyperparameterTuner
from src.utils.mlflow_setup import MlflowSetup
from src.visualizations.training import plot_classification_results, plot_regression_results
from src.visualizations.tuning import plot_tuned_classification_results, plot_tuned_regression_results
from src.models.final_evaluation import evaluate_test_classification, evaluate_test_regression, \
    plot_test_classification, plot_test_regression

# Setup logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """
    🤖 Complete AutoML Pipeline

    Orchestrates the entire ML workflows from data loading to model deployment.

    Features:
    - Automatic task detection
    - Smart preprocessing
    - Model training (user selects models)
    - Hyperparameter tuning (optional)
    - Test set evaluation
    - Model persistence

    """

    def __init__(self, config: Config = None):
        """Initialize pipeline with configuration."""
        self.config = config or Config()
        self.config.setup_directories()

        # Initialize components
        self.loader = DataLoader(self.config)
        self.validator = DataValidator(self.config)
        self.task_detector = TaskDetector(self.config)
        self.preprocessor = PreprocessingPipeline()
        self.splitter = DataSplitter(self.config)
        self.registry = ModelRegistry(self.config)

        # Setup MLflow
        mlflow_setup = MlflowSetup(self.config)
        mlflow_setup.run_full_setup()

        # State variables
        self.raw_data = None
        self.target_column = None
        self.task_type = None
        self.task_confidence = None
        self.X = None
        self.y = None
        self.X_processed = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.training_results = None
        self.tuning_results = None
        self.test_results = None
        self.best_model = None
        self.best_model_name = None

        logger.info("✅ AutoML Pipeline initialized")

    # ================================================================
    # STEP 1: LOAD DATA
    # ================================================================

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            file_path: Path to data file

        Returns:
            Loaded DataFrame
        """
        logger.info(f"📂 Loading data from: {file_path}")
        self.raw_data = self.loader.load(file_path)
        logger.info(f"✅ Data loaded: {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns")
        return self.raw_data

    # ================================================================
    # STEP 2: VALIDATE DATA
    # ================================================================

    def validate_data(self) -> Dict:
        """
        Validate loaded data.

        Returns:
            Dictionary with validation results
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("🔍 Validating data...")

        # Validate the dataframe
        is_valid, message = self.validator.validate_dataframe(self.raw_data)

        # Also run comprehensive data quality check
        if is_valid:
            quality_report = self.validator.check_data_quality(self.raw_data)
            logger.info("✅ Data validation passed")
            return {
                'valid': True,
                'message': message,
                'quality_report': quality_report,
                'issues': []  # Empty list for consistency
            }
        else:
            logger.error(f"❌ Data validation failed: {message}")
            return {
                'valid': False,
                'message': message,
                'quality_report': None,
                'issues': [message]  # Single issue in list
            }

    # ================================================================
    # STEP 3: SELECT TARGET
    # ================================================================

    def select_target(self, target_column: str):
        """
        Select target column for prediction.

        Args:
            target_column: Name of target column
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if target_column not in self.raw_data.columns:
            raise ValueError(f"Column '{target_column}' not found in data")

        self.target_column = target_column
        self.X = self.raw_data.drop(columns=[target_column])
        self.y = self.raw_data[target_column]

        logger.info(f"🎯 Target column set: {target_column}")
        logger.info(f"   Features: {self.X.shape[1]} columns")
        logger.info(f"   Samples: {len(self.y)}")

    # ================================================================
    # STEP 4: DETECT TASK TYPE
    # ================================================================

    def detect_task_type(self, override: Optional[str] = None) -> Tuple[str, float]:
        """
        Detect if task is classification or regression.

        Args:
            override: Optional manual override ('classification' or 'regression')

        Returns:
            Tuple of (task_type, confidence)
        """
        if self.y is None:
            raise ValueError("No target selected. Call select_target() first.")

        if override:
            if override not in ['classification', 'regression']:
                raise ValueError("Override must be 'classification' or 'regression'")
            self.task_type = override
            self.task_confidence = 1.0
            logger.info(f"🎯 Task type overridden: {override}")
        else:
            logger.info("🧠 Detecting task type...")
            self.task_type, self.task_confidence = self.task_detector.detect(self.y)
            logger.info(f"✅ Detected: {self.task_type} (confidence: {self.task_confidence:.2f})")

            if self.task_confidence < 0.8:
                logger.warning("⚠️  Low confidence! Consider manual override.")

        return self.task_type, self.task_confidence

    # ================================================================
    # STEP 5: PREPROCESS DATA
    # ================================================================

    def run_preprocessing(self, show_visualization: bool = False):
        """
        Run preprocessing pipeline.

        Args:
            show_visualization: Whether to show preprocessing visualizations
        """
        if self.X is None or self.y is None:
            raise ValueError("No data to preprocess. Call select_target() first.")

        logger.info("⚙️  Running preprocessing...")
        self.X_processed = self.preprocessor.fit_transform(self.X, self.y)
        logger.info(f"✅ Preprocessing complete: {self.X_processed.shape[1]} features")

        # Show what was done
        logger.info("   Applied transformations:")
        logger.info(f"   - Numeric features: {len(self.preprocessor.numeric_features_)}")
        logger.info(f"   - Categorical features: {len(self.preprocessor.categorical_features_)}")
        logger.info(f"   - Low-cardinality (one-hot): {len(self.preprocessor.low_card_categorical_)}")
        logger.info(f"   - High-cardinality (frequency encoded): {len(self.preprocessor.high_card_categorical_)}")

        # Show dropped ID columns if any
        if hasattr(self.preprocessor, 'id_columns_') and self.preprocessor.id_columns_:
            logger.info(f"   - Dropped ID columns: {len(self.preprocessor.id_columns_)}")
            logger.info(f"     IDs removed: {self.preprocessor.id_columns_}")

    # ================================================================
    # STEP 6: SPLIT DATA
    # ================================================================

    def split_data(self):
        """Split data into train/val/test sets."""
        if self.X_processed is None:
            raise ValueError("No processed data. Call run_preprocessing() first.")

        logger.info("✂️  Splitting data into train/val/test...")
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.splitter.split(self.X_processed, self.y)

        logger.info(f"✅ Data split complete:")
        logger.info(f"   Train: {len(self.X_train)} samples")
        logger.info(f"   Val:   {len(self.X_val)} samples")
        logger.info(f"   Test:  {len(self.X_test)} samples")

    # ================================================================
    # STEP 7: TRAIN MODELS
    # ================================================================

    def train_models(self,
                     model_names: List[str],
                     run_cv: bool = True,
                     show_visualization: bool = True):
        """
        Train selected models.

        Args:
            model_names: List of model names to train (or ['all'])
            run_cv: Run cross-validation
            show_visualization: Show training visualizations
        """
        if self.X_train is None:
            raise ValueError("No split data. Call split_data() first.")

        # Get all available models if 'all' specified
        if 'all' in model_names or model_names == ['all']:
            model_names = self.registry.get_models_for_task(self.task_type)
            logger.info(f"🚀 Training ALL {len(model_names)} models")
        else:
            logger.info(f"🚀 Training {len(model_names)} selected models")

        # Initialize appropriate trainer
        if self.task_type == 'classification':
            trainer = ClassificationTrainer(self.registry, self.config)
        else:
            trainer = RegressionTrainer(self.registry, self.config)

        # Train models
        self.training_results = trainer.train_multiple(
            model_names,
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            run_cv=run_cv,
            show_progress=True
        )

        # Compare results
        comparison = trainer.compare_results(self.training_results)
        print("\n" + "=" * 80)
        print("📊 TRAINING RESULTS")
        print("=" * 80)
        print(comparison.to_string(index=False))
        print("=" * 80 + "\n")

        # Show visualization
        if show_visualization:
            if self.task_type == 'classification':
                if comparison.empty:
                    logger.warning("⚠️ No models trained successfully, skipping visualization")
                else:
                    plot_classification_results(comparison)
            else:
                if comparison.empty:
                    logger.warning("⚠️ No models trained successfully, skipping visualization")
                else:
                    plot_regression_results(comparison)

        # Save best model info
        best_name, best_result = trainer.get_best_model(self.training_results)
        self.best_model_name = best_name
        self.best_model = best_result

        logger.info(f"🏆 Best model: {best_name}")

    # ================================================================
    # STEP 8: HYPERPARAMETER TUNING (OPTIONAL)
    # ================================================================

    def tune_models(self,
                    model_names: Optional[List[str]] = None,
                    n_iter: int = 20,
                    cv=Config.CV_FOLDS,
                    show_visualization: bool = True):
        """
        Tune hyperparameters for selected models.

        Args:
            model_names: List of models to tune (defaults to trained models)
            n_iter: Number of random search iterations
            cv: Number of CV folds
            show_visualization: Show tuning visualizations
        """
        if self.training_results is None:
            raise ValueError("No trained models. Call train_models() first.")

        # Default to all trained models
        if model_names is None:
            model_names = list(self.training_results.keys())

        logger.info(f"🔧 Tuning {len(model_names)} models...")

        # Initialize tuner
        tuner = HyperparameterTuner(self.registry, self.task_type, self.config)

        # Tune models
        self.tuning_results = tuner.tune_multiple(
            model_names,
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            n_iter=n_iter,
            cv=Config.CV_FOLDS,
            show_progress=True
        )

        # Compare results
        comparison = tuner.compare_results(self.tuning_results)
        print("\n" + "=" * 80)
        print("🔧 TUNING RESULTS")
        print("=" * 80)
        print(comparison.to_string(index=False))
        print("=" * 80 + "\n")

        # Show visualization
        if show_visualization:
            if self.task_type == 'classification':
                plot_tuned_classification_results(comparison)
            else:
                plot_tuned_regression_results(comparison)

        # Update best model with tuned version
        best_tuned_name = comparison.iloc[0]['Model']
        self.best_model_name = best_tuned_name
        self.best_model = self.tuning_results[best_tuned_name]

        logger.info(f"🏆 Best tuned model: {best_tuned_name}")

    # ================================================================
    # STEP 9: EVALUATE ON TEST SET
    # ================================================================

    def evaluate_test(self, show_visualization: bool = True):
        """
        Evaluate best model(s) on test set.

        Args:
            show_visualization: Show test evaluation visualizations
        """
        if self.X_test is None:
            raise ValueError("No test data. Call split_data() first.")

        # Use tuned results if available, otherwise use training results
        results_to_eval = self.tuning_results if self.tuning_results else self.training_results

        if results_to_eval is None:
            raise ValueError("No models trained. Call train_models() first.")

        logger.info("📊 Evaluating on test set...")

        # Evaluate based on task type
        if self.task_type == 'classification':
            self.test_results = evaluate_test_classification(
                results_to_eval,
                self.X_test,
                self.y_test
            )
        else:
            self.test_results = evaluate_test_regression(
                results_to_eval,
                self.X_test,
                self.y_test
            )

        print("\n" + "=" * 80)
        print("🎯 TEST SET RESULTS")
        print("=" * 80)
        print(self.test_results.to_string(index=False))
        print("=" * 80 + "\n")

        # Show visualization
        if show_visualization:
            if self.task_type == 'classification':
                plot_test_classification(self.test_results)
            else:
                plot_test_regression(self.test_results)

    # ================================================================
    # STEP 10: SAVE MODEL
    # ================================================================

    def save_best_model(self, output_path: Optional[str] = None) -> str:
        """
        Save best model to disk.
        """
        if self.best_model is None:
            raise ValueError("No best model found. Train models first.")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.MODEL_DIR / f"{self.best_model_name}_{timestamp}.pkl"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the actual model object - handle both TrainingResult and TuningResult
        if hasattr(self.best_model, 'model'):
            model_to_save = self.best_model.model
        elif hasattr(self.best_model, 'best_model'):
            model_to_save = self.best_model.best_model
        else:
            raise ValueError(f"Best model object doesn't have a model attribute: {type(self.best_model)}")

        # Get scaler
        scaler = getattr(self.best_model, 'scaler', None)

        # Get feature names - handle numpy arrays
        if hasattr(self.X_train, 'columns'):
            feature_names = list(self.X_train.columns)
        else:
            # For numpy arrays, create generic feature names
            feature_names = [f'feature_{i}' for i in range(self.X_train.shape[1])]

        # Save model and metadata
        model_package = {
            'model': model_to_save,  # Use the correct model object
            'model_name': self.best_model_name,
            'task_type': self.task_type,
            'preprocessor': self.preprocessor,
            'scaler': scaler,
            'feature_names': feature_names,
            'target_column': self.target_column,
            'metadata': {
                'train_date': datetime.now().isoformat(),
                'n_features': self.X_train.shape[1],
                'n_samples': len(self.X_train),
                'model_source': 'tuned' if hasattr(self.best_model, 'best_model') else 'trained'
            }
        }

        joblib.dump(model_package, output_path)
        logger.info(f"💾 Model saved to: {output_path}")

        return str(output_path)

    # ================================================================
    # CONVENIENCE METHOD: RUN FULL PIPELINE
    # ================================================================

    def run_full_pipeline(self,
                          file_path: str,
                          target_column: str,
                          models: List[str] = ['all'],
                          tune_models: bool = True,
                          task_type_override: Optional[str] = None):
        """
        Run complete pipeline end-to-end.

        Args:
            file_path: Path to data file
            target_column: Name of target column
            models: List of models to train
            tune_models: Tune hyperparameters
            task_type_override: Manual task type override
        """
        logger.info("🚀 Starting full AutoML pipeline...")

        # 1. Load and validate
        self.load_data(file_path)
        self.validate_data()

        # 2. Select target and detect task
        self.select_target(target_column)
        self.detect_task_type(override=task_type_override)

        # 3. Preprocess
        self.run_preprocessing()

        # 4. Split data
        self.split_data()

        # 5. Train models
        self.train_models(models)

        # 6. Tune (optional)
        if tune_models:
            self.tune_models()

        # 7. Evaluate on test
        self.evaluate_test()

        # 8. Save best model
        model_path = self.save_best_model()

        logger.info("🎉 Pipeline complete!")
        logger.info(f"   Best model: {self.best_model_name}")
        logger.info(f"   Saved to: {model_path}")

        return model_path


# ================================================================
# CLI INTERFACE
# ================================================================

def main():
    """CLI interface for developers."""
    parser = argparse.ArgumentParser(
        description='AutoML Pipeline - Train ML models automatically',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main_pipeline.py --data data.csv --target price --models all

  # Train specific models
  python main_pipeline.py --data data.csv --target survived --models "Random Forest,XGBoost"

  # Preprocessing only
  python main_pipeline.py --data data.csv --target price --preprocess-only

  # Tune specific model
  python main_pipeline.py --data data.csv --target price --tune-only --models "Random Forest"
        """
    )

    # Required arguments
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--target', required=True, help='Name of target column')

    # Model selection
    parser.add_argument('--models', default='all',
                        help='Comma-separated model names or "all" (default: all)')

    # Task type
    parser.add_argument('--task-type', choices=['classification', 'regression'],
                        help='Override auto task detection')

    # Pipeline stages
    parser.add_argument('--preprocess-only', action='store_true',
                        help='Run preprocessing only')
    parser.add_argument('--train-only', action='store_true',
                        help='Run training only (skip tuning)')
    parser.add_argument('--tune-only', action='store_true',
                        help='Run tuning only (requires trained models)')

    parser.add_argument('--no-tune', action='store_true',
                        help='Skip hyperparameter tuning')

    # Tuning parameters
    parser.add_argument('--n-iter', type=int, default=20,
                        help='Number of tuning iterations (default: 20)')

    # Output
    parser.add_argument('--output', help='Output path for saved model')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualizations')

    args = parser.parse_args()

    # Parse models
    if args.models.lower() == 'all':
        models = ['all']
    else:
        models = [m.strip() for m in args.models.split(',')]

    # Initialize pipeline
    pipeline = AutoMLPipeline()

    try:
        # Load and validate
        pipeline.load_data(args.data)
        pipeline.validate_data()
        pipeline.select_target(args.target)
        pipeline.detect_task_type(override=args.task_type)

        # Preprocessing
        pipeline.run_preprocessing()

        if args.preprocess_only:
            print("\n✅ Preprocessing complete!")
            return

        # Split data
        pipeline.split_data()

        # Training
        pipeline.train_models(models, show_visualization=not args.no_viz)

        if args.train_only:
            print("\n✅ Training complete!")
            return

        # Tuning
        if not args.no_tune and not args.tune_only:
            pipeline.tune_models(n_iter=args.n_iter, cv=args.cv,
                                 show_visualization=not args.no_viz)
        elif args.tune_only:
            pipeline.tune_models(models, n_iter=args.n_iter, cv=args.cv,
                                 show_visualization=not args.no_viz)
            return

        # Test evaluation
        pipeline.evaluate_test(show_visualization=not args.no_viz)

        # Save model
        model_path = pipeline.save_best_model(args.output)

        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"Best Model: {pipeline.best_model_name}")
        print(f"Task Type: {pipeline.task_type}")
        print(f"Saved to: {model_path}")
        print("=" * 80)

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()