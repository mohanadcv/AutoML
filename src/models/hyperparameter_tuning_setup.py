"""
Hyperparameter Tuning - Simple and Effective Model Optimization

This module provides hyperparameter tuning for models using RandomizedSearchCV.
It works with the Model Registry and Trainers to:
1. Define parameter grids for each model
2. Tune models using Randomized Search (faster than Grid Search)
3. Evaluate tuned models on validation set
4. Return tuned models with best parameters

"""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from dataclasses import dataclass

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)

from Config.config import Config

logger = logging.getLogger(__name__)


# Result Container
@dataclass
class TuningResult:
    """
    Container for hyperparameter tuning results.

    Stores:
    - Best model found
    - Best parameters
    - Best CV score
    - Validation metrics
    - Search time
    """
    model_name: str
    task_type: str  # 'classification' or 'regression'

    # Tuning info
    best_model: Any
    best_params: Dict
    best_cv_score: float
    tuning_time: float
    n_iterations: int

    # Validation metrics (after tuning)
    val_score: float  # Accuracy for clf, R² for reg

    # Additional validation metrics
    val_metrics: Optional[Dict] = None

    # Preprocessing
    scaler: Optional[StandardScaler] = None
    required_scaling: bool = False


# Hyperparameter Tuner
class HyperparameterTuner:
    """
    🔧 Hyperparameter Tuner

    Tunes model hyperparameters using RandomizedSearchCV.
    User provides:
    - Which model to tune
    - Training data (X_train, y_train)
    - Validation data (X_val, y_val)

    Tuner handles:
    - Parameter grid definition
    - Random search execution
    - Validation evaluation
    - Result collection
    """

    def __init__(self, model_registry, task_type: str, config: Config = None):
        """
        Initialize tuner.

        """
        self.registry = model_registry
        self.task_type = task_type
        self.config = config or Config()
        self.random_state = self.config.RANDOM_STATE

        logger.info(f"Hyperparameter Tuner initialized for {task_type}")

    def get_param_grid(self, model_name: str) -> Dict:
        """
        Get parameter grid for a model.

        Returns:
            Dictionary of parameter distributions for RandomizedSearchCV
        """
        # ============================================================
        # CLASSIFICATION PARAMETER GRIDS
        # ============================================================
        if self.task_type == 'classification':

            if model_name == 'Random Forest':
                return {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [10, 20, 30],
                    'min_samples_leaf': [4, 8, 12],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True]
                }

            elif model_name == 'Logistic Regression':
                return {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'saga'],
                    'l1_ratio': [0.0, 0.5, 1.0],
                    'max_iter': [200, 500]
                }

            elif model_name == 'Gradient Boosting':
                return {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [20, 30, 40],
                    'min_samples_leaf': [10, 15, 20],
                    'subsample': [0.6, 0.8],
                    'max_features': ['sqrt', 'log2']
                }

            elif model_name == 'Decision Tree':
                return {
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None]
                }

            elif model_name == 'SVM':
                return {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }

            elif model_name == 'K-Nearest Neighbors':
                return {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }

            elif model_name == 'XGBoost':
                return {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
                }

            elif model_name == 'LightGBM':
                return {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [15, 31, 63],
                    'max_depth': [3, 5, 7],
                    'min_child_samples': [20, 40, 60],
                    'subsample': [0.6, 0.8],
                    'colsample_bytree': [0.6, 0.8],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 0.5]
                }

            elif model_name == 'AdaBoost':
                return {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.5, 1.0, 1.5, 2.0]
                }

        # ============================================================
        # REGRESSION PARAMETER GRIDS
        # ============================================================
        elif self.task_type == 'regression':

            if model_name == 'Random Forest':
                return {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [None, 10, 20, 30, 40],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }

            elif model_name == 'Ridge':
                return {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                }

            elif model_name == 'Lasso':
                return {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                }

            elif model_name == 'Elastic Net':
                return {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }

            elif model_name == 'Gradient Boosting':
                return {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6],
                    'min_samples_split': [2, 5, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }

            elif model_name == 'Decision Tree':
                return {
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8]
                }

            elif model_name == 'XGBoost':
                return {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
                }

            elif model_name == 'LightGBM':
                return {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'num_leaves': [15, 31, 63, 127],
                    'max_depth': [-1, 5, 10, 15],
                    'subsample': [0.7, 0.8, 0.9, 1.0]
                }

            elif model_name == 'SVR':
                return {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto']
                }

        # Default: return empty grid (will use model defaults)
        logger.warning(f"No parameter grid defined for {model_name}. Using defaults.")
        return {}

    def tune_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_iter: int = 20,
        cv=Config.CV_FOLDS,
    ) -> TuningResult:
        """
        Tune a single model using RandomizedSearchCV.

        Returns:
            TuningResult with best model and metrics
        """
        logger.info(f"🔧 Tuning {model_name}...")
        start_time = time.time()

        # Get base model and parameter grid
        base_model = self.registry.get_model(model_name, self.task_type)
        param_grid = self.get_param_grid(model_name)

        if not param_grid:
            logger.warning(f"No parameters to tune for {model_name}. Returning base model.")
            # Just train and evaluate base model
            base_model.fit(X_train, y_train)

            if self.task_type == 'classification':
                val_score = accuracy_score(y_val, base_model.predict(X_val))
            else:
                val_score = r2_score(y_val, base_model.predict(X_val))

            return TuningResult(
                model_name=model_name,
                task_type=self.task_type,
                best_model=base_model,
                best_params={},
                best_cv_score=val_score,
                tuning_time=time.time() - start_time,
                n_iterations=0,
                val_score=val_score
            )


        # Setup RandomizedSearchCV
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=Config.CV_FOLDS,
            scoring=scoring,
            n_jobs=self.config.N_JOBS,
            random_state=self.random_state,
            verbose=0
        )

        # Fit
        random_search.fit(X_train, y_train)

        # Get best model
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_cv_score = random_search.best_score_

        tuning_time = time.time() - start_time

        # Evaluate on validation set
        y_val_pred = best_model.predict(X_val)

        if self.task_type == 'classification':
            val_score = accuracy_score(y_val, y_val_pred)

            # Additional metrics
            avg = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'
            val_metrics = {
                'accuracy': val_score,
                'precision': precision_score(y_val, y_val_pred, average=avg, zero_division=0),
                'recall': recall_score(y_val, y_val_pred, average=avg, zero_division=0),
                'f1': f1_score(y_val, y_val_pred, average=avg, zero_division=0)
            }
        else:
            val_score = r2_score(y_val, y_val_pred)

            # Additional metrics
            val_metrics = {
                'r2': val_score,
                'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'mae': mean_absolute_error(y_val, y_val_pred)
            }

        logger.info(
            f"✅ {model_name} tuned in {tuning_time:.2f}s | "
            f"Best CV: {best_cv_score:.3f} | Val: {val_score:.3f}"
        )

        return TuningResult(
            model_name=model_name,
            task_type=self.task_type,
            best_model=best_model,
            best_params=best_params,
            best_cv_score=best_cv_score,
            tuning_time=tuning_time,
            n_iterations=n_iter,
            val_score=val_score,
            val_metrics=val_metrics
        )

    def tune_multiple(
        self,
        model_names: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_iter: int = 20,
        cv=Config.CV_FOLDS,
        show_progress: bool = True
    ) -> Dict[str, TuningResult]:
        """
        Tune multiple models.

        Returns:
            Dictionary of model_name -> TuningResult
        """
        results = {}

        for i, model_name in enumerate(model_names, 1):
            if show_progress:
                print(f"\n[{i}/{len(model_names)}] Tuning {model_name}...")

            try:
                result = self.tune_model(
                    model_name,
                    X_train, y_train,
                    X_val, y_val,
                    n_iter=n_iter,
                    cv=Config.CV_FOLDS
                )
                results[model_name] = result

                if show_progress:
                    if result.val_score is not None:
                        print(f"  ✅ Val Score: {result.val_score:.3f} | Time: {result.tuning_time:.2f}s")
                    else:
                        print(f"  ✅ Best CV Score: {result.best_cv_score:.3f} | Time: {result.tuning_time:.2f}s")

                    print(f"  Best params: {result.best_params}")

            except Exception as e:
                logger.exception(f"Failed to tune {model_name}: {e}")
                raise

        return results

    def compare_results(self, results: Dict[str, TuningResult]) -> pd.DataFrame:
        """
        Create comparison table of tuning results.


        Returns:
            DataFrame with comparison
        """
        if not results:
            raise RuntimeError(
                "All models failed during tuning. Check logs for the first error."
            )

        rows = []

        for model_name, result in results.items():
            row = {
                'Model': model_name,
                'Best CV Score': result.best_cv_score,
                'Val Score': result.val_score,
                'Tuning Time (s)': result.tuning_time,
                'Iterations': result.n_iterations
            }

            # Add task-specific metrics
            if result.val_metrics:
                if self.task_type == 'classification':
                    row['Val Precision'] = result.val_metrics.get('precision', np.nan)
                    row['Val Recall'] = result.val_metrics.get('recall', np.nan)
                    row['Val F1'] = result.val_metrics.get('f1', np.nan)
                else:
                    row['Val RMSE'] = result.val_metrics.get('rmse', np.nan)
                    row['Val MAE'] = result.val_metrics.get('mae', np.nan)

            rows.append(row)

        df = pd.DataFrame(rows)

        sort_col = 'Val Score' if 'Val Score' in df.columns else 'Best CV Score'
        return df.sort_values(sort_col, ascending=False).reset_index(drop=True)