"""
Regression Models - Helper Module for Training Regression Models

This module provides a clean interface for training regression models.
It works with the Model Registry to:
1. Train individual models or multiple models
2. Handle model-specific preprocessing needs
3. Return trained models with metadata
4. Support cross-validation for better evaluation

"""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)

from Config.config import Config

logger = logging.getLogger(__name__)


# Result Container
@dataclass
class RegressionResult:
    model_name: str
    model: Any
    task_type: str = "regression"

    train_time: float = 0.0
    n_samples_train: int = 0
    n_features: int = 0

    # Training metrics
    train_mse: float = 0.0
    train_rmse: float = 0.0
    train_mae: float = 0.0
    train_r2: float = 0.0

    # Validation metrics
    val_mse: float = 0.0
    val_rmse: float = 0.0
    val_mae: float = 0.0
    val_r2: float = 0.0
    val_mape: Optional[float] = None
    val_explained_variance: Optional[float] = None

    # Cross-validation
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None

    # Extras
    residuals: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None

    required_scaling: bool = False
    scaler: Optional[Any] = None



# Trainer
class RegressionTrainer:
    """
    Regression model trainer.

    Responsibilities:
    - Fit models on X_train
    - Evaluate on X_val
    - Run CV ONLY on training data
    - Collect metrics and diagnostics

    """

    def __init__(self, model_registry, config: Config = None):
        self.registry = model_registry
        self.config = config or Config()
        self.random_state = self.config.RANDOM_STATE

        logger.info("RegressionTrainer initialized")


    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        custom_params: Optional[Dict] = None,
        run_cv: bool = True,
        cv_folds: Optional[int] = None
    ) -> RegressionResult:

        start_time = time.time()

        model = self.registry.get_model(
            model_name,
            task_type="regression",
            custom_params=custom_params
        )

        model_info = self.registry.get_model_info(model_name, "regression")

        # Data is already scaled from preprocessing pipeline
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy() if X_val is not None else None

        model.fit(X_train_processed, y_train)
        train_time = time.time() - start_time

        result = RegressionResult(
            model_name=model_name,
            model=model,
            train_time=train_time,
            n_samples_train=len(X_train),
            n_features=X_train.shape[1],
            required_scaling=model_info.requires_scaling,
            scaler=None  # No scaler needed
        )

        # ---------------- Training metrics ----------------
        y_train_pred = model.predict(X_train_processed)
        result.train_mse = mean_squared_error(y_train, y_train_pred)
        result.train_rmse = np.sqrt(result.train_mse)
        result.train_mae = mean_absolute_error(y_train, y_train_pred)
        result.train_r2 = r2_score(y_train, y_train_pred)

        # ---------------- Validation metrics ----------------
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val_processed)

            result.val_mse = mean_squared_error(y_val, y_val_pred)
            result.val_rmse = np.sqrt(result.val_mse)
            result.val_mae = mean_absolute_error(y_val, y_val_pred)
            result.val_r2 = r2_score(y_val, y_val_pred)

            if not (y_val == 0).any():
                result.val_mape = mean_absolute_percentage_error(y_val, y_val_pred)

            result.val_explained_variance = explained_variance_score(
                y_val, y_val_pred
            )

            result.residuals = y_val - y_val_pred

        # ---------------- Cross-validation ----------------
        if run_cv:
            cv_folds = cv_folds or self.config.CV_FOLDS
            cv_scores = cross_val_score(
                model,
                X_train_processed,
                y_train,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=self.config.N_JOBS
            )

            cv_rmse = np.sqrt(-cv_scores)
            result.cv_scores = cv_rmse.tolist()
            result.cv_mean = float(cv_rmse.mean())
            result.cv_std = float(cv_rmse.std())

        # ---------------- Feature importance ----------------
        if hasattr(model, "feature_importances_"):
            result.feature_importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            result.feature_importance = np.abs(model.coef_).flatten()

        return result


    def train_multiple(
        self,
        model_names: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        run_cv: bool = True,
        show_progress: bool = True
    ) -> Dict[str, RegressionResult]:

        results = {}

        for i, name in enumerate(model_names, 1):
            if show_progress:
                print(f"\n[{i}/{len(model_names)}] Training {name}...")

            result = self.train_model(
                name,
                X_train,
                y_train,
                X_val,
                y_val,
                run_cv=run_cv
            )

            results[name] = result

            if show_progress:
                print(
                    f"  ✅ R²: {result.val_r2:.3f} | "
                    f"RMSE: {result.val_rmse:.3f} | "
                    f"Time: {result.train_time:.2f}s"
                )

        return results


    def compare_results(self, results: Dict[str, RegressionResult]) -> pd.DataFrame:
        rows = []

        for name, r in results.items():
            rows.append({
                "Model": name,
                "Train R²": r.train_r2,
                "Val R²": r.val_r2,
                "Val RMSE": r.val_rmse,
                "Val MAE": r.val_mae,
                "CV Mean RMSE": r.cv_mean,
                "CV Std": r.cv_std,
                "Train Time (s)": r.train_time
            })

        return pd.DataFrame(rows).sort_values(
            "Val R²", ascending=False
        ).reset_index(drop=True)

    def get_best_model(
            self,
            results: Dict[str, RegressionResult],
            metric: str = "val_r2",
            alpha: float = 0.3  # Overfitting penalty weight
    ) -> Tuple[str, RegressionResult]:
        """
        Get the best performing model based on a metric with overfitting penalty.
        """

        if metric == "val_r2":
            # Higher R² is better, penalize overfitting
            key_fn = lambda r: r.val_r2 * (1 - alpha * abs(r.train_r2 - r.val_r2))
            best = max(results, key=lambda k: key_fn(results[k]))
        elif metric == "val_rmse":
            # Lower RMSE is better, penalize overfitting (invert for maximization)
            key_fn = lambda r: (1 / (1 + r.val_rmse)) * (1 - alpha * abs(r.train_r2 - r.val_r2))
            best = max(results, key=lambda k: key_fn(results[k]))
        else:
            raise ValueError("Unsupported metric")

        return best, results[best]


    def analyze_residuals(self, result: RegressionResult) -> Dict[str, float]:
        r = result.residuals
        return {
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "median": float(np.median(r)),
            "min": float(np.min(r)),
            "max": float(np.max(r))
        }