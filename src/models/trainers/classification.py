"""
Classification Models - Helper Module for Training Classification Models

This module provides a clean interface for training classification models.
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
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

from src.models.registry import ModelRegistry
from Config.config import Config

logger = logging.getLogger(__name__)



# Result Container
@dataclass
class ClassificationResult:
    """
    Container for classification model results.

    Stores everything about a trained model:
    - The model itself
    - Training metrics
    - Validation metrics
    - Cross-validation scores
    - Training time
    - Model metadata
    """
    model_name: str
    model: Any
    task_type: str = 'classification'

    # Training info
    train_time: float = 0.0
    n_samples_train: int = 0
    n_features: int = 0

    # Training metrics
    train_accuracy: float = 0.0
    train_precision: float = 0.0
    train_recall: float = 0.0
    train_f1: float = 0.0

    # Validation metrics
    val_accuracy: float = 0.0
    val_precision: float = 0.0
    val_recall: float = 0.0
    val_f1: float = 0.0
    val_roc_auc: Optional[float] = None

    # Cross-validation
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None

    # Additional info
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None
    feature_importance: Optional[np.ndarray] = None

    # Preprocessing info
    required_scaling: bool = False
    scaler: Optional[Any] = None



# Trainer
class ClassificationTrainer:
    """
    🎯 Classification Model Trainer

    Handles the complete training workflows:
    - Feature scaling (if needed)
    - Model training
    - Performance evaluation
    - Cross-validation
    - Result collection
    """

    def __init__(self, model_registry: ModelRegistry, config: Config = None):
        self.registry = model_registry
        self.config = config or Config()
        self.random_state = self.config.RANDOM_STATE
        logger.info("Classification Trainer initialized")

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        custom_params: Optional[Dict] = None,
        run_cv: bool = True,
        cv_folds: int = None
    ) -> ClassificationResult:
        """
        Train a single classification model (TRAIN + VAL only).
        """
        start_time = time.time()

        model = self.registry.get_model(
            model_name,
            task_type='classification',
            custom_params=custom_params
        )
        model_info = self.registry.get_model_info(model_name, 'classification')

        # Data is already scaled from preprocessing pipeline
        X_train_processed = X_train.copy()
        X_val_processed = X_val.copy() if X_val is not None else None

        model.fit(X_train_processed, y_train)
        train_time = time.time() - start_time

        result = ClassificationResult(
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
        result.train_accuracy = accuracy_score(y_train, y_train_pred)
        avg = 'binary' if len(np.unique(y_train)) == 2 else 'weighted'
        result.train_precision = precision_score(y_train, y_train_pred, average=avg, zero_division=0)
        result.train_recall = recall_score(y_train, y_train_pred, average=avg, zero_division=0)
        result.train_f1 = f1_score(y_train, y_train_pred, average=avg, zero_division=0)

        # ---------------- Validation metrics ----------------
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val_processed)
            result.val_accuracy = accuracy_score(y_val, y_val_pred)
            result.val_precision = precision_score(y_val, y_val_pred, average=avg, zero_division=0)
            result.val_recall = recall_score(y_val, y_val_pred, average=avg, zero_division=0)
            result.val_f1 = f1_score(y_val, y_val_pred, average=avg, zero_division=0)

            if hasattr(model, 'predict_proba'):
                try:
                    if len(np.unique(y_train)) == 2:
                        y_val_proba = model.predict_proba(X_val_processed)[:, 1]
                        result.val_roc_auc = roc_auc_score(y_val, y_val_proba)
                    else:
                        y_val_proba = model.predict_proba(X_val_processed)
                        result.val_roc_auc = roc_auc_score(
                            y_val, y_val_proba,
                            multi_class='ovr',
                            average='weighted'
                        )
                except Exception as e:
                    logger.debug(f"Could not calculate ROC AUC: {e}")

            result.confusion_matrix = confusion_matrix(y_val, y_val_pred)
            result.classification_report = classification_report(y_val, y_val_pred)

        # ---------------- Cross-validation ----------------
        if run_cv:
            cv_folds = cv_folds or self.config.CV_FOLDS
            try:
                cv_scores = cross_val_score(
                    model, X_train_processed, y_train,
                    cv=cv_folds, scoring='accuracy', n_jobs=self.config.N_JOBS
                )
                result.cv_scores = cv_scores.tolist()
                result.cv_mean = float(np.mean(cv_scores))
                result.cv_std = float(np.std(cv_scores))
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")

        # ---------------- Feature importance ----------------
        if hasattr(model, 'feature_importances_'):
            result.feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
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
    ) -> Dict[str, ClassificationResult]:
        """
        Train multiple classification models.
        """
        results = {}
        for i, name in enumerate(model_names, 1):
            if show_progress:
                print(f"[{i}/{len(model_names)}] Training {name}...")
            try:
                result = self.train_model(name, X_train, y_train, X_val, y_val, run_cv=run_cv)
                results[name] = result
                if show_progress:
                    print(f"  ✅ Accuracy: {result.val_accuracy:.3f} | Time: {result.train_time:.2f}s")
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                if show_progress:
                    print(f"  ❌ Failed: {e}")
        return results


    def compare_results(self, results: Dict[str, ClassificationResult]) -> pd.DataFrame:
        """
        Create comparison table of model results.
        """
        rows = []
        for name, r in results.items():
            rows.append({
                'Model': name,
                'Train Accuracy': r.train_accuracy,
                'Val Accuracy': r.val_accuracy,
                'Val Precision': r.val_precision,
                'Val Recall': r.val_recall,
                'Val F1': r.val_f1,
                'Val ROC AUC': r.val_roc_auc if r.val_roc_auc else np.nan,
                'CV Mean': r.cv_mean if r.cv_mean else np.nan,
                'CV Std': r.cv_std if r.cv_std else np.nan,
                'Train Time (s)': r.train_time
            })
        df = pd.DataFrame(rows)
        if 'Val Accuracy' in df.columns:
            return df.sort_values('Val Accuracy', ascending=False).reset_index(drop=True)
        else:
            # Try to sort by whatever metric is available
            available_metrics = ['Train Accuracy', 'CV Mean', 'Val F1', 'Val Precision']
            for metric in available_metrics:
                if metric in df.columns:
                    return df.sort_values(metric, ascending=False).reset_index(drop=True)
            # If nothing found, return unsorted
            return df.reset_index(drop=True)


    def get_best_model(self,
            results: Dict[str, ClassificationResult],
            metric: str = 'val_accuracy',
            alpha: float = 0.3  # Overfitting penalty weight
    ) -> Tuple[str, ClassificationResult]:
        """
        Get the best performing model based on a metric with overfitting penalty.
        """
        metric_funcs = {
            'val_accuracy': lambda r: r.val_accuracy * (1 - alpha * abs(r.train_accuracy - r.val_accuracy)),
            'val_f1': lambda r: r.val_f1 * (1 - alpha * abs(r.train_accuracy - r.val_accuracy)),
            'val_precision': lambda r: r.val_precision * (1 - alpha * abs(r.train_accuracy - r.val_accuracy)),
            'val_recall': lambda r: r.val_recall * (1 - alpha * abs(r.train_accuracy - r.val_accuracy)),
            'val_roc_auc': lambda r: (r.val_roc_auc if r.val_roc_auc else 0) *
                                     (1 - alpha * abs(r.train_accuracy - r.val_accuracy)),
            'cv_mean': lambda r: (r.cv_mean if r.cv_mean else 0) *
                                 (1 - alpha * abs(r.train_accuracy - r.val_accuracy)),
            'train_time': lambda r: -r.train_time  # No penalty for time
        }

        if metric not in metric_funcs:
            raise ValueError(f"Invalid metric: {metric}")

        best_name = max(results.keys(), key=lambda k: metric_funcs[metric](results[k]))
        return best_name, results[best_name]