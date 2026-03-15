"""Test hyperparameter tuning functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from Config.config import Config

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from src.models.registry import ModelRegistry
from src.models.hyperparameter_tuning_setup import HyperparameterTuner



def test_tune_single_model_classification(model_registry, split_data_classification):
    """Test tuning a single classification model."""
    X_train, X_val, y_train, y_val = split_data_classification
    tuner = HyperparameterTuner(model_registry, task_type='classification')

    result = tuner.tune_model(
        'Random Forest', X_train, y_train, X_val, y_val, n_iter=5, cv=3
    )

    assert result.model_name == 'Random Forest'
    assert result.best_params is not None
    assert result.val_score > 0


def test_tune_multiple_models_classification(model_registry, split_data_classification):
    """Test tuning multiple classification models."""
    X_train, X_val, y_train, y_val = split_data_classification
    tuner = HyperparameterTuner(model_registry, task_type='classification')

    models = ['Logistic Regression', 'Random Forest', 'Decision Tree']
    results = tuner.tune_multiple(models, X_train, y_train, X_val, y_val, n_iter=5, cv=3)

    assert len(results) == 3
    assert all(name in results for name in models)


def test_tune_regression_models(model_registry, split_data_regression):
    """Test tuning regression models."""
    X_train, X_val, y_train, y_val = split_data_regression
    tuner = HyperparameterTuner(model_registry, task_type='regression')

    models = ['Ridge', 'Random Forest']
    results = tuner.tune_multiple(models, X_train, y_train, X_val, y_val, n_iter=5, cv=3)

    assert len(results) == 2
    assert 'Ridge' in results
    assert 'Random Forest' in results


def test_compare_tuning_results(model_registry, split_data_classification):
    """Test tuning results comparison."""
    X_train, X_val, y_train, y_val = split_data_classification
    tuner = HyperparameterTuner(model_registry, task_type='classification')

    models = ['Logistic Regression', 'Random Forest']
    results = tuner.tune_multiple(models, X_train, y_train, X_val, y_val, n_iter=5, cv=3)

    comparison = tuner.compare_results(results)
    assert len(comparison) == 2
    assert 'Best CV Score' in comparison.columns
    assert 'Val Score' in comparison.columns