"""Test regression trainer functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trainers.regression import RegressionTrainer


def test_train_single_model(model_registry, split_data_regression):
    """Test training a single regression model."""
    X_train, X_val, y_train, y_val = split_data_regression
    trainer = RegressionTrainer(model_registry)

    result = trainer.train_model(
        'Random Forest', X_train, y_train, X_val, y_val, run_cv=False
    )

    assert result.model_name == 'Random Forest'
    assert result.val_r2 > 0
    assert result.val_rmse > 0


def test_train_multiple_models(model_registry, split_data_regression):
    """Test training multiple regression models."""
    X_train, X_val, y_train, y_val = split_data_regression
    trainer = RegressionTrainer(model_registry)

    models = ["Linear Regression", "Ridge", "Random Forest"]
    results = trainer.train_multiple(models, X_train, y_train, X_val, y_val, run_cv=False)

    assert len(results) == 3
    assert all(name in results for name in models)


def test_compare_results(model_registry, split_data_regression):
    """Test results comparison."""
    X_train, X_val, y_train, y_val = split_data_regression
    trainer = RegressionTrainer(model_registry)

    models = ["Linear Regression", "Ridge"]
    results = trainer.train_multiple(models, X_train, y_train, X_val, y_val, run_cv=False)

    comparison = trainer.compare_results(results)
    assert len(comparison) == 2
    assert 'Val R²' in comparison.columns


def test_get_best_model(model_registry, split_data_regression):
    """Test best model selection."""
    X_train, X_val, y_train, y_val = split_data_regression
    trainer = RegressionTrainer(model_registry)

    models = ["Linear Regression", "Ridge"]
    results = trainer.train_multiple(models, X_train, y_train, X_val, y_val, run_cv=False)

    best_name, best_result = trainer.get_best_model(results)
    assert best_name in models
    assert best_result.val_r2 > 0