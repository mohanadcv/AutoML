"""Test tuning visualization functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

from src.visualizations.tuning import plot_tuned_regression_results, plot_tuned_classification_results



def test_tuned_regression_visualizer(model_registry, split_data_regression):
    """Test tuned regression results plotting."""
    from src.models.hyperparameter_tuning_setup import HyperparameterTuner

    X_train, X_val, y_train, y_val = split_data_regression
    tuner = HyperparameterTuner(model_registry, task_type="regression")

    models = ["Ridge", "Random Forest"]
    results = tuner.tune_multiple(models, X_train, y_train, X_val, y_val, n_iter=5, cv=3)
    comparison = tuner.compare_results(results)

    fig = plot_tuned_regression_results(comparison)
    plt.show()
    assert fig is not None
    plt.close(fig)


def test_tuned_classification_visualizer(model_registry, split_data_classification):
    """Test tuned classification results plotting."""
    from src.models.hyperparameter_tuning_setup import HyperparameterTuner

    X_train, X_val, y_train, y_val = split_data_classification
    tuner = HyperparameterTuner(model_registry, task_type="classification")

    models = ["Logistic Regression", "Random Forest"]
    results = tuner.tune_multiple(models, X_train, y_train, X_val, y_val, n_iter=5, cv=3)
    comparison = tuner.compare_results(results)

    fig = plot_tuned_classification_results(comparison)
    plt.show()
    assert fig is not None
    plt.close(fig)