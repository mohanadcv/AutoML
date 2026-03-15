"""Test training visualization functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from src.visualizations.training import plot_regression_results, plot_classification_results



def test_regression_visualizer(model_registry, split_data_regression):
    """Test regression results plotting."""
    from src.models.trainers.regression import RegressionTrainer

    X_train, X_val, y_train, y_val = split_data_regression
    trainer = RegressionTrainer(model_registry)

    models = ["Linear Regression", "Ridge"]
    results = trainer.train_multiple(models, X_train, y_train, X_val, y_val, run_cv=False)
    comparison = trainer.compare_results(results)

    fig = plot_regression_results(comparison)
    plt.show()
    assert fig is not None
    plt.close(fig)


def test_classification_visualizer(model_registry, split_data_classification):
    """Test classification results plotting."""
    from src.models.trainers.classification import ClassificationTrainer

    X_train, X_val, y_train, y_val = split_data_classification
    trainer = ClassificationTrainer(model_registry)

    models = ['Logistic Regression', 'Random Forest']
    results = trainer.train_multiple(models, X_train, y_train, X_val, y_val, run_cv=False)
    comparison = trainer.compare_results(results)

    fig = plot_classification_results(comparison)
    plt.show()
    assert fig is not None
    plt.close(fig)