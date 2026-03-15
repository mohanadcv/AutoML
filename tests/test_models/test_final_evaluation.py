"""Test final evaluation functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

from src.models.final_evaluation import evaluate_test_classification, plot_test_classification
from src.models.final_evaluation import evaluate_test_regression, plot_test_regression



def test_regression_final_evaluation(model_registry, split_data_test_eval):
    """Test regression final evaluation."""
    from src.models.trainers.regression import RegressionTrainer

    data = split_data_test_eval['regression']
    X_train, X_val, X_test, y_train, y_val, y_test = data

    trainer = RegressionTrainer(model_registry)
    models = ["Linear Regression", "Ridge"]
    results = trainer.train_multiple(models, X_train, y_train, X_val, y_val)

    df_test = evaluate_test_regression(results, X_test, y_test)

    assert len(df_test) == 2
    assert 'Test R²' in df_test.columns
    assert 'Test RMSE' in df_test.columns


def test_classification_final_evaluation(model_registry, split_data_test_eval):
    """Test classification final evaluation."""
    from src.models.trainers.classification import ClassificationTrainer

    data = split_data_test_eval['classification']
    X_train, X_val, X_test, y_train, y_val, y_test = data

    trainer = ClassificationTrainer(model_registry)
    models = ["Logistic Regression", "Random Forest"]
    results = trainer.train_multiple(models, X_train, y_train, X_val, y_val)

    df_test = evaluate_test_classification(results, X_test, y_test)

    assert len(df_test) == 2
    assert 'Test Accuracy' in df_test.columns
    assert 'Test F1' in df_test.columns


def test_regression_plot(model_registry, split_data_test_eval):
    """Test regression final evaluation plotting."""
    from src.models.trainers.regression import RegressionTrainer

    data = split_data_test_eval['regression']
    X_train, X_val, X_test, y_train, y_val, y_test = data

    trainer = RegressionTrainer(model_registry)
    models = ["Linear Regression", "Ridge"]
    results = trainer.train_multiple(models, X_train, y_train, X_val, y_val)

    df_test = evaluate_test_regression(results, X_test, y_test)
    fig = plot_test_regression(df_test)
    plt.show()

    assert fig is not None
    plt.close(fig)


def test_classification_plot(model_registry, split_data_test_eval):
    """Test classification final evaluation plotting."""
    from src.models.trainers.classification import ClassificationTrainer

    data = split_data_test_eval['classification']
    X_train, X_val, X_test, y_train, y_val, y_test = data

    trainer = ClassificationTrainer(model_registry)
    models = ["Logistic Regression", "Random Forest"]
    results = trainer.train_multiple(models, X_train, y_train, X_val, y_val)

    df_test = evaluate_test_classification(results, X_test, y_test)
    fig = plot_test_classification(df_test)
    plt.show()

    assert fig is not None
    plt.close(fig)