"""Shared test fixtures for all AutoML tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['a', 'b', 'c', 'd', 'e'],
        'target': [0, 1, 0, 1, 0]
    })
    file_path = tmp_path / "test_data.csv"
    data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_dataframe():
    """Base DataFrame for validator tests."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
        'feature3': [1.1, 2.2, np.nan, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


@pytest.fixture
def classification_data(random_seed):
    """Generate synthetic classification dataset."""
    np.random.seed(random_seed)
    n_samples = 200
    return pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'score': np.random.uniform(0, 100, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })


@pytest.fixture
def binary_classification_data(random_seed):
    """Binary classification dataset for model training."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=random_seed
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y, name='target')
    return X, y


@pytest.fixture
def multiclass_classification_data(random_seed):
    """Multi-class classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=random_seed
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y, name='target')
    return X, y


@pytest.fixture
def regression_data(random_seed):
    """Synthetic regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=15,
        noise=50,
        random_state=random_seed
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y, name='target')
    return X, y


@pytest.fixture
def rare_class_data():
    """Dataset with rare class (3 samples) for stratification testing."""
    X = pd.DataFrame({'feature1': np.random.randn(50)})
    y = pd.Series([0] * 47 + [1] * 3, name='target')
    return X, y


@pytest.fixture
def balanced_small_data():
    """Small balanced dataset (50/50 split)."""
    X = pd.DataFrame({'feature1': np.random.randn(50)})
    y = pd.Series([0] * 25 + [1] * 25, name='target')
    return X, y


@pytest.fixture
def imbalanced_data(random_seed):
    """Imbalanced classification dataset."""
    np.random.seed(random_seed)
    n_samples = 200
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples)
    })
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.9, 0.1]), name='target')
    return X, y


@pytest.fixture
def split_data_regression(regression_data):
    """Pre-split regression data into train/val."""
    from sklearn.model_selection import train_test_split
    X, y = regression_data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def split_data_classification(multiclass_classification_data):
    """Pre-split classification data into train/val."""
    from sklearn.model_selection import train_test_split
    X, y = multiclass_classification_data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def split_data_test_eval(regression_data, multiclass_classification_data):
    """Create train/val/test splits for final evaluation."""
    from sklearn.model_selection import train_test_split

    # Regression
    X_reg, y_reg = regression_data
    Xr_train, Xr_temp, yr_train, yr_temp = train_test_split(
        X_reg, y_reg, test_size=0.4, random_state=42
    )
    Xr_val, Xr_test, yr_val, yr_test = train_test_split(
        Xr_temp, yr_temp, test_size=0.5, random_state=42
    )

    # Classification
    X_clf, y_clf = multiclass_classification_data
    Xc_train, Xc_temp, yc_train, yc_temp = train_test_split(
        X_clf, y_clf, test_size=0.4, random_state=42
    )
    Xc_val, Xc_test, yc_val, yc_test = train_test_split(
        Xc_temp, yc_temp, test_size=0.5, random_state=42
    )

    return {
        'regression': (Xr_train, Xr_val, Xr_test, yr_train, yr_val, yr_test),
        'classification': (Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test)
    }


@pytest.fixture
def model_registry():
    """Provide a ModelRegistry instance."""
    from src.models.registry import ModelRegistry
    return ModelRegistry()


@pytest.fixture
def mock_config():
    from Config.config import Config
    config = Config()
    config.CV_FOLDS = 2
    config.N_JOBS = 1
    return config