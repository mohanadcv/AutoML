"""Test data splitting functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.splitting import DataSplitter
from src.data_processing.validator import DataValidator


def test_balanced_classification_split(imbalanced_data):
    """Test splitting balanced classification data."""
    X, y = imbalanced_data
    splitter = DataSplitter()

    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        X, y, 'classification'
    )

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(X)
    assert len(X_train) > len(X_val) > 0


def test_imbalanced_classification_split(imbalanced_data):
    """Test splitting imbalanced classification data."""
    X, y = imbalanced_data
    splitter = DataSplitter()

    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        X, y, 'classification'
    )

    # Check class distribution preserved approximately
    original_ratio = (y == 1).sum() / len(y)
    train_ratio = (y_train == 1).sum() / len(y_train)

    assert abs(train_ratio - original_ratio) < 0.05


def test_regression_split(regression_data):
    """Test splitting regression data."""
    X, y = regression_data
    splitter = DataSplitter()

    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        X, y, 'regression'
    )

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0


def test_rare_class_validation(rare_class_data):
    """Test rare class validation (3 samples should pass)."""
    X, y = rare_class_data
    validator = DataValidator()

    is_valid, msg = validator.validate_target(y, 'target')
    assert is_valid is True
    assert "Target is valid" in msg


def test_split_info(rare_class_data, balanced_small_data):
    """Test split info dictionary generation."""
    X, y = balanced_small_data
    splitter = DataSplitter()

    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(
        X, y, 'classification'
    )

    info = splitter.get_split_info(
        X_train, X_val, X_test, y_train, y_val, y_test, 'classification'
    )

    assert 'total_samples' in info
    assert 'train_pct' in info
    assert info['total_samples'] == len(X)