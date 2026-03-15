"""Test task type detection functionality."""


# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data_processing.task_detector import detect_task_type


def test_binary_classification_detection():
    """Test binary classification detection."""
    y = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    task, confidence = detect_task_type(y)

    assert task == 'classification'
    assert confidence > 0.7


def test_multiclass_detection():
    """Test multi-class classification detection."""
    y = pd.Series(['cat', 'dog', 'bird', 'cat', 'dog', 'bird'])
    task, confidence = detect_task_type(y)

    assert task == 'classification'
    assert confidence > 0.7


def test_regression_detection():
    """Test regression detection."""
    y = pd.Series([24.5, 31.2, 28.9, 35.1, 29.3, 32.7, 27.8])
    task, confidence = detect_task_type(y)

    assert task == 'regression'
    assert confidence > 0.5


def test_small_integers_detection():
    """Test detection for small integer values (should be classification)."""
    y = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3,
                   1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2])
    task, confidence = detect_task_type(y)

    assert task == 'classification'