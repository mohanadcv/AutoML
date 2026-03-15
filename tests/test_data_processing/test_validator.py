"""Test data validation functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data_processing.validator import DataValidator


def test_dataframe_validation(sample_dataframe):
    """Test DataFrame validation passes for valid data."""
    validator = DataValidator()
    valid, msg = validator.validate_dataframe(sample_dataframe)

    assert valid is True
    assert "DataFrame is valid" in msg


def test_data_quality_report(sample_dataframe):
    """Test data quality report generation."""
    validator = DataValidator()
    quality = validator.check_data_quality(sample_dataframe)

    assert quality['n_rows'] == 10
    assert quality['n_columns'] == 4
    assert quality['null_summary']['total_nulls'] == 1
    assert quality['duplicate_rows'] == 0


def test_target_validation(sample_dataframe):
    """Test target variable validation."""
    validator = DataValidator()
    valid, msg = validator.validate_target(sample_dataframe['target'], 'target')

    assert valid is True
    assert "Target is valid" in msg


def test_feature_target_split(sample_dataframe):
    """Test feature-target split validation."""
    validator = DataValidator()
    X = sample_dataframe.drop('target', axis=1)
    y = sample_dataframe['target']
    valid, msg = validator.validate_feature_target_split(X, y)

    assert valid is True


def test_too_few_rows():
    """Test validation fails for DataFrame with too few rows."""
    validator = DataValidator()
    bad_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    valid, msg = validator.validate_dataframe(bad_df)

    assert valid is False
    assert "Too few rows" in msg


def test_constant_target():
    """Test validation fails for constant target."""
    validator = DataValidator()
    bad_target = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    valid, msg = validator.validate_target(bad_target, 'bad_target')

    assert valid is False
    assert "only one unique value" in msg