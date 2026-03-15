"""Test data loading functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
from src.data_processing.loader import DataLoader


def test_load_csv(sample_csv_file):
    """Test loading CSV file from path."""
    loader = DataLoader()
    df = loader.load(sample_csv_file)

    assert df.shape == (5, 3)
    assert list(df.columns) == ['feature1', 'feature2', 'target']


def test_file_info(tmp_path):
    """Test file info retrieval."""
    file_path = tmp_path / "test_data.csv"

    # Create LARGE dataset (10,000 rows)
    test_data = pd.DataFrame({
        'feature1': range(10000),
        'feature2': ['value'] * 10000,
        'target': [0, 1] * 5000
    })

    test_data.to_csv(file_path, index=False)

    loader = DataLoader()
    info = loader.get_file_info(file_path)

    assert info['size_mb'] > 0.1  # Now > 0.1 MB (realistic)


def test_file_preview(sample_csv_file):
    """Test file preview functionality."""
    # Write data to file
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': ['a', 'b', 'c', 'd', 'e'],
        'target': [0, 1, 0, 1, 0]
    })
    data.to_csv(sample_csv_file, index=False)

    loader = DataLoader()
    preview = loader.preview_file(sample_csv_file, n_rows=3)

    assert len(preview) == 3
    assert list(preview.columns) == ['feature1', 'feature2', 'target']


def test_load_nonexistent_file():
    """Test loading non-existent file raises error."""
    loader = DataLoader()
    with pytest.raises(Exception):
        loader.load(Path("nonexistent.csv"))