"""Test preprocessing pipeline functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data_processing.preprocessing_pipeline import PreprocessingPipeline



def test_pipeline_fitting(classification_data):
    """Test pipeline fitting."""
    X = classification_data.drop('target', axis=1)
    y = classification_data['target']

    pipeline = PreprocessingPipeline()
    pipeline.fit(X, y)

    assert hasattr(pipeline, 'numeric_features_')
    assert hasattr(pipeline, 'categorical_features_')
    assert hasattr(pipeline, 'low_card_categorical_')


def test_pipeline_transform(classification_data):
    """Test data transformation."""
    X = classification_data.drop('target', axis=1)
    y = classification_data['target']

    pipeline = PreprocessingPipeline()
    X_processed = pipeline.fit_transform(X, y)

    assert isinstance(X_processed, np.ndarray)
    assert X_processed.shape[0] == len(X)
    assert X_processed.shape[1] > 0


def test_id_column_detection():
    """Test ID column detection and removal."""
    n_samples = 100
    X = pd.DataFrame({
        'numeric': np.random.randn(n_samples),
        'id_column': [f'ID_{i}' for i in range(n_samples)],  # 100% unique
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    y = pd.Series(np.random.choice([0, 1], n_samples))

    pipeline = PreprocessingPipeline(id_threshold=0.95)
    pipeline.fit(X, y)

    assert 'id_column' in pipeline.id_columns_
    assert 'id_column' not in pipeline.numeric_features_
    assert 'id_column' not in pipeline.categorical_features_


def test_frequency_encoding():
    """Test frequency encoding for high-cardinality features."""
    from src.data_processing import FrequencyEncoder

    df = pd.DataFrame({
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'NYC']
    })

    encoder = FrequencyEncoder()
    encoder.fit(df)
    transformed = encoder.transform(df)

    # NYC appears 3/6 = 0.5
    assert abs(transformed['city'].iloc[0] - 0.5) < 0.001