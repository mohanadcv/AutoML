"""Test model registry functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.registry import ModelRegistry


def test_get_classification_models(model_registry):
    """Test retrieving classification models."""
    models = model_registry.get_models_for_task('classification')

    assert len(models) > 0
    assert 'Random Forest' in models
    assert 'Logistic Regression' in models


def test_get_regression_models(model_registry):
    """Test retrieving regression models."""
    models = model_registry.get_models_for_task('regression')

    assert len(models) > 0
    assert 'Linear Regression' in models
    assert 'Ridge' in models


def test_initialize_model(model_registry):
    """Test model initialization."""
    model = model_registry.get_model('Random Forest', 'classification')

    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_get_model_info(model_registry):
    """Test retrieving model metadata."""
    info = model_registry.get_model_info('Random Forest', 'classification')

    assert info.name == 'Random Forest'
    assert 'classification' in info.task_types
    assert len(info.strengths) > 0


def test_get_fast_models(model_registry):
    """Test retrieving fast models."""
    fast_models = model_registry.get_fast_models('classification')

    assert len(fast_models) > 0
    assert 'Logistic Regression' in fast_models


def test_get_interpretable_models(model_registry):
    """Test retrieving interpretable models."""
    interpretable = model_registry.get_interpretable_models('regression')

    assert len(interpretable) > 0
    assert 'Linear Regression' in interpretable