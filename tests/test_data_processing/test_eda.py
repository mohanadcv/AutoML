"""Test EDA visualizer functionality."""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from src.visualizations.eda import EDAGenerator


def test_eda_generation(classification_data):
    """Test EDA visualizer generates plots."""
    eda_gen = EDAGenerator()
    figures = eda_gen.generate_all(classification_data, 'target', 'classification')
    plt.show()

    assert len(figures) > 0
    assert 'distributions' in figures
    assert 'correlation' in figures
    assert 'target_analysis' in figures

    # Clean up
    plt.close('all')


def test_eda_returns_figures(classification_data):
    """Test EDA returns matplotlib figures."""
    eda_gen = EDAGenerator()
    figures = eda_gen.generate_all(classification_data, 'target', 'classification')
    plt.show()

    for fig_name, fig in figures.items():
        assert fig is not None
        assert hasattr(fig, 'savefig')  # Check it's a matplotlib figure