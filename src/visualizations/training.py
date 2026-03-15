"""
Visualization of Trained Models - Regression & Classification
Plots professional comparison charts for models trained with RegressionTrainer or ClassificationTrainer.
"""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted", font_scale=1.1)


# Regression Visualization
def plot_regression_results(df: pd.DataFrame, title: str = "Regression Model Comparison"):
    """
    Plots professional comparison charts for regression models.

    Metrics included:
    - Val R²
    - Val RMSE
    - Train time
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Barplot Val R²
    sns.barplot(x="Model", y="Val R²", data=df, ax=axes[0])
    axes[0].set_title("Validation R²")
    axes[0].set_ylim(0, 1)

    # Barplot Val RMSE
    sns.barplot(x="Model", y="Val RMSE", data=df, ax=axes[1])
    axes[1].set_title("Validation RMSE")
    axes[1].invert_yaxis()  # Lower RMSE better

    # Train time
    sns.barplot(x="Model", y="Train Time (s)", data=df, ax=axes[2])
    axes[2].set_title("Training Time (s)")

    for ax in axes:
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# Classification Visualization
def plot_classification_results(df: pd.DataFrame, title: str = "Classification Model Comparison"):
    """
    Plots professional comparison charts for classification models.

    Metrics included:
    - Val Accuracy
    - Val F1
    - Train time
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Val Accuracy
    sns.barplot(x="Model", y="Val Accuracy", data=df, ax=axes[0])
    axes[0].set_title("Validation Accuracy")
    axes[0].set_ylim(0, 1)

    # Val F1
    sns.barplot(x="Model", y="Val F1", data=df, ax=axes[1])
    axes[1].set_title("Validation F1 Score")
    axes[1].set_ylim(0, 1)

    # Train Time
    sns.barplot(x="Model", y="Train Time (s)", data=df, ax=axes[2])
    axes[2].set_title("Training Time (s)")

    for ax in axes:
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
