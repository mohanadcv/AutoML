"""
Visualization of Tuned Models - Regression & Classification
Plots professional comparison charts for tuned models from HyperparameterTuner.
"""
# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", palette="muted", font_scale=1.1)


# Regression Visualization
def plot_tuned_regression_results(df: pd.DataFrame, title: str = "Tuned Regression Models Comparison"):
    """
    Plots comparison for tuned regression models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Best CV Score
    sns.barplot(x="Model", y="Best CV Score", data=df, ax=axes[0])
    axes[0].set_title("Best CV Score")
    axes[0].set_ylim(0, 1)

    # Val Score
    sns.barplot(x="Model", y="Val Score", data=df, ax=axes[1])
    axes[1].set_title("Validation R²")
    axes[1].set_ylim(0, 1)

    # Tuning Time
    sns.barplot(x="Model", y="Tuning Time (s)", data=df, ax=axes[2])
    axes[2].set_title("Tuning Time (s)")

    for ax in axes:
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig



# Classification Visualization
def plot_tuned_classification_results(df: pd.DataFrame, title: str = "Tuned Classification Models Comparison"):
    """
    Plots comparison for tuned classification models.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    sns.barplot(x="Model", y="Best CV Score", data=df, ax=axes[0])
    axes[0].set_title("Best CV Score")
    axes[0].set_ylim(0, 1)

    sns.barplot(x="Model", y="Val Score", data=df, ax=axes[1])
    axes[1].set_title("Validation Accuracy")
    axes[1].set_ylim(0, 1)

    sns.barplot(x="Model", y="Val Precision", data=df, ax=axes[2])
    axes[2].set_title("Val Precision")
    axes[2].set_ylim(0, 1)

    sns.barplot(x="Model", y="Val F1", data=df, ax=axes[3])
    axes[3].set_title("Val F1 Score")
    axes[3].set_ylim(0, 1)

    for ax in axes:
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig