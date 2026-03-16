"""
Final Test Evaluation – Regression & Classification

Applies tuned models on the test set and generates professional
comparison visualizations for all metrics.

"""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


# Regression Test Evaluation
def evaluate_test_regression(tuned_results, X_test, y_test):
    """
    Apply tuned regression models on the test set and return DataFrame of metrics.
    """
    records = []

    for model_name, result in tuned_results.items():
        X_test_proc = X_test.copy()

        # Get the actual model - handle both TrainingResult and TuningResult
        if hasattr(result, 'model'):
            model = result.model
        elif hasattr(result, 'best_model'):
            model = result.best_model
        else:
            continue  # Skip if no model found

        # Data is already scaled from preprocessing pipeline
        y_pred = model.predict(X_test)

        records.append({
            "Model": model_name,
            "Test MSE": mean_squared_error(y_test, y_pred),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Test MAE": mean_absolute_error(y_test, y_pred),
            "Test R²": r2_score(y_test, y_pred),
            "Test Explained Variance": explained_variance_score(y_test, y_pred)
        })

    return pd.DataFrame(records)


def plot_test_regression(df_test):
    """
    Professional regression visualization:
    1. R² & Explained Variance
    2. Error metrics: MSE, RMSE, MAE
    """
    # ---------------- R² & Explained Variance ----------------
    df_r2 = df_test.melt(
        id_vars="Model",
        value_vars=["Test R²", "Test Explained Variance"],
        var_name="Metric",
        value_name="Value"
    )
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x="Model", y="Value", hue="Metric", data=df_r2, palette="Set2")
    plt.title("Regression Models – R² & Explained Variance on Test Set", fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# Classification Test Evaluation
def evaluate_test_classification(tuned_results, X_test, y_test):
    """
    Apply tuned classification models on the test set and return DataFrame of metrics.
    """
    records = []

    for model_name, result in tuned_results.items():
        X_test_proc = X_test.copy()

        # Get the actual model - handle both TrainingResult and TuningResult
        if hasattr(result, 'model'):
            model = result.model
        elif hasattr(result, 'best_model'):
            model = result.best_model
        else:
            continue  # Skip if no model found

        # Data is already scaled from preprocessing pipeline
        y_pred = model.predict(X_test_proc)

        avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            else:
                roc_auc = np.nan
        except:
            roc_auc = np.nan

        records.append({
            "Model": model_name,
            "Test Accuracy": accuracy,
            "Test Precision": precision,
            "Test Recall": recall,
            "Test F1": f1,
            "Test ROC AUC": roc_auc
        })

    return pd.DataFrame(records)


def plot_test_classification(df_test):
    """
    Plot classification metrics for the test set.
    """
    metrics = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC AUC"]
    df_melt = df_test.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")

    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="Value", hue="Metric", data=df_melt, palette="Spectral")
    plt.title("Classification Models – Test Set Metrics", fontsize=16)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig