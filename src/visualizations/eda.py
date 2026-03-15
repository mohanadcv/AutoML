"""
EDA Generator - Automatically generate EDA visualizations for any dataset

Creates task-appropriate visualizations:
- Feature distributions (histograms for numeric, bar charts for categorical)
- Correlation heatmap
- Target analysis dashboard

"""
# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


class EDAGenerator:
    """
    Automatically generates EDA visualizations for any dataset.
    Adapts to:
    - Number of features
    - Data types (numeric vs categorical)
    - Task type (classification vs regression)
    """

    def __init__(self, figsize_base: Tuple[int, int] = (15, 10)):
        self.figsize_base = figsize_base

    # ================================================================
    # MAIN ENTRY POINT
    # ================================================================
    def generate_all(self, df: pd.DataFrame, target_column: str,
                     task_type: str = 'classification') -> Dict[str, plt.Figure]:
        """
        Generate all EDA visualizations.

        Returns:
            Dictionary of {plot_name: matplotlib.Figure}
        """
        logger.info(f"Generating EDA for {df.shape[1]} features, task: {task_type}")

        figures = {}

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 1. Feature Distributions
        logger.info("Creating feature distributions...")
        figures['distributions'] = self.plot_feature_distributions(X)

        # 2. Correlation Matrix
        logger.info("Creating correlation matrix...")
        figures['correlation'], figures['correlation_table'] = self.plot_correlation_matrix(X)

        # 3. Target Analysis Dashboard
        logger.info("Creating target analysis dashboard...")
        figures['target_analysis'] = self.plot_target_analysis(
            X, y, target_column, task_type
        )

        logger.info("✅ EDA generation complete!")
        return figures


    # VISUALIZATION 1: FEATURE DISTRIBUTIONS
    def plot_feature_distributions(self, X: pd.DataFrame) -> plt.Figure:
        """
        Plot distributions for all features.
        - Numeric features → Histograms with KDE
        - Categorical features → Bar charts
        Automatically determines grid size based on number of features.
        """
        n_features = len(X.columns)
        n_cols = 2
        n_rows = int(np.ceil(n_features / n_cols))

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
        fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=0.995)

        # Flatten axes for easy iteration
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each feature
        for idx, col in enumerate(X.columns):
            ax = axes[idx] if n_features > 1 else axes[0]

            # Numeric features
            if X[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Remove nulls for plotting
                data = X[col].dropna()

                if len(data) > 0:
                    # Histogram with KDE
                    ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

                    # Add KDE if enough data points
                    if len(data) > 10:
                        try:
                            data.plot(kind='kde', ax=ax, secondary_y=True, color='red', linewidth=2)
                            ax.right_ax.set_ylabel('Density', fontsize=10)
                            ax.right_ax.grid(False)
                        except:
                            pass  # Skip KDE if it fails

                    ax.set_title(f'{col}\n(Numeric)', fontsize=11, fontweight='bold')
                    ax.set_xlabel('Value', fontsize=9)
                    ax.set_ylabel('Frequency', fontsize=9)

                    # Add stats
                    mean_val = data.mean()
                    median_val = data.median()
                    ax.axvline(mean_val, color='green', linestyle='--', linewidth=1.5,
                               label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='orange', linestyle='--', linewidth=1.5,
                               label=f'Median: {median_val:.2f}')
                    ax.legend(fontsize=8)

            # Categorical features
            else:
                data = X[col].dropna()

                if len(data) > 0:
                    # Get value counts
                    value_counts = data.value_counts().head(10)  # Top 10 categories

                    # Bar chart
                    value_counts.plot(kind='barh', ax=ax, color='coral', edgecolor='black')

                    # Count total unique categories
                    n_categories = len(data.value_counts())
                    n_shown = len(value_counts)  # How many we're showing

                    # Adaptive title
                    if n_shown < n_categories:
                        title = f'{col}\n(Categorical - Top {n_shown} of {n_categories})'
                    else:
                        title = f'{col}\n(Categorical - All {n_shown} categories)'

                    ax.set_title(title, fontsize=11, fontweight='bold')
                    ax.set_xlabel('Count', fontsize=9)
                    ax.set_ylabel('Category', fontsize=9)

                    # Add count labels
                    for i, (cat, count) in enumerate(value_counts.items()):
                        ax.text(count, i, f' {count}', va='center', fontsize=8)

            # Add null count if any
            null_count = X[col].isnull().sum()
            if null_count > 0:
                null_pct = (null_count / len(X)) * 100
                ax.text(0.02, 0.98, f'Missing: {null_count} ({null_pct:.1f}%)',
                        transform=ax.transAxes, fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # Hide empty subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    # VISUALIZATION 2: Data Quality
    def plot_data_quality(self, X: pd.DataFrame) -> plt.Figure:
        """
        Data Quality Dashboard — missing values + outlier overview.
        Shows at a glance what needs cleaning before any modeling.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        n_panels = 2 if len(numeric_cols) > 0 else 1

        fig, axes = plt.subplots(1, n_panels, figsize=(14, 6))
        fig.suptitle('Data Quality Overview', fontsize=16, fontweight='bold')

        if n_panels == 1:
            axes = [axes]

        # Panel 1: Missing values per column
        ax1 = axes[0]
        null_counts = X.isnull().sum()
        null_pct = (null_counts / len(X)) * 100
        null_df = pd.DataFrame({'Missing %': null_pct}).sort_values('Missing %', ascending=True)
        null_df = null_df[null_df['Missing %'] > 0]

        if len(null_df) == 0:
            ax1.text(0.5, 0.5, '✅ No missing values found',
                     ha='center', va='center', fontsize=14, color='green')
            ax1.axis('off')
        else:
            colors = ['#ef4444' if v > 30 else '#f59e0b' if v > 10 else '#3b82f6'
                      for v in null_df['Missing %']]
            null_df['Missing %'].plot(kind='barh', ax=ax1, color=colors)
            ax1.set_xlabel('Missing %')
            ax1.set_title('Missing Values by Column', fontweight='bold')
            for i, val in enumerate(null_df['Missing %']):
                ax1.text(val + 0.3, i, f'{val:.1f}%', va='center', fontsize=9)
            ax1.axvline(30, color='red', linestyle='--', linewidth=1, alpha=0.5, label='30% threshold')
            ax1.legend(fontsize=8)

        # Panel 2: Outlier count per numeric column (IQR method)
        if len(numeric_cols) > 0:
            ax2 = axes[1]
            outlier_counts = {}
            for col in numeric_cols:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)).sum()
                outlier_pct = (outliers / len(X)) * 100
                if outlier_pct > 0:
                    outlier_counts[col] = outlier_pct

            if not outlier_counts:
                ax2.text(0.5, 0.5, '✅ No outliers detected (IQR method)',
                         ha='center', va='center', fontsize=14, color='green')
                ax2.axis('off')
            else:
                outlier_series = pd.Series(outlier_counts).sort_values(ascending=True)
                colors2 = ['#ef4444' if v > 10 else '#f59e0b' if v > 5 else '#3b82f6'
                           for v in outlier_series]
                outlier_series.plot(kind='barh', ax=ax2, color=colors2)
                ax2.set_xlabel('Outlier %')
                ax2.set_title('Outliers per Numeric Feature (IQR)', fontweight='bold')
                for i, val in enumerate(outlier_series):
                    ax2.text(val + 0.1, i, f'{val:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        return fig



    # VISUALIZATION 3: CORRELATION MATRIX
    def plot_correlation_matrix(self, X: pd.DataFrame):
        """
        Plot correlation heatmap for numeric features.
        - ≤15 features: heatmap with values inside boxes
        - >15 features: heatmap (color only) + separate sorted table of all pairs
        Returns: fig (always), fig_table (only when >15 features, else None)
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No numeric features to correlate',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig, None

        corr_matrix = X[numeric_cols].corr()
        many_features = len(numeric_cols) > 15

        # --- Heatmap ---
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=not many_features,  # hide values if >15
            fmt='.2f',
            cmap='coolwarm',
            vmin=-1, vmax=1,
            center=0,
            square=True,
            linewidths=0.5 if not many_features else 0.2,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            ax=ax
        )
        ax.set_title('Feature Correlation Matrix',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()

        if not many_features:
            return fig, None

        # --- Top pairs bar chart (only when >15) ---
        pairs = (
                corr_matrix
                .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                .stack()
                .reset_index()
        )
        pairs.columns = ['Feature A', 'Feature B', 'Correlation']
        pairs['Abs'] = pairs['Correlation'].abs()
        pairs = pairs.sort_values('Abs', ascending=False).drop(columns='Abs').reset_index(drop=True)
        pairs['Correlation'] = pairs['Correlation'].round(4)

        top20 = pairs.head(20).copy()
        top20['Pair'] = top20['Feature A'] + '  ↔  ' + top20['Feature B']

        fig_table, ax2 = plt.subplots(figsize=(10, 8))
        colors = ['#ef4444' if abs(v) >= 0.7 else '#f59e0b' if abs(v) >= 0.4 else '#3b82f6'
                      for v in top20['Correlation']]
        bars = ax2.barh(top20['Pair'], top20['Correlation'].abs(), color=colors, edgecolor='none', height=0.6)

        # Value labels
        for bar, val in zip(bars, top20['Correlation']):
            ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{val:+.3f}', va='center', fontsize=9)

        # Multicollinearity warning
        high_corr_count = (pairs['Correlation'].abs() >= 0.85).sum()
        if high_corr_count > 0:
             ax2.set_title(
                f'Top 20 Correlated Feature Pairs\n'
                f'Top 20 Correlated Feature Pairs  |  WARNING: {high_corr_count} pairs with |r| >= 0.85 — potential multicollinearity',
                fontsize=13, fontweight='bold', color='#111'
            )
        else:
           ax2.set_title('Top 20 Correlated Feature Pairs', fontsize=13, fontweight='bold')

           ax2.set_xlabel('|Correlation|')
           ax2.set_xlim(0, 1.15)
           ax2.invert_yaxis()
           ax2.axvline(0.85, color='#ef4444', linestyle='--', linewidth=1, alpha=0.6, label='Multicollinearity (0.85)')
           ax2.axvline(0.4, color='#f59e0b', linestyle='--', linewidth=1, alpha=0.6, label='Moderate (0.4)')

        legend_elements = [
        Patch(facecolor='#ef4444', label='Strong ≥ 0.7'),
        Patch(facecolor='#f59e0b', label='Moderate ≥ 0.4'),
        Patch(facecolor='#3b82f6', label='Weak < 0.4'),
        ]
        ax2.legend(handles=legend_elements, fontsize=9, loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        return fig, fig_table




    # VISUALIZATION 4: TARGET ANALYSIS DASHBOARD
    def plot_target_analysis(self, X: pd.DataFrame, y: pd.Series,
                             target_name: str, task_type: str) -> plt.Figure:
        """
        Comprehensive target analysis dashboard.
        For Classification:
        - Class distribution (pie chart)
        - Class balance (bar chart)
        - Top features by mutual information

        For Regression:
        - Target distribution (histogram + KDE)
        - Target statistics
        - Top correlated features
        """
        if task_type == 'classification':
            return self._plot_classification_target_analysis(X, y, target_name)
        else:
            return self._plot_regression_target_analysis(X, y, target_name)

    def _plot_classification_target_analysis(self, X: pd.DataFrame,
                                             y: pd.Series, target_name: str) -> plt.Figure:
        """Target analysis for classification tasks"""

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        fig.suptitle(f'Target Analysis: {target_name} (Classification)',
                     fontsize=16, fontweight='bold')

        # 1. Class Distribution (Pie Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        value_counts = y.value_counts()
        colors = plt.cm.Set3(range(len(value_counts)))
        ax1.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors)
        ax1.set_title('Class Distribution', fontweight='bold')

        # 2. Class Balance (Bar Chart)
        ax2 = fig.add_subplot(gs[0, 1])
        value_counts.plot(kind='bar', ax=ax2, color=colors, edgecolor='black')
        ax2.set_title('Class Counts', fontweight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)

        # Add count labels
        for i, count in enumerate(value_counts):
            ax2.text(i, count, f'{count}', ha='center', va='bottom', fontweight='bold')

        # 3. Class Statistics
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        ax3.set_title('Class Statistics', fontweight='bold', pad=10)

        n_classes = len(value_counts)
        total = len(y)
        majority_class = value_counts.index[0]
        majority_pct = (value_counts.iloc[0] / total) * 100
        minority_pct = (value_counts.iloc[-1] / total) * 100

        stats_text = (
            f"Total Samples: {total}\n"
            f"Number of Classes: {n_classes}\n\n"
            f"Majority Class: {majority_class}\n"
            f"Majority %: {majority_pct:.1f}%\n\n"
            f"Minority %: {minority_pct:.1f}%\n\n"
            f"Balance Ratio: {value_counts.iloc[0] / value_counts.iloc[-1]:.2f}:1"
        )

        ax3.text(
            0.5, 0.65,
            stats_text,
            fontsize=11,
            va='center',
            ha='center',
            family='monospace',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='#E3F2FD',
                edgecolor='#90CAF9',
                alpha=0.9
            )
        )

        # 4. Top Features by Mutual Information
        ax4 = fig.add_subplot(gs[1, :])

        try:
            # Calculate mutual information (only for numeric features)
            X_numeric = X.select_dtypes(include=[np.number])

            if len(X_numeric.columns) > 0:
                # Handle missing values
                X_clean = X_numeric.fillna(X_numeric.median())
                y_clean = y.dropna()

                # Align indices
                common_idx = X_clean.index.intersection(y_clean.index)
                X_clean = X_clean.loc[common_idx]
                y_clean = y_clean.loc[common_idx]

                # Calculate mutual information
                mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)
                mi_scores = pd.Series(mi_scores, index=X_numeric.columns).sort_values(ascending=False)

                # Plot top 10
                top_n = min(10, len(mi_scores))
                mi_scores.head(top_n).plot(kind='barh', ax=ax4, color='teal', edgecolor='black')
                ax4.set_title(
                    f'Top {top_n} Most Informative Features for Predicting Target\n' +
                    '(Higher Mutual Information Score = More Predictive Power)',
                    fontweight='bold', fontsize=12)
                ax4.set_xlabel('Mutual Information Score')
                ax4.set_ylabel('Feature')

                # Add scores as text
                for i, score in enumerate(mi_scores.head(top_n)):
                    ax4.text(score, i, f' {score:.3f}', va='center', fontsize=9)
            else:
                ax4.text(0.5, 0.5, 'No numeric features for mutual information analysis',
                         ha='center', va='center', fontsize=12)
                ax4.axis('off')
        except Exception as e:
            ax4.text(0.5, 0.5, f'Could not calculate mutual information:\n{str(e)}',
                     ha='center', va='center', fontsize=10)
            ax4.axis('off')

        return fig

    def _plot_regression_target_analysis(self, X: pd.DataFrame,
                                         y: pd.Series, target_name: str) -> plt.Figure:
        """Target analysis for regression tasks"""

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        fig.suptitle(f'Target Analysis: {target_name} (Regression)',
                     fontsize=16, fontweight='bold')

        # Remove nulls
        y_clean = y.dropna()

        # 1. Target Distribution (Histogram + KDE)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(y_clean, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)

        # Add KDE
        try:
            y_clean.plot(kind='kde', ax=ax1, color='red', linewidth=2, label='KDE')
            ax1.legend()
        except:
            pass

        ax1.set_title('Target Distribution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Target Value')
        ax1.set_ylabel('Density')

        # Add mean and median lines
        mean_val = y_clean.mean()
        median_val = y_clean.median()
        ax1.axvline(mean_val, color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_val:.2f}')
        ax1.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {median_val:.2f}')
        ax1.legend()

        # 2. Target Statistics
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')

        stats_text = f"""
        Count: {len(y_clean)}

        Mean: {y_clean.mean():.2f}
        Median: {y_clean.median():.2f}
        Std Dev: {y_clean.std():.2f}

        Min: {y_clean.min():.2f}
        25%: {y_clean.quantile(0.25):.2f}
        75%: {y_clean.quantile(0.75):.2f}
        Max: {y_clean.max():.2f}

        Skewness: {y_clean.skew():.2f}
        Kurtosis: {y_clean.kurtosis():.2f}
        """

        ax2.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                 family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax2.set_title('Target Statistics', fontweight='bold')

        # 3. Top Correlated Features
        ax3 = fig.add_subplot(gs[1, :])

        try:
            # Get numeric features
            X_numeric = X.select_dtypes(include=[np.number])

            if len(X_numeric.columns) > 0:
                # Calculate correlations with target
                correlations = X_numeric.corrwith(y).abs().sort_values(ascending=False)
                correlations = correlations.dropna()

                # Plot top 10
                top_n = min(10, len(correlations))
                correlations.head(top_n).plot(kind='barh', ax=ax3, color='purple', edgecolor='black')
                ax3.set_title(f'Top {top_n} Features by Absolute Correlation with Target', fontweight='bold')
                ax3.set_xlabel('Absolute Correlation')
                ax3.set_ylabel('Feature')

                # Add correlation values
                for i, corr in enumerate(correlations.head(top_n)):
                    ax3.text(corr, i, f' {corr:.3f}', va='center', fontsize=9)
            else:
                ax3.text(0.5, 0.5, 'No numeric features for correlation analysis',
                         ha='center', va='center', fontsize=12)
                ax3.axis('off')
        except Exception as e:
            ax3.text(0.5, 0.5, f'Could not calculate correlations:\n{str(e)}',
                     ha='center', va='center', fontsize=10)
            ax3.axis('off')

        return fig


# Convenience function
def generate_eda(df: pd.DataFrame, target_column: str,
                 task_type: str = 'classification') -> Dict[str, plt.Figure]:
    """
    Quick function to generate all EDA plots.
    """

    generator = EDAGenerator()
    return generator.generate_all(df, target_column, task_type)