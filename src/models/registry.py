"""
Model Registry - Dynamic Model Selection and Management System

This is the BRAIN of the AutoML system's modeling layer. It:
1. Dynamically selects appropriate models based on task type
2. Manages model instantiation with proper configurations
3. Provides model metadata and capabilities
4. Handles model validation and compatibility checks

It is a "Model Factory" that knows:
- Which models work for which tasks
- How to initialize each model
- What each model is good at
- Default hyperparameters for each model

"""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from Config.config import Config

# Scikit-learn models
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

# Advanced models (XGBoost, LightGBM, CatBoost)
try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False



logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """
    Metadata container for each model.

    This stores everything we need to know about a model:
    - What it's called (display name)
    - The actual Python class
    - Default hyperparameters
    - What it's good at (strengths)
    - When to avoid it (weaknesses)
    - Time complexity (fast/medium/slow)
    """
    name: str  # Display name (e.g., "Random Forest")
    model_class: Any  # The actual sklearn/xgboost/etc class
    default_params: Dict  # Default hyperparameters
    task_types: List[str]  # ['classification'] or ['regression'] or both
    strengths: List[str]  # What this model excels at
    weaknesses: List[str]  # When to be cautious
    complexity: str  # 'fast', 'medium', 'slow'
    requires_scaling: bool  # Does it need feature scaling?
    handles_missing: bool  # Can it handle NaN values natively?
    handles_categorical: bool  # Can it handle categorical features natively?
    interpretable: bool  # Is it easy to interpret?


class ModelRegistry:
    """
    🧠 THE MODEL BRAIN - Manages all available models for AutoML.

    This class is the central hub for model management. It knows:
    - ALL available models (classification + regression)
    - How to initialize each model with proper params
    - Which models are suitable for which tasks
    - Model characteristics (speed, interpretability, etc.)

    Why We Need This:
    ----------------
    In traditional ML, we hardcode: model = RandomForestClassifier(...)
    In AutoML, we need DYNAMIC model selection based on:
    - Task type (classification vs regression)
    - Data characteristics (size, features, etc.)
    - User preferences (fast models vs accurate models)
    - Available libraries (XGBoost installed or not?)

    This registry makes it ALL dynamic and manageable!

    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.random_state = self.config.RANDOM_STATE
        self.n_jobs = self.config.N_JOBS

        # Build the registry
        self._classification_models = self._build_classification_registry()
        self._regression_models = self._build_regression_registry()

        logger.info(
            f"Model Registry initialized: "
            f"{len(self._classification_models)} classification models, "
            f"{len(self._regression_models)} regression models"
        )

    def _build_classification_registry(self) -> Dict[str, ModelInfo]:
        """
        Build registry of ALL classification models.

        Returns:
            Dictionary mapping model names to ModelInfo objects
        """
        registry = {}

        # ============================================================
        # 1. LOGISTIC REGRESSION - The Classic Linear Classifier
        # ============================================================
        registry['Logistic Regression'] = ModelInfo(
            name='Logistic Regression',
            model_class=LogisticRegression,
            default_params={
                'random_state': self.random_state,
                'max_iter': 1000
            },
            task_types=['classification'],
            strengths=[
                'Very fast training and prediction',
                'Highly interpretable (feature coefficients)',
                'Works well with linearly separable data',
                'Low memory footprint',
                'Probabilistic outputs'
            ],
            weaknesses=[
                'Assumes linear decision boundary',
                'Poor with complex non-linear patterns',
                'Requires feature scaling'
            ],
            complexity='fast',
            requires_scaling=True,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 2. RANDOM FOREST - The Reliable Ensemble
        # ============================================================
        registry['Random Forest'] = ModelInfo(
            name='Random Forest',
            model_class=RandomForestClassifier,
            default_params={
                'n_estimators': 100,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 4
            },
            task_types=['classification'],
            strengths=[
                'Excellent out-of-the-box performance',
                'Handles non-linear relationships well',
                'Resistant to overfitting (with enough trees)',
                'Feature importance built-in',
                'No scaling required',
                'Handles mixed data types well'
            ],
            weaknesses=[
                'Can be slow with large datasets',
                'Large memory footprint',
                'Less interpretable than single trees',
                'Can overfit on noisy data'
            ],
            complexity='medium',
            requires_scaling=False,
            handles_missing=False,
            handles_categorical=False,
            interpretable=False
        )

        # ============================================================
        # 3. GRADIENT BOOSTING - The Powerful Sequential Learner
        # ============================================================
        registry['Gradient Boosting'] = ModelInfo(
            name='Gradient Boosting',
            model_class=GradientBoostingClassifier,
            default_params={
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': self.random_state
            },
            task_types=['classification'],
            strengths=[
                'Often achieves best accuracy',
                'Handles complex patterns well',
                'Feature importance available',
                'Good with imbalanced data'
            ],
            weaknesses=[
                'Slow training (sequential)',
                'Prone to overfitting if not tuned',
                'Requires careful hyperparameter tuning',
                'Sensitive to outliers'
            ],
            complexity='slow',
            requires_scaling=False,
            handles_missing=False,
            handles_categorical=False,
            interpretable=False
        )

        # ============================================================
        # 4. DECISION TREE - The Simple Interpretable Model
        # ============================================================
        registry['Decision Tree'] = ModelInfo(
            name='Decision Tree',
            model_class=DecisionTreeClassifier,
            default_params={
                'random_state': self.random_state,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            task_types=['classification'],
            strengths=[
                'Extremely interpretable (visualize tree)',
                'Very fast training and prediction',
                'Handles non-linear data',
                'No scaling needed',
                'Works with mixed data types'
            ],
            weaknesses=[
                'Prone to overfitting',
                'Unstable (small data changes = big tree changes)',
                'Biased toward dominant classes',
                'Not the most accurate'
            ],
            complexity='fast',
            requires_scaling=False,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 5. K-NEAREST NEIGHBORS - The Instance-Based Learner
        # ============================================================
        registry['K-Nearest Neighbors'] = ModelInfo(
            name='K-Nearest Neighbors',
            model_class=KNeighborsClassifier,
            default_params={
                'n_neighbors': 5,
                'n_jobs': self.n_jobs
            },
            task_types=['classification'],
            strengths=[
                'Simple and intuitive',
                'No training required (lazy learning)',
                'Naturally handles multi-class',
                'Works well with low-dimensional data'
            ],
            weaknesses=[
                'Very slow prediction with large datasets',
                'Requires feature scaling',
                'Sensitive to irrelevant features',
                'Poor with high-dimensional data (curse of dimensionality)',
                'Sensitive to class imbalance'
            ],
            complexity='medium',
            requires_scaling=True,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 6. SUPPORT VECTOR MACHINE - The Margin Maximizer
        # ============================================================
        registry['SVM'] = ModelInfo(
            name='SVM',
            model_class=SVC,
            default_params={
                'random_state': self.random_state,
                'kernel': 'rbf',
                'probability': True  # Enable probability estimates
            },
            task_types=['classification'],
            strengths=[
                'Effective in high-dimensional spaces',
                'Memory efficient (uses support vectors)',
                'Versatile (different kernel functions)',
                'Good with clear margin of separation'
            ],
            weaknesses=[
                'Very slow with large datasets (O(n²) to O(n³))',
                'Requires careful hyperparameter tuning',
                'Requires feature scaling',
                'Doesn\'t provide probability estimates by default',
                'Sensitive to class imbalance'
            ],
            complexity='slow',
            requires_scaling=True,
            handles_missing=False,
            handles_categorical=False,
            interpretable=False
        )

        # ============================================================
        # 7. NAIVE BAYES - The Probabilistic Speedster
        # ============================================================
        registry['Naive Bayes'] = ModelInfo(
            name='Naive Bayes',
            model_class=GaussianNB,
            default_params={},
            task_types=['classification'],
            strengths=[
                'Extremely fast training and prediction',
                'Works well with small datasets',
                'Naturally handles multi-class',
                'Probabilistic predictions',
                'Low memory usage'
            ],
            weaknesses=[
                'Assumes feature independence (rarely true)',
                'Lower accuracy than complex models',
                'Sensitive to feature distributions'
            ],
            complexity='fast',
            requires_scaling=False,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 8. AdaBoost - The Adaptive Booster
        # ============================================================
        registry['AdaBoost'] = ModelInfo(
            name='AdaBoost',
            model_class=AdaBoostClassifier,
            default_params={
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': self.random_state
            },
            task_types=['classification'],
            strengths=[
                'Good accuracy with proper tuning',
                'Less prone to overfitting than single trees',
                'Feature importance available',
                'Works well with weak learners'
            ],
            weaknesses=[
                'Sensitive to noisy data and outliers',
                'Can be slow with large datasets',
                'Requires careful tuning'
            ],
            complexity='medium',
            requires_scaling=False,
            handles_missing=False,
            handles_categorical=False,
            interpretable=False
        )

        # ============================================================
        # 9. XGBoost - The Competition Winner (if available)
        # ============================================================
        if XGBOOST_AVAILABLE:
            registry['XGBoost'] = ModelInfo(
                name='XGBoost',
                model_class=XGBClassifier,
                default_params={
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                    'eval_metric': 'logloss'
                },
                task_types=['classification'],
                strengths=[
                    'Often wins Kaggle competitions',
                    'Very fast (optimized C++ backend)',
                    'Handles missing values natively',
                    'Built-in regularization',
                    'Feature importance',
                    'Excellent accuracy'
                ],
                weaknesses=[
                    'Many hyperparameters to tune',
                    'Can overfit with small datasets',
                    'Less interpretable'
                ],
                complexity='medium',
                requires_scaling=False,
                handles_missing=True,
                handles_categorical=False,
                interpretable=False
            )

        # ============================================================
        # 10. LightGBM - The Fast Gradient Booster (if available)
        # ============================================================
        if LIGHTGBM_AVAILABLE:
            registry['LightGBM'] = ModelInfo(
                name='LightGBM',
                model_class=LGBMClassifier,
                default_params={
                    'n_estimators': 100,
                    'learning_rate': 0.05,  # Smaller learning rate
                    'num_leaves': 31,  # Limit leaves to prevent overfitting
                    'max_depth': 7,
                    'min_child_samples': 20,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,  # L1 regularization
                    'reg_lambda': 0.1,  # L2 regularization
                    'verbose': -1
                },
                task_types=['classification'],
                strengths=[
                    'Extremely fast training',
                    'Low memory usage',
                    'Handles large datasets well',
                    'Handles categorical features natively',
                    'Excellent accuracy',
                    'Handles missing values'
                ],
                weaknesses=[
                    'Can overfit on small datasets',
                    'Sensitive to hyperparameters',
                    'May require more tuning'
                ],
                complexity='fast',
                requires_scaling=False,
                handles_missing=True,
                handles_categorical=True,
                interpretable=False
            )

        # ============================================================
        # 11. CatBoost - The Categorical Specialist (if available)
        # ============================================================
        if CATBOOST_AVAILABLE:
            registry['CatBoost'] = ModelInfo(
                name='CatBoost',
                model_class=CatBoostClassifier,
                default_params={
                    'iterations': 100,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'random_state': self.random_state,
                    'verbose': False
                },
                task_types=['classification'],
                strengths=[
                    'Best-in-class categorical handling',
                    'Excellent out-of-the-box performance',
                    'Robust to overfitting',
                    'Handles missing values',
                    'Less hyperparameter tuning needed'
                ],
                weaknesses=[
                    'Slower than LightGBM',
                    'Larger memory footprint',
                    'Fewer community resources'
                ],
                complexity='medium',
                requires_scaling=False,
                handles_missing=True,
                handles_categorical=True,
                interpretable=False
            )

        return registry

    def _build_regression_registry(self) -> Dict[str, ModelInfo]:
        """
        Build registry of ALL regression models.

        Returns:
            Dictionary mapping model names to ModelInfo objects
        """
        registry = {}

        # ============================================================
        # 1. LINEAR REGRESSION - The Classic Baseline
        # ============================================================
        registry['Linear Regression'] = ModelInfo(
            name='Linear Regression',
            model_class=LinearRegression,
            default_params={
            },
            task_types=['regression'],
            strengths=[
                'Extremely fast',
                'Highly interpretable (coefficients)',
                'Works well for linear relationships',
                'Low memory usage',
                'No hyperparameters to tune'
            ],
            weaknesses=[
                'Assumes linear relationship',
                'Sensitive to outliers',
                'Poor with complex non-linear patterns',
                'Can overfit with many features'
            ],
            complexity='fast',
            requires_scaling=True,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 2. RIDGE REGRESSION - L2 Regularized Linear
        # ============================================================
        registry['Ridge'] = ModelInfo(
            name='Ridge',
            model_class=Ridge,
            default_params={
                'alpha': 1.0,
                'random_state': self.random_state
            },
            task_types=['regression'],
            strengths=[
                'Reduces overfitting (L2 regularization)',
                'Fast training',
                'Interpretable',
                'Handles multicollinearity well',
                'Works well with many features'
            ],
            weaknesses=[
                'Assumes linear relationship',
                'Doesn\'t perform feature selection',
                'Requires scaling',
                'Poor with highly non-linear data'
            ],
            complexity='fast',
            requires_scaling=True,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 3. LASSO - L1 Regularized (Feature Selection)
        # ============================================================
        registry['Lasso'] = ModelInfo(
            name='Lasso',
            model_class=Lasso,
            default_params={
                'alpha': 1.0,
                'random_state': self.random_state
            },
            task_types=['regression'],
            strengths=[
                'Automatic feature selection (L1)',
                'Reduces overfitting',
                'Interpretable',
                'Fast training',
                'Good with sparse features'
            ],
            weaknesses=[
                'Assumes linear relationship',
                'Can be unstable with correlated features',
                'Requires scaling'
            ],
            complexity='fast',
            requires_scaling=True,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 4. ELASTIC NET - Best of Ridge + Lasso
        # ============================================================
        registry['Elastic Net'] = ModelInfo(
            name='Elastic Net',
            model_class=ElasticNet,
            default_params={
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'random_state': self.random_state
            },
            task_types=['regression'],
            strengths=[
                'Combines L1 and L2 regularization',
                'Feature selection + handles multicollinearity',
                'More stable than Lasso',
                'Fast training'
            ],
            weaknesses=[
                'Two hyperparameters to tune (alpha, l1_ratio)',
                'Assumes linear relationship',
                'Requires scaling'
            ],
            complexity='fast',
            requires_scaling=True,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 5. DECISION TREE REGRESSOR
        # ============================================================
        registry['Decision Tree'] = ModelInfo(
            name='Decision Tree',
            model_class=DecisionTreeRegressor,
            default_params={
                'random_state': self.random_state,
                'max_depth': 10,
                'min_samples_split': 2
            },
            task_types=['regression'],
            strengths=[
                'Highly interpretable',
                'Fast training',
                'Handles non-linear relationships',
                'No scaling needed'
            ],
            weaknesses=[
                'Prone to overfitting',
                'Unstable',
                'Not the most accurate'
            ],
            complexity='fast',
            requires_scaling=False,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 6. RANDOM FOREST REGRESSOR
        # ============================================================
        registry['Random Forest'] = ModelInfo(
            name='Random Forest',
            model_class=RandomForestRegressor,
            default_params={
                'n_estimators': 100,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'max_depth': None
            },
            task_types=['regression'],
            strengths=[
                'Excellent out-of-the-box performance',
                'Handles non-linear data',
                'Feature importance',
                'Resistant to overfitting',
                'No scaling needed'
            ],
            weaknesses=[
                'Can be slow with large data',
                'Large memory usage',
                'Less interpretable'
            ],
            complexity='medium',
            requires_scaling=False,
            handles_missing=False,
            handles_categorical=False,
            interpretable=False
        )

        # ============================================================
        # 7. GRADIENT BOOSTING REGRESSOR
        # ============================================================
        registry['Gradient Boosting'] = ModelInfo(
            name='Gradient Boosting',
            model_class=GradientBoostingRegressor,
            default_params={
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': self.random_state
            },
            task_types=['regression'],
            strengths=[
                'Often achieves best accuracy',
                'Handles complex patterns',
                'Feature importance',
                'Robust to outliers (with proper loss)'
            ],
            weaknesses=[
                'Slow training',
                'Prone to overfitting',
                'Requires tuning'
            ],
            complexity='slow',
            requires_scaling=False,
            handles_missing=False,
            handles_categorical=False,
            interpretable=False
        )

        # ============================================================
        # 8. K-NEAREST NEIGHBORS REGRESSOR
        # ============================================================
        registry['K-Nearest Neighbors'] = ModelInfo(
            name='K-Nearest Neighbors',
            model_class=KNeighborsRegressor,
            default_params={
                'n_neighbors': 5,
                'n_jobs': self.n_jobs
            },
            task_types=['regression'],
            strengths=[
                'Simple and intuitive',
                'No training required',
                'Handles non-linear data'
            ],
            weaknesses=[
                'Slow prediction with large datasets',
                'Requires scaling',
                'Poor with high dimensions',
                'Sensitive to outliers'
            ],
            complexity='medium',
            requires_scaling=True,
            handles_missing=False,
            handles_categorical=False,
            interpretable=True
        )

        # ============================================================
        # 9. SUPPORT VECTOR REGRESSOR
        # ============================================================
        registry['SVR'] = ModelInfo(
            name='SVR',
            model_class=SVR,
            default_params={
                'kernel': 'rbf'
            },
            task_types=['regression'],
            strengths=[
                'Effective in high dimensions',
                'Memory efficient',
                'Versatile (different kernels)'
            ],
            weaknesses=[
                'Very slow with large datasets',
                'Requires scaling',
                'Many hyperparameters',
                'Sensitive to outliers'
            ],
            complexity='slow',
            requires_scaling=True,
            handles_missing=False,
            handles_categorical=False,
            interpretable=False
        )

        # ============================================================
        # 10. AdaBoost REGRESSOR
        # ============================================================
        registry['AdaBoost'] = ModelInfo(
            name='AdaBoost',
            model_class=AdaBoostRegressor,
            default_params={
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': self.random_state
            },
            task_types=['regression'],
            strengths=[
                'Good accuracy',
                'Less prone to overfitting',
                'Works with weak learners'
            ],
            weaknesses=[
                'Sensitive to outliers',
                'Can be slow',
                'Requires tuning'
            ],
            complexity='medium',
            requires_scaling=False,
            handles_missing=False,
            handles_categorical=False,
            interpretable=False
        )

        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            registry['XGBoost'] = ModelInfo(
                name='XGBoost',
                model_class=XGBRegressor,
                default_params={
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs
                },
                task_types=['regression'],
                strengths=[
                    'Excellent accuracy',
                    'Fast training',
                    'Handles missing values',
                    'Built-in regularization'
                ],
                weaknesses=[
                    'Many hyperparameters',
                    'Can overfit on small data'
                ],
                complexity='medium',
                requires_scaling=False,
                handles_missing=True,
                handles_categorical=False,
                interpretable=False
            )

        if LIGHTGBM_AVAILABLE:
            registry['LightGBM'] = ModelInfo(
                name='LightGBM',
                model_class=LGBMRegressor,
                default_params={
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'random_state': self.random_state,
                    'n_jobs': self.n_jobs,
                    'verbose': -1
                },
                task_types=['regression'],
                strengths=[
                    'Very fast',
                    'Low memory',
                    'Handles large datasets',
                    'Excellent accuracy'
                ],
                weaknesses=[
                    'Can overfit on small data',
                    'Requires tuning'
                ],
                complexity='fast',
                requires_scaling=False,
                handles_missing=True,
                handles_categorical=True,
                interpretable=False
            )

        if CATBOOST_AVAILABLE:
            registry['CatBoost'] = ModelInfo(
                name='CatBoost',
                model_class=CatBoostRegressor,
                default_params={
                    'iterations': 100,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'random_state': self.random_state,
                    'verbose': False
                },
                task_types=['regression'],
                strengths=[
                    'Excellent categorical handling',
                    'Great out-of-box performance',
                    'Handles missing values',
                    'Less tuning needed'
                ],
                weaknesses=[
                    'Slower than LightGBM',
                    'Higher memory usage'
                ],
                complexity='medium',
                requires_scaling=False,
                handles_missing=True,
                handles_categorical=True,
                interpretable=False
            )

        return registry

    # ============================================================
    # PUBLIC API METHODS
    # ============================================================

    def get_models_for_task(self, task_type: str) -> List[str]:
        """
        Get list of all available model names for a task type.

        Returns:
            List of model names

        """
        if task_type == 'classification':
            return list(self._classification_models.keys())
        elif task_type == 'regression':
            return list(self._regression_models.keys())
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def get_model(self, model_name: str, task_type: str, custom_params: Dict = None):
        """
        Initialize and return a model instance.

        Returns:
            Initialized sklearn/xgboost/lightgbm model instance

        """
        # Get the appropriate registry
        if task_type == 'classification':
            registry = self._classification_models
        elif task_type == 'regression':
            registry = self._regression_models
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Check if model exists
        if model_name not in registry:
            available = list(registry.keys())
            raise ValueError(
                f"Model '{model_name}' not found for {task_type}. "
                f"Available models: {available}"
            )

        # Get model info
        model_info = registry[model_name]

        # Merge default params with custom params
        params = model_info.default_params.copy()
        if custom_params:
            params.update(custom_params)

        # Initialize model
        model = model_info.model_class(**params)

        logger.debug(f"Initialized {model_name} with params: {params}")
        return model

    def get_model_info(self, model_name: str, task_type: str = None) -> ModelInfo:
        """
        Get metadata about a model.


        Returns:
            ModelInfo object with model metadata
        """
        # Search in both registries if task_type not specified
        if task_type == 'classification' or task_type is None:
            if model_name in self._classification_models:
                return self._classification_models[model_name]

        if task_type == 'regression' or task_type is None:
            if model_name in self._regression_models:
                return self._regression_models[model_name]

        raise ValueError(f"Model '{model_name}' not found")

    def get_fast_models(self, task_type: str) -> List[str]:
        """Get list of fast models for quick experimentation."""
        registry = (self._classification_models if task_type == 'classification'
                    else self._regression_models)
        return [name for name, info in registry.items() if info.complexity == 'fast']

    def get_interpretable_models(self, task_type: str) -> List[str]:
        """Get list of interpretable models."""
        registry = (self._classification_models if task_type == 'classification'
                    else self._regression_models)
        return [name for name, info in registry.items() if info.interpretable]

    def get_models_handling_missing(self, task_type: str) -> List[str]:
        """Get models that can handle missing values natively."""
        registry = (self._classification_models if task_type == 'classification'
                    else self._regression_models)
        return [name for name, info in registry.items() if info.handles_missing]

    def get_default_models(self, task_type: str) -> List[str]:
        """
        Get recommended default models from config.

        Returns models defined in Config.DEFAULT_CLASSIFICATION_MODELS
        or Config.DEFAULT_REGRESSION_MODELS that are available.
        """
        if task_type == 'classification':
            default_names = self.config.DEFAULT_CLASSIFICATION_MODELS
            registry = self._classification_models
        else:
            default_names = self.config.DEFAULT_REGRESSION_MODELS
            registry = self._regression_models

        # Filter to only include available models
        return [name for name in default_names if name in registry]

    def print_summary(self, task_type: str = None):
        """
        Print a nice summary of available models.

        """

        def _print_registry(registry_name: str, registry: Dict):
            print(f"\n{'=' * 70}")
            print(f"  {registry_name.upper()} MODELS")
            print(f"{'=' * 70}\n")

            for name, info in registry.items():
                print(f"📊 {name}")
                print(f"   Complexity: {info.complexity.upper()}")
                print(f"   Interpretable: {'Yes' if info.interpretable else 'No'}")
                print(f"   Handles Missing: {'Yes' if info.handles_missing else 'No'}")
                print(f"   Strengths: {', '.join(info.strengths[:2])}...")
                print()

        if task_type == 'classification' or task_type is None:
            _print_registry("Classification", self._classification_models)

        if task_type == 'regression' or task_type is None:
            _print_registry("Regression", self._regression_models)