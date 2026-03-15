"""
Preprocessing Pipeline - Robust Data Preparation

Consolidated Preprocessing Logic:
- Automatic Type Detection (Numeric, Categorical, ID)
- ID Column Removal (prevents overfitting/scaling errors)
- High Cardinality -> Frequency Encoding
- Low Cardinality -> One-Hot Encoding
- Numeric -> Standard Scaling
- Missing Value Imputation

"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, List, Optional, Dict
import logging
from Config.config import Config

logger = logging.getLogger(__name__)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency encoder for high-cardinality categorical features.
    Replaces category with its frequency (count/total) in the dataset.
    """
    def __init__(self):
        self.freq_maps_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Learn frequency maps from training data"""
        for col in X.columns:
            # Calculate normalized frequency
            self.freq_maps_[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency maps"""
        X_out = X.copy()
        for col in X.columns:
            if col in self.freq_maps_:
                # Map values, fill unknowns with 0
                X_out[col] = X_out[col].map(self.freq_maps_[col]).fillna(0)
            else:
                # Fallback if column wasn't seen in fit
                X_out[col] = 0
        return X_out


class PreprocessingPipeline:
    """
    Unified Preprocessing Pipeline.
    Handles cleaning, encoding, and scaling in a single pass.
    """

    def __init__(self,
                 one_hot_threshold: int = 15,
                 id_threshold: float = 0.95, # If >95% unique, treat as ID and drop
                 config: Config = None):

        self.one_hot_threshold = one_hot_threshold
        self.id_threshold = id_threshold
        self.config = config or Config()

        # Components
        self.preprocessor = None
        self.freq_encoder = None

        # Feature tracking
        self.feature_names_in_ = []
        self.numeric_features_ = []
        self.categorical_features_ = []
        self.low_card_categorical_ = []
        self.high_card_categorical_ = []
        self.id_columns_ = []
        self.processed_feature_names_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Analyze data and fit transformers.
        """
        if y is None:
            raise ValueError("Target variable 'y' is required for preprocessing fit")

        logger.info("⚙️ Fitting unified preprocessing pipeline...")

        # Basic cleaning
        X = X.copy()
        X = self._clean_boolean_and_types(X)
        self.feature_names_in_ = list(X.columns)

        # Step 1: Identify feature types & DROP IDs
        self._identify_feature_types(X)

        # Step 2: Fit Frequency Encoder for high-cardinality features
        if self.high_card_categorical_:
            logger.info(f"   Fitting frequency encoder for {len(self.high_card_categorical_)} features...")
            self.freq_encoder = FrequencyEncoder()
            self.freq_encoder.fit(X[self.high_card_categorical_])

        # Step 3: Build sklearn pipeline
        self._build_preprocessor(X)

        # Store output feature names for summary
        self.processed_feature_names_ = self.get_feature_names_out()

        logger.info("✅ Preprocessing pipeline fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply transformations to new data.
        """
        X = X.copy()
        X = self._clean_boolean_and_types(X)

        # 1. Frequency Encoding (High Cardinality -> Numeric)
        X_encoded = X.copy()
        if self.high_card_categorical_ and self.freq_encoder:
            freq_data = self.freq_encoder.transform(X[self.high_card_categorical_])
            for col in self.high_card_categorical_:
                X_encoded[col] = freq_data[col]

        # 2. Main Preprocessing (Scaling + OneHot)
        # Note: High card columns are now numbers in X_encoded, handled by numeric pipeline
        X_processed = self.preprocessor.transform(X_encoded)

        return X_processed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def _clean_boolean_and_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """Helper to clean boolean and mixed types"""
        for col in X.columns:
            if X[col].dtype == 'bool' or X[col].apply(lambda x: isinstance(x, bool)).any():
                X[col] = X[col].astype(str)
                X[col] = X[col].apply(lambda x: str(x) if x is not None else 'missing')
        return X

    def _identify_feature_types(self, X: pd.DataFrame):
        """Identify types and DROP ID columns"""
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()

        # Categorical candidates
        all_categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()

        self.categorical_features_ = []
        self.low_card_categorical_ = []
        self.high_card_categorical_ = []
        self.id_columns_ = []

        n_samples = len(X)

        for col in all_categorical:
            n_unique = X[col].nunique()

            # CHECK FOR ID COLUMNS
            # If > 95% unique and string, it's likely an ID.
            if n_samples > 50 and (n_unique / n_samples) > self.id_threshold:
                self.id_columns_.append(col)
                continue

            self.categorical_features_.append(col)
            if n_unique <= self.one_hot_threshold:
                self.low_card_categorical_.append(col)
            else:
                self.high_card_categorical_.append(col)

        logger.info(f"   Feature Analysis:")
        logger.info(f"   - Numeric Features: {len(self.numeric_features_)}")
        logger.info(f"   - Dropped IDs: {len(self.id_columns_)}")
        logger.info(f"   - Low Card (One-Hot): {len(self.low_card_categorical_)}")
        logger.info(f"   - High Card (Freq Enc): {len(self.high_card_categorical_)}")

    def _build_preprocessor(self, X: pd.DataFrame):
        transformers = []

        # Numeric Pipeline
        # We process: Original Numeric + High Cardinality (now Frequency Encoded)
        numeric_cols = self.numeric_features_ + self.high_card_categorical_

        if numeric_cols:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numeric_pipeline, numeric_cols))

        # Categorical Pipeline (One-Hot)
        if self.low_card_categorical_:
            low_card_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', low_card_pipeline, self.low_card_categorical_))

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )

        # Fake fit to initialize structure (High card cols need to be numeric for this check)
        X_temp = X.copy()
        if self.high_card_categorical_:
            # Temporarily force them to numeric so ColumnTransformer accepts them in 'num' pipe
            for col in self.high_card_categorical_:
                X_temp[col] = 0.5

        self.preprocessor.fit(X_temp)

    def get_feature_names_out(self) -> List[str]:
        """Robustly get output feature names for logging/summary."""
        feature_names = []

        # 1. Numeric + Frequency Encoded Features
        numeric_cols = self.numeric_features_ + self.high_card_categorical_
        feature_names.extend(numeric_cols)

        # 2. One-Hot Encoded Features
        if self.low_card_categorical_ and hasattr(self.preprocessor, 'named_transformers_'):
            try:
                ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                ohe_features = ohe.get_feature_names_out(self.low_card_categorical_)
                feature_names.extend(list(ohe_features))
            except Exception as e:
                logger.warning(f"Could not get OHE names: {e}")
                # Fallback naming
                for col in self.low_card_categorical_:
                    feature_names.extend([f"{col}_cat_{i}" for i in range(2)])

        return feature_names