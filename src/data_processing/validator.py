"""
Data Validator - Validates uploaded data before processing

Checks:
- File validity (size, type, readability)
- DataFrame validity (not empty, has columns, etc.)
- Target column validity (not all null, has variance, etc.)
- Data quality (null ratios, data types, etc.)
"""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple
import logging
from Config.config import Config

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data at multiple levels:
    1. File level - before loading
    2. DataFrame level - after loading
    3. Column level - specific columns
    4. Target level - ML target variable
    """

    def __init__(self, config: Config = None):
        """
        Initialize validator with configuration.
        """
        self.config = config or Config()
        self.max_file_size_bytes = self.config.MAX_FILE_SIZE_MB * 1_000_000
        self.allowed_extensions = self.config.ALLOWED_EXTENSIONS
        self.min_rows = self.config.MIN_ROWS
        self.min_columns = self.config.MIN_COLUMNS
        self.max_null_ratio = self.config.MAX_NULL_RATIO

    # ================================================================
    # FILE VALIDATION (Before Loading)
    # ================================================================

    def validate_file(self, file_path: Union[str, Path] = None,
                      uploaded_file=None) -> Tuple[bool, str]:
        """
        Validate file before loading.

        Returns:
            Tuple of (is_valid, message)
            - is_valid: True if valid, False otherwise
            - message: Error message if invalid, success message if valid
        """
        try:
            # Determine which type of file we're validating
            if file_path is not None:
                return self._validate_file_path(Path(file_path))
            elif uploaded_file is not None:
                return self._validate_uploaded_file(uploaded_file)
            else:
                return False, "No file provided for validation"

        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"Validation error: {str(e)}"

    def _validate_file_path(self, file_path: Path) -> Tuple[bool, str]:
        """Validate a file path"""

        # Check 1: File exists
        if not file_path.exists():
            return False, f"File not found: {file_path}"

        # Check 2: Is a file (not directory)
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"

        # Check 3: File extension
        extension = file_path.suffix.lower()
        if extension not in self.allowed_extensions:
            return False, f"Unsupported file type: {extension}. Allowed: {self.allowed_extensions}"

        # Check 4: File size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size_bytes:
            size_mb = file_size / 1_000_000
            max_mb = self.config.MAX_FILE_SIZE_MB
            return False, f"File too large: {size_mb:.1f}MB (max: {max_mb}MB)"

        # Check 5: File not empty
        if file_size == 0:
            return False, "File is empty (0 bytes)"

        logger.info(f"✅ File validation passed: {file_path.name}")
        return True, "File is valid"

    def _validate_uploaded_file(self, uploaded_file) -> Tuple[bool, str]:
        """Validate Streamlit uploaded file"""

        # Check 1: File exists
        if uploaded_file is None:
            return False, "No file uploaded"

        # Check 2: Has name
        if not hasattr(uploaded_file, 'name'):
            return False, "Invalid file object"

        # Check 3: File extension
        extension = Path(uploaded_file.name).suffix.lower()
        if extension not in self.allowed_extensions:
            return False, f"Unsupported file type: {extension}. Allowed: {self.allowed_extensions}"

        # Check 4: File size
        if hasattr(uploaded_file, 'size'):
            if uploaded_file.size > self.max_file_size_bytes:
                size_mb = uploaded_file.size / 1_000_000
                max_mb = self.config.MAX_FILE_SIZE_MB
                return False, f"File too large: {size_mb:.1f}MB (max: {max_mb}MB)"

            if uploaded_file.size == 0:
                return False, "File is empty (0 bytes)"

        logger.info(f"✅ Uploaded file validation passed: {uploaded_file.name}")
        return True, "File is valid"

    # ================================================================
    # DATAFRAME VALIDATION (After Loading)
    # ================================================================

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate loaded DataFrame.

        Returns:
            Tuple of (is_valid, message)

        Checks:
        - Not empty
        - Has minimum rows
        - Has minimum columns
        - No completely null columns
        - Has at least some numeric data
        """
        try:
            # Check 1: DataFrame exists and is not None
            if df is None:
                return False, "DataFrame is None"

            # Check 2: Not empty
            if df.empty:
                return False, "DataFrame is empty (no data)"

            # Check 3: Minimum rows
            if len(df) < self.min_rows:
                return False, f"Too few rows: {len(df)} (minimum: {self.min_rows})"

            # Check 4: Minimum columns
            if len(df.columns) < self.min_columns:
                return False, f"Too few columns: {len(df.columns)} (minimum: {self.min_columns})"

            # Check 5: No completely null columns
            null_columns = df.columns[df.isnull().all()].tolist()
            if null_columns:
                return False, f"Columns are entirely null: {null_columns}"

            # Check 6: Has at least one numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return False, "No numeric columns found. Dataset must have at least one numeric feature."

            # Check 7: Not all columns are the same
            if df.nunique().max() == 1:
                return False, "All columns contain only one unique value"

            logger.info(f"✅ DataFrame validation passed: {df.shape[0]} rows, {df.shape[1]} columns")
            return True, f"DataFrame is valid ({df.shape[0]} rows, {df.shape[1]} columns)"

        except Exception as e:
            logger.error(f"DataFrame validation error: {e}")
            return False, f"Validation error: {str(e)}"

    # ================================================================
    # COLUMN VALIDATION
    # ================================================================

    def validate_column_exists(self, df: pd.DataFrame, column_name: str) -> Tuple[bool, str]:
        """
        Check if a column exists in DataFrame.

        Returns:
            Tuple of (exists, message)
        """
        if column_name not in df.columns:
            available = list(df.columns)
            return False, f"Column '{column_name}' not found. Available columns: {available}"

        return True, f"Column '{column_name}' exists"

    def validate_columns_exist(self, df: pd.DataFrame, column_names: List[str]) -> Tuple[bool, str]:
        """
        Check if multiple columns exist.

        Returns:
            Tuple of (all_exist, message)
        """
        missing = [col for col in column_names if col not in df.columns]

        if missing:
            return False, f"Missing columns: {missing}"

        return True, f"All {len(column_names)} columns exist"

    # ================================================================
    # TARGET VALIDATION (ML Specific)
    # ================================================================

    def validate_target(self, y: pd.Series, target_name: str = "target") -> Tuple[bool, str]:
        """
        Validate ML target variable.

        Args:
            y: Target variable (pandas Series)
            target_name: Name of target (for error messages)

        Returns:
            Tuple of (is_valid, message)

        Checks:
        - Not all null
        - Has variance (not constant)
        - Acceptable null ratio
        - At least 2 unique values
        """
        try:
            # Check 1: Not None
            if y is None:
                return False, f"Target '{target_name}' is None"

            # Check 2: Not empty
            if len(y) == 0:
                return False, f"Target '{target_name}' is empty"

            # Check 3: Not all null
            if y.isnull().all():
                return False, f"Target '{target_name}' is entirely null"

            # Check 4: Null ratio acceptable
            null_ratio = y.isnull().sum() / len(y)
            if null_ratio > self.max_null_ratio:
                return False, f"Target '{target_name}' has too many missing values: {null_ratio * 100:.1f}% (max: {self.max_null_ratio * 100:.0f}%)"

            # Check 5: Has variance (not constant)
            non_null_values = y.dropna()
            if len(non_null_values.unique()) == 1:
                return False, f"Target '{target_name}' has only one unique value: {non_null_values.unique()[0]}"

            # Check 6: At least 2 unique values for ML
            n_unique = y.nunique()
            if n_unique < 2:
                return False, f"Target '{target_name}' must have at least 2 unique values (has {n_unique})"

            # Check 7: For classification - not too many classes
            # If target looks like classification (few unique values), check class balance
            if n_unique <= 50:  # Likely classification
                # Warn if severely imbalanced
                value_counts = y.value_counts()
                min_samples = value_counts.min()
                if min_samples < 3:
                    return False, f"Target has class with only {min_samples} sample(s). Each class needs at least 3 samples for proper train/val/test splitting."
                majority_ratio = value_counts.iloc[0] / len(y)
                if min_samples < 4:
                    logger.warning(f"Class with only {min_samples} samples - may not appear in all splits")

                if majority_ratio > 0.95:
                    logger.warning(f"Target is highly imbalanced: {majority_ratio * 100:.1f}% in majority class")
                    # Don't fail - just warn

            logger.info(f"✅ Target validation passed: {len(y)} samples, {n_unique} unique values")
            return True, f"Target is valid ({len(y)} samples, {n_unique} unique values)"

        except Exception as e:
            logger.error(f"Target validation error: {e}")
            return False, f"Validation error: {str(e)}"

    # ================================================================
    # DATA QUALITY CHECKS
    # ================================================================

    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Comprehensive data quality report.

        Returns dictionary with:
        - n_rows
        - n_columns
        - null_summary
        - dtype_summary
        - duplicate_rows
        - constant_columns
        - high_cardinality_columns
        """
        report = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1_000_000,
        }

        # Null analysis
        null_counts = df.isnull().sum()
        null_ratios = null_counts / len(df)
        report['null_summary'] = {
            'total_nulls': null_counts.sum(),
            'columns_with_nulls': (null_counts > 0).sum(),
            'max_null_ratio': null_ratios.max(),
            'columns_above_50pct_null': null_ratios[null_ratios > 0.5].index.tolist()
        }

        # Data type summary
        report['dtype_summary'] = df.dtypes.value_counts().to_dict()

        # Duplicate rows
        report['duplicate_rows'] = df.duplicated().sum()

        # Constant columns (only one unique value)
        constant_cols = df.columns[df.nunique() == 1].tolist()
        report['constant_columns'] = constant_cols

        # High cardinality columns (many unique values - might be IDs)
        high_card_threshold = len(df) * 0.9
        high_card_cols = df.columns[df.nunique() > high_card_threshold].tolist()
        report['high_cardinality_columns'] = high_card_cols

        return report

    def validate_feature_target_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[bool, str]:
        """
        Validate that features and target are compatible.

        Returns:
            Tuple of (is_valid, message)
        """
        # Check 1: Same number of rows
        if len(X) != len(y):
            return False, f"Features and target have different lengths: {len(X)} vs {len(y)}"

        # Check 2: Features has columns
        if len(X.columns) == 0:
            return False, "Feature matrix has no columns"

        # Check 3: No overlap (target shouldn't be in features)
        if hasattr(y, 'name') and y.name in X.columns:
            return False, f"Target '{y.name}' is still in feature matrix"

        return True, f"Feature-target split is valid ({len(X)} samples, {len(X.columns)} features)"


# Convenience function
def validate_data(df: pd.DataFrame, target_column: str = None, config: Config = None) -> Dict[str, any]:
    """
    Quick validation of DataFrame with optional target.

    Returns:
        Dictionary with validation results
    """
    validator = DataValidator(config)

    results = {
        'dataframe_valid': False,
        'target_valid': False,
        'quality_report': None
    }

    # Validate DataFrame
    df_valid, df_msg = validator.validate_dataframe(df)
    results['dataframe_valid'] = df_valid
    results['dataframe_message'] = df_msg

    # Validate target if provided
    if target_column:
        col_valid, col_msg = validator.validate_column_exists(df, target_column)
        if col_valid:
            target_valid, target_msg = validator.validate_target(df[target_column], target_column)
            results['target_valid'] = target_valid
            results['target_message'] = target_msg
        else:
            results['target_valid'] = False
            results['target_message'] = col_msg

    # Quality report
    if df_valid:
        results['quality_report'] = validator.check_data_quality(df)

    return results