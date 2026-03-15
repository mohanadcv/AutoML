"""
Data Splitting - Split data into train/validation/test sets

Provides stratified splits for classification and random splits for regression.
Handles edge cases like small datasets and imbalanced classes.
"""
# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Union
import logging
from Config.config import Config

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Splits data into train/validation/test sets.

    Features:
    - Stratified splitting for classification
    - Random splitting for regression
    - Configurable split ratios
    - Reproducible with random seed
    - Handles small datasets gracefully

    """

    def __init__(self, config: Config = None):
        """
        Initialize data splitter.
        """
        self.config = config or Config()
        self.test_size = self.config.TEST_SIZE
        self.val_size = self.config.VAL_SIZE
        self.random_state = self.config.RANDOM_STATE

    def split(self, X: pd.DataFrame, y: pd.Series,
              task_type: str = 'classification') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/validation/test sets.

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

        Process:
            1. Split into train+val (80%) and test (20%)
            2. Split train+val into train (80%) and val (20%)

        Final sizes (default):
            - Train: 64% of data
            - Val: 16% of data
            - Test: 20% of data

        Example:
            >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y, 'classification')
            >>> print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        """
        logger.info(f"Splitting data: {len(X)} samples, task: {task_type}")

        # Determine if we should stratify
        use_stratify = self._should_stratify(y, task_type)

        # Step 1: Split into (train+val) and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if use_stratify else None
        )

        logger.info(f"Split 1: Train+Val={len(X_temp)}, Test={len(X_test)}")

        # Step 2: Split (train+val) into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp if use_stratify else None
        )

        logger.info(f"Split 2: Train={len(X_train)}, Val={len(X_val)}")

        # Log final distribution
        self._log_split_info(X_train, X_val, X_test, y_train, y_val, y_test, task_type)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _should_stratify(self, y: pd.Series, task_type: str) -> bool:
        """
        Determine if stratified splitting should be used.

        Stratify for classification if:
        - Task is classification
        - Enough samples per class (at least 2)
        - Not too many classes

        Returns:
            True if should stratify, False otherwise
        """
        # Only stratify for classification
        if task_type != 'classification':
            return False

        # Check class distribution
        value_counts = y.value_counts()
        n_classes = len(value_counts)
        min_samples_per_class = value_counts.min()

        # Need at least 2 samples per class for stratification
        if min_samples_per_class < 2:
            logger.warning(
                f"Cannot stratify: Class '{value_counts.idxmin()}' has only "
                f"{min_samples_per_class} sample(s). Using random split."
            )
            return False

        # Too many classes might cause issues
        if n_classes > 50:
            logger.warning(
                f"Too many classes ({n_classes}) for stratification. Using random split."
            )
            return False

        logger.info(f"Using stratified split for {n_classes} classes")
        return True

    def _log_split_info(self, X_train, X_val, X_test, y_train, y_val, y_test, task_type):
        """Log detailed information about the split"""

        total = len(X_train) + len(X_val) + len(X_test)

        logger.info("\n" + "=" * 70)
        logger.info("📊 DATA SPLIT SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total samples: {total}")
        logger.info(f"  • Training:   {len(X_train):5} samples ({len(X_train) / total * 100:5.1f}%)")
        logger.info(f"  • Validation: {len(X_val):5} samples ({len(X_val) / total * 100:5.1f}%)")
        logger.info(f"  • Test:       {len(X_test):5} samples ({len(X_test) / total * 100:5.1f}%)")

        if task_type == 'classification':
            # Show class distribution
            logger.info("\nClass distribution:")

            train_dist = y_train.value_counts(normalize=True).sort_index()
            val_dist = y_val.value_counts(normalize=True).sort_index()
            test_dist = y_test.value_counts(normalize=True).sort_index()

            for cls in train_dist.index:
                logger.info(
                    f"  Class {cls}: Train={train_dist.get(cls, 0) * 100:5.1f}%, "
                    f"Val={val_dist.get(cls, 0) * 100:5.1f}%, "
                    f"Test={test_dist.get(cls, 0) * 100:5.1f}%"
                )
        else:
            # Show target statistics for regression
            logger.info("\nTarget statistics:")
            logger.info(f"  • Train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
            logger.info(f"  • Val:   mean={y_val.mean():.2f}, std={y_val.std():.2f}")
            logger.info(f"  • Test:  mean={y_test.mean():.2f}, std={y_test.std():.2f}")

        logger.info("=" * 70)

    def get_split_info(self, X_train, X_val, X_test, y_train, y_val, y_test,
                       task_type: str = 'classification') -> dict:
        """
        Get split information as dictionary (for UI display).

        Returns:
            Dictionary with split statistics
        """
        total = len(X_train) + len(X_val) + len(X_test)

        info = {
            'total_samples': total,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'train_pct': (len(X_train) / total * 100),
            'val_pct': (len(X_val) / total * 100),
            'test_pct': (len(X_test) / total * 100),
            'n_features': X_train.shape[1]
        }

        if task_type == 'classification':
            info['train_class_dist'] = y_train.value_counts().to_dict()
            info['val_class_dist'] = y_val.value_counts().to_dict()
            info['test_class_dist'] = y_test.value_counts().to_dict()
        else:
            info['train_target_mean'] = float(y_train.mean())
            info['val_target_mean'] = float(y_val.mean())
            info['test_target_mean'] = float(y_test.mean())
            info['train_target_std'] = float(y_train.std())
            info['val_target_std'] = float(y_val.std())
            info['test_target_std'] = float(y_test.std())

        return info


# Convenience function
def split_data(X: pd.DataFrame, y: pd.Series,
               task_type: str = 'classification',
               config: Config = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
pd.Series, pd.Series, pd.Series]:
    """
    Convenience function to split data without creating splitter instance.


    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    splitter = DataSplitter(config)
    return splitter.split(X, y, task_type)
