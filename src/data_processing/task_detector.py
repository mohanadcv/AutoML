"""
Task Detector - Intelligently infer if ML task is classification or regression

Analyze the target variable and determines the appropriate
ML task type using multiple heuristics and confidence scoring.

"""

# Add project root to path
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from Config.config import Config

logger = logging.getLogger(__name__)


class TaskDetector:
    """
    Detects whether a machine learning task is classification or regression
    based on target variable characteristics.

    Uses multiple heuristics:
    1. Data type analysis
    2. Unique value counting
    3. Distribution analysis
    4. Pattern recognition

    Returns both prediction and confidence score.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.unique_ratio_threshold = self.config.UNIQUE_RATIO_THRESHOLD
        self.max_unique_for_classification = self.config.MAX_UNIQUE_FOR_CLASSIFICATION

    def detect(self, y: pd.Series) -> Tuple[str, float]:
        """
        Main detection method - analyzes target and returns task type.

        Returns:
            Tuple of (task_type, confidence_score)
            - task_type: 'classification' or 'regression'
            - confidence_score: float between 0 and 1
        """
        logger.info(f"Analyzing target variable: {len(y)} samples, {y.nunique()} unique values")

        # Run all detection heuristics
        rules = [
            self._check_data_type(y),
            self._check_binary(y),
            self._check_unique_count(y),
            self._check_unique_ratio(y),
            self._check_integer_pattern(y),
            self._check_continuous_distribution(y)
        ]

        # Filter out None results
        valid_rules = [r for r in rules if r is not None]

        if not valid_rules:
            logger.warning("No detection rules matched - defaulting to classification")
            return 'classification', 0.5

        # Weighted voting system
        task_type, confidence = self._aggregate_results(valid_rules)

        # SAFETY CHECK: If low confidence, reduce further
        if confidence < 0.70:
            logger.warning(f"Low confidence ({confidence:.2f}) - user confirmation recommended")
            # Don't change task_type, just flag it

        logger.info(f"Detection result: {task_type} (confidence: {confidence:.2f})")
        return task_type, confidence

    def _check_data_type(self, y: pd.Series) -> Tuple[str, float] | None:
        """
        Rule 1: Check obvious data types

        Logic:
        - String/object/category → Classification (high confidence)
        - Boolean → Classification (very high confidence)
        """
        dtype = y.dtype

        # String or categorical
        if dtype == 'object' or dtype.name == 'category':
            logger.debug("Rule: Object/Category dtype → Classification")
            return 'classification', 0.95

        # Boolean
        if dtype == 'bool':
            logger.debug("Rule: Boolean dtype → Classification")
            return 'classification', 0.99

        return None

    def _check_binary(self, y: pd.Series) -> Tuple[str, float] | None:
        """
        Rule 2: Check for binary variables

        Logic:
        - Only 2 unique values (like 0/1, True/False, Yes/No) → Classification
        """
        unique_values = set(y.dropna().unique())

        # Binary numeric (0, 1)
        if unique_values.issubset({0, 1}) or unique_values.issubset({0.0, 1.0}):
            logger.debug("Rule: Binary (0,1) → Classification")
            return 'classification', 0.99

        # Binary boolean
        if unique_values.issubset({True, False}):
            logger.debug("Rule: Binary (True,False) → Classification")
            return 'classification', 0.99

        # Any two values only
        if len(unique_values) == 2:
            logger.debug("Rule: Two unique values → Classification")
            return 'classification', 0.90

        return None

    def _check_unique_count(self, y: pd.Series) -> Tuple[str, float] | None:
        """
        Rule 3: Count unique values

        Logic:
        - Few unique values (< threshold) → Classification
        - Many unique values → Regression
        """
        n_unique = y.nunique()
        n_total = len(y.dropna())

        # Very few unique values
        if n_unique <= 10:
            if n_total > 20 and n_unique / n_total < 0.3:
               logger.debug(f"Rule: {n_unique} unique values (≤10) → Classification")
               return 'classification', 0.95

        # Moderate number of unique values
        if n_unique <= self.max_unique_for_classification:
            logger.debug(f"Rule: {n_unique} unique values (≤{self.max_unique_for_classification}) → Classification")
            return 'classification', 0.80

        # Many unique values
        if n_unique > self.max_unique_for_classification:
            logger.debug(f"Rule: {n_unique} unique values (>{self.max_unique_for_classification}) → Regression")
            return 'regression', 0.75

        return None

    def _check_unique_ratio(self, y: pd.Series) -> Tuple[str, float] | None:
        """
        Rule 4: Analyze unique value ratio

        Logic:
        - unique_ratio = unique_values / total_samples
        - Low ratio → Classification (values repeat)
        - High ratio → Regression (mostly unique)
        """
        n_unique = y.nunique()
        n_total = len(y.dropna())

        if n_total == 0:
            return None

        unique_ratio = n_unique / n_total

        # Very low ratio - values repeat a lot
        if unique_ratio < self.unique_ratio_threshold:
            logger.debug(f"Rule: Unique ratio {unique_ratio:.4f} < {self.unique_ratio_threshold} → Classification")
            return 'classification', 0.85

        # High ratio - most values are unique
        if unique_ratio > 0.5:
            logger.debug(f"Rule: Unique ratio {unique_ratio:.4f} > 0.5 → Regression")
            return 'regression', 0.80

        return None

    def _check_integer_pattern(self, y: pd.Series) -> Tuple[str, float] | None:
        """
        Rule 5: Analyze integer patterns

        Logic:
        - Integers with few unique values → Likely classification
        - Integers with sequential pattern → Could be IDs (skip)
        """
        # Only for numeric types
        if y.dtype not in ['int64', 'int32', 'int16', 'int8']:
            return None

        n_unique = y.nunique()

        # Few unique integers - likely classes
        if n_unique < 20:
            logger.debug(f"Rule: Integer with {n_unique} unique values → Classification")
            return 'classification', 0.70

        # Check if sequential (might be IDs)
        y_sorted = sorted(y.dropna().unique())
        if len(y_sorted) > 1:
            gaps = np.diff(y_sorted)
            if np.all(gaps == 1):
                logger.debug("Rule: Sequential integers (might be IDs) → Inconclusive")
                return None

        return None

    def _check_continuous_distribution(self, y: pd.Series) -> Tuple[str, float] | None:
        """Rule 6: Check for continuous distribution"""
        # Only for numeric types
        if y.dtype not in ['float64', 'float32', 'int64', 'int32']:
            return None

        n_unique = y.nunique()
        n_total = len(y.dropna())

        if n_total == 0:
            return None

        unique_ratio = n_unique / n_total

        # HIGH UNIQUE RATIO = REGRESSION (Make this stronger!)
        if unique_ratio > 0.8:
            logger.debug(f"Rule: High unique ratio {unique_ratio:.4f} → Regression")
            return 'regression', 0.95

        # For integers with high unique ratio (like prices)
        if y.dtype in ['int64', 'int32'] and unique_ratio > 0.5:
            logger.debug(f"Rule: Integer with high ratio {unique_ratio:.4f} → Regression")
            return 'regression', 0.90  # new rule

        return None

    def _aggregate_results(self, rules: list) -> Tuple[str, float]:
        """
        Aggregate results from multiple rules using weighted voting.

        Returns:
            Final (task_type, confidence) tuple
        """
        # Separate by task type
        classification_votes = [conf for task, conf in rules if task == 'classification']
        regression_votes = [conf for task, conf in rules if task == 'regression']

        # Calculate weighted scores
        classification_score = sum(classification_votes) if classification_votes else 0
        regression_score = sum(regression_votes) if regression_votes else 0

        # Determine winner
        if classification_score > regression_score:
            # Average confidence from classification votes
            avg_confidence = np.mean(classification_votes)
            return 'classification', float(avg_confidence)
        else:
            # Average confidence from regression votes
            avg_confidence = np.mean(regression_votes)
            return 'regression', float(avg_confidence)

    def get_detection_details(self, y: pd.Series) -> Dict:
        """
        Get detailed analysis for debugging/display.

        Returns dictionary with:
        - task_type
        - confidence
        - n_samples
        - n_unique
        - unique_ratio
        - dtype
        - sample_values
        """
        task_type, confidence = self.detect(y)

        n_unique = y.nunique()
        n_total = len(y.dropna())
        unique_ratio = n_unique / n_total if n_total > 0 else 0

        return {
            'task_type': task_type,
            'confidence': confidence,
            'n_samples': len(y),
            'n_unique': n_unique,
            'unique_ratio': unique_ratio,
            'dtype': str(y.dtype),
            'sample_values': y.dropna().head(10).tolist(),
            'null_count': y.isnull().sum(),
            'null_ratio': y.isnull().sum() / len(y) if len(y) > 0 else 0
        }


# Convenience function for quick detection
def detect_task_type(y: pd.Series, config: Config = None) -> Tuple[str, float]:
    """
    Quick function to detect task type without creating detector instance.

    Returns:
        Tuple of (task_type, confidence)
    """
    detector = TaskDetector(config)
    return detector.detect(y)