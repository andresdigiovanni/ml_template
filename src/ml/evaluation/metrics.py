from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class Metrics:
    """
    A class to compute various evaluation metrics for classification and regression problems.

    Attributes:
        problem_type (str): The type of problem ('classification' or 'regression').
        results (dict): A dictionary to store the last computed metrics.
    """

    def __init__(self, problem_type: str):
        """
        Initialize the Metrics class based on the problem type.

        Args:
            problem_type (str): Type of the problem. Can be either 'classification' or 'regression'.

        Raises:
            ValueError: If the provided problem_type is not 'classification' or 'regression'.
        """
        if problem_type not in ["classification", "regression"]:
            raise ValueError(
                "Invalid problem_type. Choose 'classification' or 'regression'."
            )

        self.problem_type: str = problem_type
        self.results: Dict[str, float] = {}

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute the appropriate metrics based on the problem type.

        Args:
            y_true (array-like): Ground truth labels or target values.
            y_pred (array-like): Predicted labels or values.
            y_prob (array-like, optional): Predicted probabilities for classification (default is None).

        Returns:
            dict: A dictionary of computed metrics with metric names as keys and their values.
        """
        if self.problem_type == "classification":
            return self._compute_classification_metrics(y_true, y_pred, y_prob)

        elif self.problem_type == "regression":
            return self._compute_regression_metrics(y_true, y_pred)

    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute classification metrics such as accuracy, precision, recall, F1 score, and ROC AUC.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.
            y_prob (array-like, optional): Predicted probabilities for the positive class.

        Returns:
            dict: A dictionary with classification metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
        }

        # If probabilities are provided, calculate ROC AUC (for binary classification)
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

        self.results = metrics
        return metrics

    def _compute_regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute regression metrics such as Mean Squared Error, Mean Absolute Error, and R² score.

        Args:
            y_true (array-like): Ground truth target values.
            y_pred (array-like): Predicted target values.

        Returns:
            dict: A dictionary with regression metrics.
        """
        metrics = {
            "mean_squared_error": mean_squared_error(y_true, y_pred),
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
        }

        self.results = metrics
        return metrics

    def get_results(self) -> Dict[str, float]:
        """
        Retrieve the last computed metrics.

        Returns:
            dict: A dictionary containing the last computed metrics.
        """
        return self.results
