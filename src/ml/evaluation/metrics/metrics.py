from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    problem_type: str = "classification",
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute evaluation metrics for classification or regression problems.

    Args:
        model (BaseEstimator): The machine learning model to evaluate.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        problem_type (str): The type of problem. Either 'classification' or 'regression'.

    Returns:
        dict: A dictionary of computed metrics with metric names as keys and their values.

    Raises:
        ValueError: If problem_type is not 'classification' or 'regression'.
    """

    y_pred = model.predict(X)

    if problem_type == "classification":
        y_prob = model.predict_proba(X)
        return _compute_classification_metrics(y, y_pred, y_prob)

    elif problem_type == "regression":
        return _compute_regression_metrics(y, y_pred)

    else:
        raise ValueError(
            "Invalid problem_type. Choose 'classification' or 'regression'."
        )


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics such as accuracy, precision, recall, F1 score, and ROC AUC.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray, optional): Predicted probabilities for the positive class.

    Returns:
        dict: A dictionary with classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }

    if y_prob is not None:
        unique_classes = np.unique(y_true)

        # ROC AUC
        if len(unique_classes) == 2:
            y_prob_positive = y_prob[:, 1]  # Probabilities of positive class
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob_positive)

        else:
            y_true_binarized = label_binarize(y_true, classes=unique_classes)
            metrics["roc_auc"] = roc_auc_score(
                y_true_binarized, y_prob, multi_class="ovr", average="macro"
            )

        # Log Loss
        metrics["log_loss"] = log_loss(y_true, y_prob)

    return metrics


def _compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics such as Mean Squared Error, Mean Absolute Error, and R² score.

    Args:
        y_true (np.ndarray): Ground truth target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: A dictionary with regression metrics.
    """
    return {
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred),
    }
