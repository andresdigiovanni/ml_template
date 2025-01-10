from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import StratifiedKFold

from src.utils.data import subset_data
from src.utils.metrics import average_metrics


def cross_validate_with_predictions(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    scoring: Union[str, Callable],
    cv: Union[int, Callable] = 5,
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    """
    Perform cross-validation, compute scores for each fold, and return predictions.

    Args:
        model (BaseEstimator): The machine learning model to evaluate.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        scoring (str or Callable): Scoring method. Can be a string (e.g., 'accuracy') or a callable function.
        cv (int or Callable, optional): Number of cross-validation splits or a custom splitter. Defaults to 5.

    Returns:
        Tuple[List[Dict[str, float]], np.ndarray, np.ndarray, Union[np.ndarray, None]]:
            - Scores for each fold.
            - Concatenated true labels from all folds.
            - Concatenated predicted values from all folds.
            - Concatenated predicted probabilities from all folds (None if not available).
    """
    if X is None or y is None:
        raise ValueError("Feature matrix X and target vector y cannot be None.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    splitter = _get_splitter_function(cv)
    scoring_function = _get_scoring_function(scoring)

    scores = []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for train_idx, val_idx in splitter.split(X, y):
        X_train, X_val = subset_data(X, train_idx), subset_data(X, val_idx)
        y_train, y_val = subset_data(y, train_idx), subset_data(y, val_idx)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        all_y_true.append(y_val)
        all_y_pred.append(y_pred)

        # Handle models that do not implement predict_proba
        try:
            y_prob = model.predict_proba(X_val)
            all_y_prob.append(y_prob)
        except AttributeError:
            y_prob = None

        scores.append(scoring_function(model, X_val, y_val))

    if any(isinstance(item, dict) for item in scores):
        scores = average_metrics(scores)

    return (
        scores,
        np.concatenate(all_y_true),
        np.concatenate(all_y_pred),
        np.concatenate(all_y_prob) if all_y_prob else None,
    )


def _get_splitter_function(cv: Union[int, Callable]) -> Callable:
    """
    Get the splitter function for cross-validation.

    Args:
        cv (Union[int, Callable]): Number of splits or callable splitter.

    Returns:
        Callable: Splitter function.
    """
    return (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        if isinstance(cv, int)
        else cv
    )


def _get_scoring_function(scoring: Union[str, Callable]) -> Callable:
    """
    Get the scoring function.

    Args:
        scoring (Union[str, Callable]): Scoring method as a string or callable.

    Returns:
        Callable: Scoring function.
    """
    return (
        scoring if callable(scoring) else make_scorer(get_scorer(scoring)._score_func)
    )
