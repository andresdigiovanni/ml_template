from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold


def cross_val_score(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    scoring: Union[str, Callable],
    cv: int = 5,
    stratify: bool = True,
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform cross-validation, compute scores for each fold, and return predictions.

    Args:
        model (BaseEstimator): The machine learning model to evaluate.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        scoring (str or Callable): Scoring method. Can be a string (e.g., 'accuracy') or a custom callable.
        cv (int, optional): Number of cross-validation splits. Defaults to 5.
        stratify (bool, optional): Whether to use stratified splitting for classification problems. Defaults to True.

    Returns:
        Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
            - Scores for each fold.
            - Concatenated true labels from all folds.
            - Concatenated predicted values from all folds.
            - Concatenated predicted probabilities from all folds.
    """
    splitter = _get_splitter(cv=cv, stratify=stratify, y=y)
    scoring_function = _get_scoring_function(scoring)
    scores = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for train_idx, val_idx in splitter.split(X, y):
        X_train, X_val = _extract_data(X, train_idx), _extract_data(X, val_idx)
        y_train, y_val = _extract_data(y, train_idx), _extract_data(y, val_idx)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        all_y_true.append(y_val)
        all_y_pred.append(y_pred)
        all_y_prob.append(y_prob)

        try:
            scores.append(scoring_function(y_val, y_pred, y_prob))

        except TypeError:
            scores.append(scoring_function(y_val, y_pred))

    return (
        scores,
        np.concatenate(all_y_true),
        np.concatenate(all_y_pred),
        np.concatenate(all_y_prob),
    )


def _get_splitter(
    cv: int, stratify: bool, y: Union[pd.Series, np.ndarray]
) -> Union[KFold, StratifiedKFold]:
    """
    Get the appropriate KFold or StratifiedKFold splitter.

    Args:
        cv (int): Number of splits.
        stratify (bool): Whether to use stratified splitting.
        y (pd.Series or np.ndarray): Target vector for stratification.

    Returns:
        KFold or StratifiedKFold: Configured cross-validation splitter.
    """
    if stratify:
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    else:
        return KFold(n_splits=cv, shuffle=True, random_state=42)


def _get_scoring_function(scoring: Union[str, Callable]) -> Callable:
    """
    Get the scoring function.

    Args:
        scoring (Union[str, Callable]): Scoring method as a string or callable.

    Returns:
        Callable: Scoring function.
    """
    return scoring if callable(scoring) else get_scorer(scoring)._score_func


def _extract_data(
    data: Union[pd.DataFrame, pd.Series, np.ndarray], indices: np.ndarray
) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Extract a subset of data using the given indices.

    Args:
        data (Union[pd.DataFrame, pd.Series, np.ndarray]): Input data.
        indices (np.ndarray): Indices to extract.

    Returns:
        Union[pd.DataFrame, pd.Series, np.ndarray]: Extracted subset of data.

    Raises:
        TypeError: If the data type is not compatible.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[indices]

    if isinstance(data, np.ndarray):
        return data[indices]

    raise TypeError(f"Unsupported data type: {type(data)}")
