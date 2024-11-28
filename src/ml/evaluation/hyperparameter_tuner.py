from typing import Callable, Union

import numpy as np
import optuna
from sklearn.base import BaseEstimator

from .cross_val_score import cross_val_score


class HyperparameterTuner:
    """
    A class for hyperparameter tuning and cross-validation using Optuna.

    Attributes:
        model (BaseEstimator): The machine learning model to optimize.
        scoring (str or callable): Scoring metric, either a string recognized by scikit-learn or a custom callable.
        direction (str): Optimization direction ('maximize' or 'minimize').
        param_grid (dict): Hyperparameter search space.
        cv (int): Number of cross-validation folds.
        n_trials (int): Number of optimization trials.
        best_params_ (dict): The best hyperparameters found after optimization.
        best_score_ (float): The best score achieved during optimization.
    """

    def __init__(
        self,
        model: BaseEstimator,
        scoring: Union[str, Callable],
        direction: str,
        param_grid: dict = None,
        cv: int = 5,
        n_trials: int = 50,
    ):
        """
        Initializes the HyperparameterTuner.

        Args:
            model (BaseEstimator): Machine learning model to optimize.
            scoring (str or callable): Scoring metric for evaluation.
            direction (str): Optimization direction ('maximize' or 'minimize').
            param_grid (dict, optional): Hyperparameter search space.
            cv (int, optional): Number of cross-validation folds. Default is 5.
            n_trials (int, optional): Number of optimization trials. Default is 50.
        """
        self.model = model
        self.scoring = scoring
        self.direction = direction
        self.param_grid = param_grid or self._get_default_param_grid(
            model.__class__.__name__
        )
        self.cv = cv
        self.n_trials = n_trials
        self.best_params_ = None
        self.best_score_ = None

    def _get_default_param_grid(self, model_name: str) -> dict:
        """
        Provides default hyperparameter ranges based on the model type.

        Args:
            model_name (str): Name of the model class.

        Returns:
            dict: Hyperparameter ranges.

        Raises:
            ValueError: If the model type is not supported.
        """
        if model_name in {"DecisionTreeClassifier", "DecisionTreeRegressor"}:
            return {
                "max_depth": {"range": (1, 50)},
                "min_samples_split": {"range": (2, 20)},
                "min_samples_leaf": {"range": (1, 10)},
                "max_features": {"range": (0.1, 1.0)},
                "criterion": {
                    "range": ["gini", "entropy", "log_loss"]
                    if "Classifier" in model_name
                    else ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
            }
        elif model_name == "LinearRegression":
            return {
                "fit_intercept": {"range": [True, False]},
            }
        elif model_name == "LogisticRegression":
            return {
                "penalty": {"range": ["l1", "l2", "elasticnet", "none"]},
                "C": {"range": (1e-4, 10.0), "log": True},
                "solver": {"range": ["lbfgs", "liblinear", "saga", "newton-cg"]},
                "max_iter": {"range": (100, 1000)},
                "tol": {"range": (1e-5, 1e-1), "log": True},
            }
        elif model_name in {"RandomForestClassifier", "RandomForestRegressor"}:
            return {
                "n_estimators": {"range": (50, 300)},
                "max_depth": {"range": (5, 50)},
                "min_samples_split": {"range": (2, 20)},
                "min_samples_leaf": {"range": (1, 10)},
                "max_features": {"range": (0.1, 1.0)},
                "bootstrap": {"range": [True, False]},
                "criterion": {
                    "range": ["gini", "entropy", "log_loss"]
                    if "Classifier" in model_name
                    else ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                },
            }
        elif model_name in ("XGBClassifier", "XGBRegressor"):
            return {
                "colsample_bytree": {"range": (0.5, 1.0)},
                "gamma": {"range": (1e-6, 1.0), "log": True},
                "learning_rate": {"range": (1e-4, 0.3), "log": True},
                "max_depth": {"range": (4, 20)},
                "min_child_weight": {"range": (1e-6, 10.0), "log": True},
                "n_estimators": {"range": (20, 200)},
                "reg_alpha": {"range": (1e-6, 10.0), "log": True},
                "reg_lambda": {"range": (1e-6, 10.0), "log": True},
                "subsample": {"range": (0.5, 1.0)},
                "scale_pos_weight": {"range": (1e-2, 10.0), "log": True},
                "max_bin": {"range": (128, 512)},
                "grow_policy": {"range": ["depthwise", "lossguide"]},
            }

        raise ValueError(f"Unsupported model: {model_name}")

    def _objective(self, trial: optuna.trial.Trial, X, y) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            float: The mean score across cross-validation folds.
        """
        params = {}
        for key, value in self.param_grid.items():
            range_ = value["range"]
            log = value.get("log", False)

            if isinstance(range_, list):
                params[key] = trial.suggest_categorical(key, range_)

            elif isinstance(range_[0], int) and isinstance(range_[1], int):
                params[key] = trial.suggest_int(key, range_[0], range_[1], log=log)

            else:
                params[key] = trial.suggest_float(key, range_[0], range_[1], log=log)

        self.model.set_params(**params)
        scores, _, _, _ = cross_val_score(self.model, X, y, self.scoring, self.cv)
        return np.mean(scores)

    def fit(self, X, y) -> dict:
        """
        Performs hyperparameter optimization and cross-validation.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target vector.

        Returns:
            dict: Best hyperparameters found.
        """
        study = optuna.create_study(direction=self.direction)
        study.optimize(
            lambda trial: self._objective(trial, X, y), n_trials=self.n_trials
        )
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        return self.best_params_

    def get_best_score(self) -> float:
        """
        Retrieves the best score from optimization.

        Returns:
            float: Best score achieved.

        Raises:
            RuntimeError: If `fit` has not been called.
        """
        if self.best_score_ is None:
            raise RuntimeError("fit must be called before accessing the best score.")

        return self.best_score_

    def get_best_params(self) -> dict:
        """
        Retrieves the best parameters from optimization.

        Returns:
            dict: Best hyperparameters found.

        Raises:
            RuntimeError: If `fit` has not been called.
        """
        if self.best_params_ is None:
            raise RuntimeError(
                "fit must be called before accessing the best parameters."
            )

        return self.best_params_
