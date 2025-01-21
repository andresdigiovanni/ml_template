from typing import Callable, Optional, Union

import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate


class HyperparameterTuner:
    """
    A class for hyperparameter tuning and cross-validation using Optuna.

    Attributes:
        model (BaseEstimator): The machine learning model to optimize.
        scoring (Union[str, Callable]): Scoring metric, either a string recognized by scikit-learn or a custom callable.
        direction (str): Optimization direction ('maximize' or 'minimize').
        model_tracker (Optional): A model tracker connector for saving models and results.
        param_grid (dict): Hyperparameter search space.
        cv (Union[int, Callable]): Number of cross-validation folds or a custom cross-validation generator.
        groups (Optional): Group labels for cross-validation splitting.
        n_trials (int): Number of optimization trials.
        best_params_ (Optional[dict]): The best hyperparameters found after optimization.
        best_score_ (Optional[float]): The best score achieved during optimization.
    """

    def __init__(
        self,
        model: BaseEstimator,
        scoring: Union[str, Callable],
        direction: str,
        model_tracker=None,
        param_grid: Optional[dict] = None,
        cv: Union[int, Callable] = 5,
        groups: Optional = None,
        n_trials: int = 100,
    ) -> None:
        """
        Initializes the HyperparameterTuner.

        Args:
            model (BaseEstimator): Machine learning model to optimize.
            scoring (Union[str, Callable]): Scoring metric for evaluation.
            direction (str): Optimization direction ('maximize' or 'minimize').
            model_tracker (Optional): The connector for saving models and results.
            param_grid (Optional[dict]): Hyperparameter search space.
            cv (Union[int, Callable]): Number of cross-validation folds or a custom generator.
            groups (Optional): Group labels for the samples.
            n_trials (int): Number of optimization trials. Default is 100.
        """
        self.model = model
        self.scoring = scoring
        self.direction = direction
        self.model_tracker = model_tracker
        self.param_grid = param_grid or self._get_default_param_grid(
            model.__class__.__name__
        )
        self.cv = cv
        self.groups = groups
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
        elif model_name in ("LGBMClassifier", "LGBMRegressor"):
            return {
                "num_leaves": {"range": (20, 150)},
                "learning_rate": {"range": (1e-4, 0.1), "log": True},
                "max_depth": {"range": (3, 20)},
                "min_child_samples": {"range": (10, 100)},
                "subsample": {"range": (0.5, 0.9)},
                "colsample_bytree": {"range": (0.5, 0.9)},
                "reg_alpha": {"range": (1e-6, 10.0), "log": True},
                "reg_lambda": {"range": (1e-6, 10.0), "log": True},
                "n_estimators": {"range": (50, 200)},
                "min_split_gain": {"range": (0.0, 1.0)},
                "max_bin": {"range": (128, 512)},
                "feature_fraction": {"range": (0.5, 1.0)},
                "boosting_type": {"range": ["gbdt", "dart", "goss"]},
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
                "gamma": {"range": (1e-6, 5.0), "log": True},
                "learning_rate": {"range": (1e-4, 0.5), "log": True},
                "max_depth": {"range": (3, 20)},
                "min_child_weight": {"range": (1e-4, 10.0), "log": True},
                "n_estimators": {"range": (50, 200)},
                "reg_alpha": {"range": (1e-6, 10.0), "log": True},
                "reg_lambda": {"range": (1e-6, 10.0), "log": True},
                "subsample": {"range": (0.5, 1.0)},
                "max_bin": {"range": (128, 512)},
                "grow_policy": {"range": ["depthwise", "lossguide"]},
            }

        raise ValueError(f"Unsupported model: {model_name}")

    def _objective(self, trial: optuna.trial.Trial, X, y) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

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
        scores = cross_validate(
            self.model, X, y, scoring=self.scoring, cv=self.cv, groups=self.groups
        )
        score_mean = np.mean(scores["test_score"])

        if self.model_tracker:
            with self.model_tracker.run(nested=True):
                self.model_tracker.log_params(params)
                self.model_tracker.log_metric(self.scoring, score_mean)

        return score_mean

    def fit(self, X, y) -> dict:
        """
        Performs hyperparameter optimization and cross-validation.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

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
