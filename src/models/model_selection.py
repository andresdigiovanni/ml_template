import time

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LarsCV,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    LogisticRegression,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    RidgeCV,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.linear_model._quantile import QuantileRegressor
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import SVC, SVR, LinearSVR, NuSVC, NuSVR
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from tqdm import tqdm


class ModelSelection:
    def __init__(self, X, y, *, task, models=None, cv=5):
        """
        Initialize the class with models and data.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or np.ndarray): Target vector.
            task (str): Type of task ('classification' or 'regression').
            models (list, optional): List of model instances. Defaults to None.
            cv (int or callable, optional): Cross-validation strategy. Defaults to 5.
        """
        if task not in ["classification", "regression"]:
            raise ValueError(
                "Invalid task type. Choose 'classification' or 'regression'."
            )

        self.X = X
        self.y = y
        self.task = task
        self.models = models or self._select_models()
        self.cv = cv
        self.scoring = self._select_scoring()

    def _select_models(self):
        """
        Select default models based on the task (classification or regression).

        Returns:
            dict: A dictionary of default models.
        """
        n_samples = self.X.shape[0]

        model_selector = {
            "classification": self._classification_models,
            "regression": self._regression_models,
        }

        return model_selector[self.task](n_samples)

    def _classification_models(self, n_samples):
        models = [
            BernoulliNB(),
            CalibratedClassifierCV(),
            cb.CatBoostClassifier(verbose=0),
            ExtraTreeClassifier(),
            GaussianNB(),
            HistGradientBoostingClassifier(),
            LinearDiscriminantAnalysis(),
            lgb.LGBMClassifier(verbose=-1),
            LogisticRegression(max_iter=1_000),
            QuadraticDiscriminantAnalysis(),
            xgb.XGBClassifier(eval_metric="logloss"),
        ]

        if n_samples < 100_000:
            models.extend(
                [
                    AdaBoostClassifier(),
                    BaggingClassifier(),
                    DecisionTreeClassifier(),
                    ExtraTreesClassifier(),
                    GradientBoostingClassifier(),
                    KNeighborsClassifier(),
                    RandomForestClassifier(),
                ]
            )

        if n_samples < 10_000:
            models.extend(
                [
                    LabelPropagation(),
                    LabelSpreading(),
                    MLPClassifier(max_iter=1_000),
                    NuSVC(probability=True),
                    SVC(probability=True),
                ]
            )

        return models

    def _regression_models(self, n_samples):
        models = [
            BayesianRidge(),
            cb.CatBoostRegressor(verbose=0),
            ElasticNet(),
            ElasticNetCV(),
            ExtraTreeRegressor(),
            HistGradientBoostingRegressor(),
            HuberRegressor(),
            LassoCV(),
            LassoLars(),
            LassoLarsCV(),
            LassoLarsIC(),
            Lars(),
            LarsCV(),
            lgb.LGBMRegressor(verbose=-1),
            LinearRegression(),
            LinearSVR(),
            OrthogonalMatchingPursuit(),
            OrthogonalMatchingPursuitCV(),
            PassiveAggressiveRegressor(),
            RANSACRegressor(),
            Ridge(),
            RidgeCV(),
            SGDRegressor(),
            TransformedTargetRegressor(),
            TweedieRegressor(),
            xgb.XGBRegressor(),
        ]

        if n_samples < 100_000:
            models.extend(
                [
                    AdaBoostRegressor(),
                    BaggingRegressor(),
                    DecisionTreeRegressor(),
                    ExtraTreesRegressor(),
                    GradientBoostingRegressor(),
                    KNeighborsRegressor(),
                    MLPRegressor(max_iter=1_000),
                    RandomForestRegressor(),
                ]
            )

        if n_samples < 10_000:
            models.extend(
                [
                    GaussianProcessRegressor(),
                    KernelRidge(),
                    NuSVR(),
                    QuantileRegressor(),
                    SVR(),
                ]
            )

        return models

    def _select_scoring(self):
        """
        Select the scoring metrics based on the task.

        Returns:
            dict: A dictionary of scoring functions.
        """
        if self.task == "classification":
            n_classes = len(np.unique(self.y))
            metrics = {
                "Accuracy": "accuracy",
                "Balanced Accuracy": "balanced_accuracy",
                "Precision": "precision_weighted",
                "Recall": "recall_weighted",
                "F1 Score": "f1_weighted",
                "Avg Precision": "average_precision",
                "Log Loss": "neg_log_loss",
            }
            if n_classes == 2:
                metrics.update({"ROC AUC": "roc_auc"})
            else:
                metrics.update({"ROC AUC OVR": "roc_auc_ovr"})

            return metrics

        elif self.task == "regression":
            return {
                "R2": "r2",
                "Neg MSE": "neg_mean_squared_error",
                "Neg RMSE": "neg_root_mean_squared_error",
                "Neg MAE": "neg_mean_absolute_error",
                "MAPE": "neg_mean_absolute_percentage_error",
            }

    def evaluate(self, verbose=False):
        """
        Train and evaluate all models, returning results as a DataFrame.

        Args:
            verbose (bool): Whether to print progress during evaluation.

        Returns:
            pd.DataFrame: A DataFrame containing model names, metrics, and execution times.
        """
        results = []
        with tqdm(total=len(self.models), disable=(not verbose)) as pbar:
            for model in self.models:
                model_name = model.__class__.__name__
                pbar.set_description(model_name)
                pbar.update(1)

                start_time = time.time()
                scores = cross_validate(
                    model, self.X, self.y, cv=self.cv, scoring=self.scoring, n_jobs=-1
                )
                elapsed_time = round(time.time() - start_time, 2)

                result = {
                    "Model": model_name,
                    "Time Taken": elapsed_time,
                }
                for metric, values in scores.items():
                    if metric.startswith("test_"):
                        result[metric.replace("test_", "")] = np.mean(values)

                results.append(result)

        return (
            pd.DataFrame(results)
            .set_index("Model")
            .sort_values(
                by="Accuracy" if self.task == "classification" else "R2",
                ascending=False,
            )
        )


# Example of use
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression

    print("\n----- Classification -----\n")
    X, y = make_classification(n_samples=1_000, n_features=20, random_state=42)
    selector = ModelSelection(X, y, task="classification")
    results_df = selector.evaluate(verbose=True)
    print(results_df)

    print("\n----- Regression -----\n")
    X, y = make_regression(n_samples=1_000, n_features=20, random_state=42)
    selector = ModelSelection(X, y, task="regression")
    results_df = selector.evaluate(verbose=True)
    print(results_df)
