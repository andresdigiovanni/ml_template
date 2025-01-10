from typing import Optional, Union

import numpy as np
import pandas as pd
from cross import CrossTransformer, auto_transform


class DataProcessor:
    """
    Automates data preprocessing and feature transformations using CrossTransformer.

    Attributes:
        model: The ML model used for optimization in auto-transformations.
        scoring (str): Scoring metric to evaluate transformations.
        direction (str): Optimization direction ('maximize' or 'minimize').
        transformer (CrossTransformer): Manages transformations.
    """

    def __init__(
        self, model, scoring: Optional[str] = None, direction: Optional[str] = None
    ) -> None:
        """
        Initializes the DataProcessor.

        Args:
            model: The machine learning model to guide the transformation process.
            scoring (Optional[str]): Scoring metric (e.g., 'accuracy', 'r2').
            direction (Optional[str]): Direction of optimization ('maximize' or 'minimize').

        Raises:
            ValueError: If direction is not 'maximize' or 'minimize'.
        """
        if direction not in {None, "maximize", "minimize"}:
            raise ValueError("`direction` must be 'maximize', 'minimize', or None.")

        self.model = model
        self.scoring = scoring
        self.direction = direction
        self.transformer: Optional[CrossTransformer] = None

    def get_params(self) -> dict:
        """
        Retrieves current transformation parameters.

        Returns:
            dict: Parameters of the transformation pipeline.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.transformer is None:
            raise RuntimeError("No transformations are set. Call fit() first.")

        return self.transformer.get_params()

    def set_params(self, transformations: dict) -> None:
        """
        Manually sets the transformation pipeline.

        Args:
            transformations (dict): A dictionary defining the transformations to apply.
        """
        self.transformer = CrossTransformer()
        self.transformer.set_params(**transformations)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "DataProcessor":
        """
        Fits transformations to the data using the model and scoring.

        Args:
            X (pd.DataFrame or np.ndarray): Feature data.
            y (Optional[pd.Series or np.ndarray]): Target data.

        Returns:
            DataProcessor: The fitted instance.
        """
        self.transformations = auto_transform(
            X, y, self.model, self.scoring, self.direction
        )
        self.transformer = CrossTransformer(self.transformations)
        self.transformer.fit(X, y)
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Applies transformations to the data.

        Args:
            X (pd.DataFrame or np.ndarray): Feature data.
            y (Optional[pd.Series or np.ndarray]): Target data.

        Returns:
            Transformed data.

        Raises:
            TransformationNotFittedError: If fit() has not been called.
        """
        if self.transformer is None:
            raise RuntimeError("fit() must be called before applying transform().")

        return self.transformer.transform(X, y)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fits and transforms the data.

        Args:
            X (pd.DataFrame or np.ndarray): Feature data.
            y (Optional[pd.Series or np.ndarray]): Target data.

        Returns:
            Transformed data.
        """
        return self.fit(X, y).transform(X, y)
