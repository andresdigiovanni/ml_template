from typing import Optional

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, load_wine

from src.utils.logging import Logger

from .base_data_connector import BaseDataConnector


class SklearnDataConnector(BaseDataConnector):
    """Data connector for loading datasets from sklearn's built-in library.

    This class provides an interface to retrieve datasets like Wine, Iris, and Diabetes
    from the sklearn library, converting them into pandas DataFrames.

    Methods:
        get_data(source): Loads a specified dataset from sklearn and returns it as a DataFrame.
    """

    def __init__(self, logger: Optional[Logger] = None):
        """Initializes the SklearnDataConnector with supported datasets."""
        self.logger = logger

        self._datasets = {
            "wine": load_wine,
            "iris": load_iris,
            "diabetes": load_diabetes,
        }

    def get_data(self, source: str) -> pd.DataFrame:
        """Loads a dataset from sklearn's built-in collection.

        Args:
            source (str): The name of the dataset to load (e.g., 'wine', 'iris', 'diabetes').

        Returns:
            pd.DataFrame: A DataFrame containing the dataset's features and target column.

        Raises:
            ValueError: If the specified dataset is not supported.
        """
        if self.logger:
            self.logger.info(f"Loading data: {source}")

        if source not in self._datasets:
            raise ValueError(
                f"Dataset '{source}' is not supported. Available datasets: {list(self._datasets.keys())}."
            )

        # Load the dataset
        data = self._datasets[source]()

        # Convert to DataFrame and add target column
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

        return df
