from typing import Optional

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, load_wine

from src.utils.logging import Logger

from .base_data_connector import BaseDataConnector


class SklearnDataConnector(BaseDataConnector):
    """Data connector for loading datasets from sklearn's built-in library.

    Provides an interface to retrieve popular datasets (e.g., Wine, Iris, Diabetes)
    from sklearn, converting them into pandas DataFrames.

    Attributes:
        SUPPORTED_DATASETS (dict): Mapping of dataset names to their corresponding
            sklearn loading functions.
    """

    SUPPORTED_DATASETS = {
        "wine": load_wine,
        "iris": load_iris,
        "diabetes": load_diabetes,
    }

    def __init__(self, logger: Optional[Logger] = None):
        """Initializes the SklearnDataConnector.

        Args:
            logger (Optional[Logger]): Logger instance for logging operations. Defaults to None.
        """
        self.logger = logger

    def get_data(self, source: str) -> pd.DataFrame:
        """Loads a dataset from sklearn's built-in collection.

        Args:
            source (str): Name of the dataset to load (e.g., 'wine', 'iris', 'diabetes').

        Returns:
            pd.DataFrame: A DataFrame containing the dataset's features and target column.

        Raises:
            ValueError: If the specified dataset is not supported or an error occurs during loading.
        """
        if not source or not isinstance(source, str):
            raise ValueError("The 'source' must be a non-empty string.")

        if self.logger:
            self.logger.info(f"Attempting to load dataset: '{source}'")

        if source not in self.SUPPORTED_DATASETS:
            available_datasets = ", ".join(self.SUPPORTED_DATASETS.keys())
            error_message = f"Dataset '{source}' is not supported. Available datasets: {available_datasets}."

            if self.logger:
                self.logger.error(error_message)

            raise ValueError(error_message)

        try:
            # Load the dataset
            data = self.SUPPORTED_DATASETS[source]()

            # Convert to DataFrame and add target column
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target

            if self.logger:
                self.logger.info(f"Dataset '{source}' loaded successfully.")

            return df

        except Exception as error:
            if self.logger:
                self.logger.error(f"Error loading dataset '{source}': {error}")

            raise ValueError(f"Error loading dataset '{source}': {error}")
