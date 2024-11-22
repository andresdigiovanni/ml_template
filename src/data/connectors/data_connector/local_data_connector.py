from typing import Optional

import pandas as pd

from src.utils import Logger

from .base_data_connector import BaseDataConnector


class LocalDataConnector(BaseDataConnector):
    """Data connector for local file sources."""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    def get_data(self, source: str) -> pd.DataFrame:
        """Retrieves data from a local file.

        Args:
            source (str): Path to the local file.

        Returns:
            pd.DataFrame: The data loaded into a DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
        """
        if self.logger:
            self.logger.info(f"Loading data: {source}")

        if source.endswith(".csv"):
            return pd.read_csv(source)

        elif source.endswith(".parquet"):
            return pd.read_parquet(source)

        else:
            raise ValueError(f"Unsupported file format for: {source}")
