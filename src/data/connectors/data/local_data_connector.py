from typing import Optional

import pandas as pd

from src.utils.logging import Logger

from .base_data_connector import BaseDataConnector


class LocalDataConnector(BaseDataConnector):
    """Data connector for local file sources."""

    SUPPORTED_FORMATS = [".csv", ".parquet"]

    def __init__(self, logger: Optional[Logger] = None):
        """
        Initializes the LocalDataConnector.

        Args:
            logger (Optional[Logger]): Logger instance for logging operations. Defaults to None.
        """
        self.logger = logger

    def get_data(self, source: str) -> pd.DataFrame:
        """Retrieves data from a local file.

        Args:
            source (str): Path to the local file.

        Returns:
            pd.DataFrame: The data loaded into a DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported or there's an error loading the file.
        """
        if not source or not isinstance(source, str):
            raise ValueError(
                "The 'source' must be a non-empty string representing a file path."
            )

        if self.logger:
            self.logger.info(f"Attempting to load data from: {source}")

        try:
            if source.endswith(".csv"):
                data = pd.read_csv(source)

            elif source.endswith(".parquet"):
                data = pd.read_parquet(source)

            else:
                raise ValueError(
                    f"Unsupported file format for: {source}. "
                    f"Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}"
                )

            if self.logger:
                self.logger.info(f"Data successfully loaded from: {source}")

            return data

        except FileNotFoundError:
            if self.logger:
                self.logger.error(f"File not found: {source}")

            raise

        except Exception as error:
            if self.logger:
                self.logger.error(f"Error loading data from {source}: {error}")

            raise ValueError(f"Error loading data from {source}: {error}")
