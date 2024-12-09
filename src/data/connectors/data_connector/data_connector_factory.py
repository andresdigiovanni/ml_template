from typing import Optional

from src.utils import Logger

from .local_data_connector import LocalDataConnector
from .sklearn_data_connector import SklearnDataConnector


def create_data_connector(source_type: str, logger: Optional[Logger] = None):
    """Creates and returns a data connector based on the specified source type.

    Args:
        source_type (str): The type of data source. Supported values are: "local", "sklearn", "s3" or "gcs".
        logger (Optional[Logger]): An optional logger instance for logging purposes.

    Returns:
        LocalDataConnector or SklearnDataConnector: The corresponding data connector instance based on the `source_type`.

    Raises:
        NotImplementedError: If the `source_type` is "s3" or "gcs", as these connectors are not yet implemented.
        ValueError: If the `source_type` is not one of the supported values.
    """
    if source_type == "local":
        return LocalDataConnector(logger)

    elif source_type == "sklearn":
        return SklearnDataConnector(logger)

    elif source_type == "s3":
        raise NotImplementedError("The connector for S3 is not implemented yet.")

    elif source_type == "gcs":
        raise NotImplementedError("The connector for GCS is not implemented yet.")

    else:
        raise ValueError(f"Data source '{source_type}' is not supported.")
