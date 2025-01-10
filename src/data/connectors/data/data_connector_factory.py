from typing import Optional

from src.utils.logging import Logger

from .base_data_connector import BaseDataConnector
from .local_data_connector import LocalDataConnector
from .sklearn_data_connector import SklearnDataConnector


def create_data_connector(
    connector_type: str, logger: Optional[Logger] = None
) -> BaseDataConnector:
    """Creates and returns a data connector based on the specified source type.

    Args:
        connector_type (str): The type of data source. Supported values are: "local", "sklearn".
        logger (Optional[Logger]): An optional logger instance for logging purposes.

    Returns:
        BaseDataConnector: The corresponding data connector instance based on the `connector_type`.

    Raises:
        ValueError: If the `connector_type` is not one of the supported values.
    """
    SUPPORTED_CONNECTORS = {
        "local": LocalDataConnector,
        "sklearn": SklearnDataConnector,
    }

    if logger:
        logger.info(f"Creating data connector for type: '{connector_type}'")

    if connector_type in SUPPORTED_CONNECTORS:
        return SUPPORTED_CONNECTORS[connector_type](logger)

    else:
        error_message = f"Data source '{connector_type}' is not supported."
        if logger:
            logger.error(error_message)

        raise ValueError(error_message)
