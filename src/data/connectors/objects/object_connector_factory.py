from typing import Optional

from src.utils.logging import Logger

from .local_object_connector import LocalObjectConnector


def create_object_connector(connector_type, logger: Optional[Logger] = None, **kwargs):
    """
    Creates an ObjectConnector based on the configuration.

    Args:
        connector_type (str): The type of connector to create. Supported types: "local".
        logger (Optional[Logger]): An optional logger instance.
        **kwargs: Additional keyword arguments for the connector.

    Returns:
        BaseObjectConnector: An instance of the selected connector.

    Raises:
        ValueError: If the connector type is unsupported.
    """
    SUPPORTED_CONNECTORS = {"local": LocalObjectConnector}

    if logger:
        logger.info(f"Creating object connector for type: '{connector_type}'")

    if connector_type in SUPPORTED_CONNECTORS:
        return SUPPORTED_CONNECTORS[connector_type](logger=logger, **kwargs)

    else:
        error_message = f"Connector '{connector_type}' is not supported."
        if logger:
            logger.error(error_message)

        raise ValueError(error_message)
