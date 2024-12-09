from typing import Optional

from src.utils import Logger

from .local_object_connector import LocalObjectConnector


def create_object_connector(connector_type, logger: Optional[Logger] = None, **kwargs):
    """
    Creates an ObjectConnector based on the configuration.

    Args:
        connector_type (str): The type of connector to create.
        logger (Optional[Logger]): An optional logger instance.
        **kwargs: Additional keyword arguments for the connector.

    Returns:
        ObjectConnector: An instance of the selected connector.

    Raises:
        ValueError: If the connector type is unsupported.
    """

    if connector_type == "local":
        return LocalObjectConnector(logger=logger, **kwargs)

    elif connector_type == "s3":
        raise NotImplementedError("The connector for S3 is not implemented yet.")

    elif connector_type == "gcs":
        raise NotImplementedError("The connector for GCS is not implemented yet.")

    else:
        raise ValueError(f"Connector type '{connector_type}' is not supported.")
