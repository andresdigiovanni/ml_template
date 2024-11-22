from typing import Optional

from src.utils import Logger

from .local_object_connector import LocalObjectConnector


def create_object_connector(connector_type, logger: Optional[Logger] = None, **kwargs):
    """
    Crea un ObjectConnector basado en la configuración.

    Args:
        config (dict): Diccionario de configuración.

    Returns:
        ObjectConnector: Instancia del conector seleccionado.
    """

    if connector_type == "local":
        return LocalObjectConnector(logger=logger, **kwargs)

    elif connector_type == "s3":
        raise NotImplementedError("El conector para S3 aún no está implementado.")

    elif connector_type == "gcs":
        raise NotImplementedError("El conector para GCS aún no está implementado.")

    else:
        raise ValueError(f"Tipo de conector '{connector_type}' no soportado.")
