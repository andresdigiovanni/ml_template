from typing import Optional

from src.utils import Logger

from .local_data_connector import LocalDataConnector
from .sklearn_data_connector import SklearnDataConnector


def create_data_connector(source_type: str, logger: Optional[Logger] = None):
    if source_type == "local":
        return LocalDataConnector(logger)

    elif source_type == "sklearn":
        return SklearnDataConnector(logger)

    elif source_type == "s3":
        raise NotImplementedError("El conector para S3 aún no está implementado.")

    elif source_type == "gcs":
        raise NotImplementedError("El conector para GCS aún no está implementado.")

    else:
        raise ValueError(f"Origen de datos '{source_type}' no soportado.")
