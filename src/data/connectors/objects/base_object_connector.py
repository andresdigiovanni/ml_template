from abc import ABC, abstractmethod
from typing import Any


class BaseObjectConnector(ABC):
    """
    Abstract base class defining the interface for object storage connectors.

    This class serves as a blueprint for implementing connectors that interact
    with various object storage systems (e.g., local filesystem, S3, Azure Blob Storage).
    """

    @abstractmethod
    def get_object(self, object_path: str, destination: str) -> None:
        """
        Downloads an object from the storage to a local destination.

        Args:
            object_path (str): Path or identifier of the object in the storage.
            destination (str): Local path where the object will be downloaded.

        Raises:
            NotImplementedError: Must be implemented in the subclass.
        """
        raise NotImplementedError("The 'get_object' method must be implemented.")

    @abstractmethod
    def put_object(self, data: Any, object_path: str) -> None:
        """
        Uploads an object to the storage.

        Args:
            data (Any): The object or data to be uploaded (e.g., file content, serialized data).
            object_path (str): Path or identifier where the object will be stored.

        Raises:
            NotImplementedError: Must be implemented in the subclass.
        """
        raise NotImplementedError("The 'put_object' method must be implemented.")
