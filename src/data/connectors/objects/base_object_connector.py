from abc import ABC, abstractmethod


class BaseObjectConnector(ABC):
    """
    Abstract base class for object storage connectors.

    Provides a blueprint for implementing connectors to interact with various
    object storage systems (e.g., local filesystem, S3, Azure Blob Storage).
    """

    @abstractmethod
    def get_object(self, object_path: str, destination: str) -> None:
        """
        Downloads an object from the storage to a local destination.

        Args:
            object_path (str): Path or identifier of the object in the storage.
            destination (str): Local path where the object will be downloaded.
        """
        pass

    @abstractmethod
    def put_object(self, data: bytes, object_path: str) -> None:
        """
        Uploads an object to the storage.

        Args:
            data (bytes): The data to be uploaded (e.g., file content in bytes).
            object_path (str): Path or identifier where the object will be stored.
        """
        pass

    @abstractmethod
    def delete_object(self, object_path: str) -> None:
        """
        Deletes an object from the storage.

        Args:
            object_path (str): Path or identifier of the object to be deleted.
        """
        pass
