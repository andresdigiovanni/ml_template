from abc import ABC, abstractmethod
from typing import Any


class BaseDataConnector(ABC):
    """Abstract base class for data connectors.

    This class serves as a blueprint for all data connectors, defining the required
    methods to be implemented by subclasses. It enforces the implementation of the
    `get_data` method, which retrieves data from a specific source.
    """

    @abstractmethod
    def get_data(self, source: Any) -> Any:
        """Retrieve data from a specified source.

        Subclasses must implement this method to define how data is fetched from the
        given source.

        Args:
            source (Any): The source from which to retrieve data. This could vary
                depending on the implementation (e.g., a file path, a database URI, etc.).

        Returns:
            Any: The retrieved data. The return type depends on the specific implementation.
        """
        pass
