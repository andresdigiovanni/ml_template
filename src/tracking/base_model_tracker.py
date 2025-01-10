from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModelTracker(ABC):
    """
    Abstract base class for model trackers, defining the required interface for tracking experiments,
    logging parameters, metrics, models, and artifacts.

    This class should be subclassed to create specific implementations for different tracking systems.
    """

    @abstractmethod
    def __init__(self, experiment_name: str):
        """
        Initialize the model tracker with a given experiment name.

        Args:
            experiment_name (str): The name of the experiment to track.
        """
        pass

    @abstractmethod
    def run_id(self) -> str:
        """
        Retrieve the current run ID.

        Returns:
            str: The identifier of the current run.
        """
        pass

    @abstractmethod
    def run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Context manager for managing a run's lifecycle.

        Args:
            run_name (str, optional): The name of the run. Defaults to None.
            nested (bool, optional): Whether the run is nested within a parent run. Defaults to False.

        Returns:
            Context manager for the run.
        """
        pass

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Start a new run.

        Args:
            run_name (str, optional): The name of the run. Defaults to None.
            nested (bool, optional): Whether the run is nested within a parent run. Defaults to False.
        """
        pass

    @abstractmethod
    def end_run(self):
        """
        End the current run.
        """
        pass

    @abstractmethod
    def log_param(self, name: str, value: Any):
        """
        Log a single parameter.

        Args:
            name (str): The name of the parameter.
            value (Any): The value of the parameter.
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """
        Log multiple parameters.

        Args:
            params (dict): A dictionary of parameter names and values.
        """
        pass

    @abstractmethod
    def log_metric(self, name: str, value: float):
        """
        Log a single metric.

        Args:
            name (str): The name of the metric.
            value (float): The value of the metric.
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log multiple metrics.

        Args:
            metrics (dict): A dictionary of metric names and values.
        """
        pass

    @abstractmethod
    def log_dict(self, name: str, obj: Dict[Any, Any]):
        """
        Log a dictionary as an artifact.

        Args:
            name (str): The name of the artifact.
            obj (dict): The dictionary to log.
        """
        pass

    @abstractmethod
    def log_figure(self, fig: Any, name: str):
        """
        Log a figure as an artifact.

        Args:
            fig (matplotlib.figure.Figure): The figure to log.
            name (str): The name of the artifact.
        """
        pass

    @abstractmethod
    def log_model(self, model: Any, artifact_path: str):
        """
        Log a model as an artifact.

        Args:
            model (Any): The model to log.
            artifact_path (str): The path where the model artifact will be saved.
        """
        pass

    @abstractmethod
    def log_artifact(self, artifact: Any, artifact_path: str):
        """
        Log an artifact by serializing it to a file.

        Args:
            artifact (Any): The artifact to log.
            artifact_path (str): The file path to save the artifact.
        """
        pass

    @abstractmethod
    def search_last_run(
        self, run_name: Optional[str] = None, status: str = "FINISHED"
    ) -> Any:
        """
        Search for the last run with the specified name and status.

        Args:
            run_name (str, optional): The name of the run. Defaults to None.
            status (str, optional): The status of the run (e.g., "FINISHED"). Defaults to "FINISHED".

        Returns:
            Any: Information about the last run found.
        """
        pass

    @abstractmethod
    def load_dict(self, artifact_uri: str) -> Dict[Any, Any]:
        """
        Load a dictionary artifact from the specified URI.

        Args:
            artifact_uri (str): The URI of the artifact to load.

        Returns:
            dict: The loaded dictionary.
        """
        pass

    @abstractmethod
    def load_model(self, model_uri: str) -> Any:
        """
        Load a model artifact from the specified URI.

        Args:
            model_uri (str): The URI of the model to load.

        Returns:
            Any: The loaded model.
        """
        pass

    @abstractmethod
    def load_artifact(self, artifact_uri: str) -> Any:
        """
        Load an artifact from the specified URI.

        Args:
            artifact_uri (str): The URI of the artifact to load.

        Returns:
            Any: The loaded artifact.
        """
        pass
