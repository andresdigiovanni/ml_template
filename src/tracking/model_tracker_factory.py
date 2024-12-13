from .base_model_tracker import BaseModelTracker
from .local_model_tracker import LocalModelTracker


def create_model_tracker(connector_type: str, experiment_name: str) -> BaseModelTracker:
    """
    Creates and returns a model tracker instance based on the specified connector type.

    Args:
        connector_type (str): The type of connector to use for the model tracker.
        experiment_name (str): Additional keyword arguments to initialize the model tracker.

    Returns:
        BaseModelTracker: An instance of the selected connector.

    Raises:
        ValueError: If the provided `connector_type` is not supported.
    """
    if connector_type == "local":
        return LocalModelTracker(experiment_name)

    else:
        raise ValueError(f"Connector type '{connector_type}' is not supported.")
