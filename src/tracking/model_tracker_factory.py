from .base_model_tracker import BaseModelTracker
from .local_model_tracker import LocalModelTracker


def create_model_tracker(tracker_type: str, experiment_name: str) -> BaseModelTracker:
    """
    Creates and returns a model tracker instance based on the specified connector type.

    Args:
        tracker_type (str): The type of connector to use for the model tracker.
        experiment_name (str): Additional keyword arguments to initialize the model tracker.

    Returns:
        BaseModelTracker: An instance of the selected connector.

    Raises:
        ValueError: If the provided `tracker_type` is not supported.
    """
    TRACKER_REGISTRY = {
        "local": LocalModelTracker,
    }

    if tracker_type in TRACKER_REGISTRY:
        return TRACKER_REGISTRY[tracker_type](experiment_name)

    else:
        error_message = f"Connector type '{tracker_type}' is not supported."
        raise ValueError(error_message)
