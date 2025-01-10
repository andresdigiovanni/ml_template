import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.test_preset import DataDriftTestPreset

from .shared import generate_and_log_report, generate_and_log_test


class ModelMonitoringTool:
    """
    A class to monitor model drift by comparing predictions on reference and current data.

    Attributes:
        reference_predictions (pd.DataFrame): Predictions made on the reference data.
        model_tracker: An object for logging reports and artifacts (e.g., MLflow).
    """

    def __init__(self, reference_predictions: pd.DataFrame, model_tracker):
        """
        Initializes the ModelMonitoringTool.

        Args:
            reference_predictions (pd.DataFrame): The predictions made on the reference data.
            model_tracker: The model tracker for logging reports and artifacts.
        """
        if reference_predictions.empty:
            raise ValueError("Reference predictions cannot be empty.")

        self.reference_predictions = reference_predictions
        self.model_tracker = model_tracker

    def run_checks(self, current_predictions: pd.DataFrame) -> dict:
        """
        Compares the predictions made on the reference data and the current data to detect model drift.

        Args:
            current_predictions (pd.DataFrame): The predictions made by the model on the current data.

        Returns:
            dict: A summary of the drift test results.

        Raises:
            ValueError: If current_predictions is empty or columns do not match reference_predictions.
        """
        if current_predictions.empty:
            raise ValueError("Current predictions cannot be empty.")

        # Ensure both datasets have the same columns
        common_columns = self.reference_predictions.columns.intersection(
            current_predictions.columns
        )

        if common_columns.empty:
            raise ValueError(
                "No matching columns between reference and current predictions."
            )

        reference_predictions = self.reference_predictions[common_columns]
        current_predictions = current_predictions[common_columns]

        drift_results = {}

        # Generate and log the drift report
        generate_and_log_report(
            self.model_tracker,
            "model_drift",
            DataDriftPreset(),
            reference_predictions,
            current_predictions,
        )

        # Generate and log the drift test
        drift_results["model_drift"] = generate_and_log_test(
            self.model_tracker,
            "model_drift",
            DataDriftTestPreset(stattest="psi"),
            reference_predictions,
            current_predictions,
        )

        return drift_results
