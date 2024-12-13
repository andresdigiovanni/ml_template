import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.test_preset import DataDriftTestPreset

from .shared import generate_and_log_report, generate_and_log_test


class ModelMonitoringTool:
    def __init__(self, reference_predictions: pd.DataFrame, model_tracker):
        """
        Initializes the ModelDriftMonitor.

        Args:
            reference_predictions (pd.DataFrame): The predictions made on the reference data.
            model_tracker: The model tracker for logging reports and artifacts.
        """
        self.reference_predictions = reference_predictions
        self.model_tracker = model_tracker

    def run_checks(self, current_predictions: pd.DataFrame) -> dict:
        """
        Compares the predictions made on the reference data and the current data to detect model drift.

        Args:
            current_predictions (pd.DataFrame): The predictions made by the model on the current data.
        """
        drift_results = {}

        # Generate and log the drift report for the reference vs. current data
        generate_and_log_report(
            self.model_tracker,
            "model_drift",
            DataDriftPreset(),
            self.reference_predictions,
            current_predictions,
        )
        drift_results["model_drift"] = generate_and_log_test(
            self.model_tracker,
            "model_drift",
            DataDriftTestPreset(stattest="psi"),
            self.reference_predictions,
            current_predictions,
        )

        return drift_results
