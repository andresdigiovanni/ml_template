import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_preset import (
    DataDriftTestPreset,
    DataQualityTestPreset,
    DataStabilityTestPreset,
)

from .shared import generate_and_log_report, generate_and_log_test


class DataMonitoringTool:
    """
    A class to encapsulate data drift, data quality, and data stability checks using Evidently.

    Attributes:
        train_data (pd.DataFrame): The training data used as a reference.
        model_tracker: An object responsible for tracking and logging results (e.g., MLflow).
    """

    def __init__(self, train_data: pd.DataFrame, model_tracker):
        """
        Initializes the DataDriftChecker with training data and a model tracker.

        Args:
            train_data (pd.DataFrame): The training data used as the reference dataset.
            model_tracker: The model tracker for logging reports and artifacts.
        """
        self.train_data = train_data
        self.model_tracker = model_tracker

    def run_checks(self, current_data: pd.DataFrame) -> dict:
        """
        Checks for data drift, data quality, and data stability between the training data and current data.

        Args:
            current_data (pd.DataFrame): The current data used for inference.

        Returns:
            dict: A summary of the drift, quality, and stability test results.
        """
        drift_results = {}

        # Ensure both datasets have the same columns
        columns_to_compare = self.train_data.columns.intersection(current_data.columns)
        reference_data = self.train_data[columns_to_compare]
        current_data = current_data[columns_to_compare]

        # Run and log data drift report
        generate_and_log_report(
            self.model_tracker,
            "data_drift",
            DataDriftPreset(),
            reference_data,
            current_data,
        )

        # Run and log data quality report
        generate_and_log_report(
            self.model_tracker,
            "data_quality",
            DataQualityPreset(),
            reference_data,
            current_data,
        )

        # Run and log data stability test
        drift_results["data_stability"] = generate_and_log_test(
            self.model_tracker,
            "data_stability",
            DataStabilityTestPreset(),
            reference_data,
            current_data,
        )

        # Run and log data quality test
        drift_results["data_quality"] = generate_and_log_test(
            self.model_tracker,
            "data_quality",
            DataQualityTestPreset(),
            reference_data,
            current_data,
        )

        # Run and log data drift test
        drift_results["data_drift"] = generate_and_log_test(
            self.model_tracker,
            "data_drift",
            DataDriftTestPreset(stattest="psi"),
            reference_data,
            current_data,
        )

        return drift_results
