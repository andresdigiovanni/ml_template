import os

import pandas as pd

from src.data.connectors import BaseObjectConnector
from src.data.preprocessors import DataProcessor
from src.drift import DataMonitoringTool, ModelMonitoringTool
from src.tracking import BaseModelTracker
from src.utils.data import create_predictions_dataframe
from src.utils.logging import Logger


class InferencePipeline:
    def __init__(
        self,
        model_tracker: BaseModelTracker,
        object_connector: BaseObjectConnector,
        logger: Logger,
    ):
        """
        Initializes the inference pipeline by loading the model and preprocessing parameters.

        Args:
            model_tracker (BaseModelTracker): The connector for loading models and artifacts.
            object_connector (BaseObjectConnector): The connector for saving results.
            logger (Logger): The logger for logging messages.
        """
        self.logger = logger
        self.model_tracker = model_tracker
        self.object_connector = object_connector

        # Load the trained model
        self.logger.info("Loading the trained model...")
        run = self.model_tracker.search_last_run(run_name="training")
        artifact_uri = run.info.artifact_uri
        model_uri = f"{artifact_uri}/model.pkl"
        self.model = self.model_tracker.load_model(model_uri)

        # Load preprocessing parameters
        self.logger.info("Loading data processor parameters...")
        processor_uri = f"{artifact_uri}/processor.pkl"
        self.processor_params = self.model_tracker.load_artifact(processor_uri)

        # Initialize the data processor
        self.data_processor = DataProcessor(self.model)
        self.data_processor.set_params(self.processor_params)

        # Cargar el snapshot de los datos de entrenamiento
        self.logger.info("Loading snapshot of training data...")
        train_data = object_connector.get_object(
            os.path.join(
                "training",
                f"{run.info.run_id}_data_snapshot.parquet",
            ),
        )
        train_predictions = object_connector.get_object(
            os.path.join(
                "training",
                f"{run.info.run_id}_predictions_snapshot.parquet",
            ),
        )

        # Inicializa las herramientas de monitoreo
        self.data_monitoring_tool = DataMonitoringTool(train_data, model_tracker)
        self.model_monitoring_tool = ModelMonitoringTool(
            train_predictions, model_tracker
        )

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the inference pipeline, transforming data and generating predictions.

        Args:
            data (pd.DataFrame): The input data for inference.

        Returns:
            pd.DataFrame: A DataFrame containing the original data and predictions.
        """
        with self.model_tracker.run(run_name="inference"):
            # Detectar data drift
            self.logger.info("Checking for data drift...")
            data_drift_results = self.data_monitoring_tool.run_checks(data)
            self.model_tracker.log_params(data_drift_results)

            # Transform the data using the fitted data processor
            self.logger.info("Transforming input data...")
            transformed_data = self.data_processor.transform(data)

            # Generate predictions using the model
            self.logger.info("Generating predictions...")
            predictions = self.model.predict(transformed_data)
            predictions_prob = self.model.predict_proba(transformed_data)
            predictions_df = create_predictions_dataframe(predictions, predictions_prob)

            # Detectar model drift
            self.logger.info("Checking for model drift...")
            model_drift_results = self.model_monitoring_tool.run_checks(predictions_df)
            self.model_tracker.log_params(model_drift_results)

            # Save the predictions and the original data
            self.logger.info("Saving predictions...")
            results = pd.concat([data, predictions_df], axis=1)
            self.object_connector.put_object(
                results,
                os.path.join(
                    "inference", f"{self.model_tracker.run_id()}_predictions.csv"
                ),
            )
