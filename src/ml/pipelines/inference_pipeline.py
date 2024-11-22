import pandas as pd
from sklearn.base import BaseEstimator

from src.data.connectors import BaseObjectConnector
from src.data.preprocessors import DataProcessor
from src.utils import Logger


class InferencePipeline:
    """
    A class to manage the inference pipeline, which handles the data processing,
    prediction generation, and saving the results.

    Attributes:
        model: The trained machine learning model to make predictions.
        data_processor: A data processor to handle the transformations on input data.
        object_connector: An object storage connector for retrieving models and saving results.
    """

    def __init__(
        self,
        model: BaseEstimator,
        data_processor: DataProcessor,
        object_connector: BaseObjectConnector,
        logger: Logger,
    ):
        """
        Initializes the inference pipeline with necessary components.

        Args:
            model (BaseEstimator): The trained machine learning model to be used for predictions.
            data_processor (DataProcessor): The processor for transforming input data.
            object_connector (BaseObjectConnector): The connector for retrieving and saving objects.
            logger (Logger): The logger for logging messages.
        """
        self.model = model
        self.data_processor = data_processor
        self.object_connector = object_connector
        self.logger = logger

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the inference pipeline, which involves retrieving preprocessing parameters,
        transforming input data, generating predictions, and saving the results.

        Args:
            data (pd.DataFrame): The input data to make predictions on.

        Returns:
            pd.DataFrame: A DataFrame containing the original data and the predicted values.
        """
        # Retrieve and set the data processing parameters
        processor_params = self.object_connector.get_object("processor.pkl")
        self.data_processor.set_params(processor_params)

        # Transform the data using the fitted data processor
        self.logger.info("Transforming input data...")
        transformed_data = self.data_processor.transform(data)

        # Generate predictions using the model
        self.logger.info("Generating predictions...")
        predictions = self.model.predict(transformed_data)

        # Save the predictions and the original data
        results = data.copy()
        results["prediction"] = predictions

        # Store results in a file
        self.object_connector.put_object(results, "predictions.csv")

        return results
