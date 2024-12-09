import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from src.data.connectors import BaseObjectConnector
from src.data.preprocessors import DataProcessor
from src.ml.evaluation import (
    ConfusionMatrixFigure,
    HyperparameterTuner,
    RocAucFigure,
    compute_metrics,
)
from src.utils import Logger


class TrainingPipeline:
    """
    A class to manage the complete machine learning training pipeline,
    including preprocessing, hyperparameter tuning, model training,
    evaluation, and saving the model and results.

    Attributes:
        model: The machine learning model to train.
        metrics: A metrics evaluator to compute model performance.
        hyperparameter_tuner: A hyperparameter optimization tool.
        data_processor: A data transformation and preprocessing tool.
        object_connector: An object storage connector for saving models and results.
    """

    def __init__(
        self,
        model: BaseEstimator,
        hyperparameter_tuner: HyperparameterTuner,
        data_processor: DataProcessor,
        object_connector: BaseObjectConnector,
        logger: Logger,
        problem_type: str,
    ):
        """
        Initializes the pipeline with necessary components.

        Args:
            model (BaseEstimator): The machine learning model to be used.
            hyperparameter_tuner (HyperparameterTuner): The tuner to optimize hyperparameters.
            data_processor (DataProcessor): The processor to handle data transformations.
            object_connector (BaseObjectConnector): The connector for saving models and results.
            logger (Logger): The logger for logging messages.
            problem_type (str): The type of problem. Either 'classification' or 'regression'.
        """
        self.model = model
        self.hyperparameter_tuner = hyperparameter_tuner
        self.data_processor = data_processor
        self.object_connector = object_connector
        self.logger = logger
        self.problem_type = problem_type

    def run(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """
        Runs the full training pipeline, including data preprocessing, hyperparameter tuning,
        model training, evaluation, and saving the model and results.

        Args:
            data (pd.DataFrame): The input dataset.
            target_column (str): The name of the target column.
            test_size (float): The fraction of data to use as test data.
            random_state (int): The random seed for reproducibility.

        Returns:
            dict: A dictionary containing the evaluation results.
        """

        # Split the data into training and test sets
        self.logger.info("Splitting data into training and test sets...")
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Preprocess the training and test data
        self.logger.info("Processing training data...")
        X_train = self.data_processor.fit_transform(X_train, y_train)
        X_test = self.data_processor.transform(X_test)
        processor_params = self.data_processor.get_params()

        # Hyperparameter tuning
        self.logger.info("Starting hyperparameter tuning...")
        best_params = self.hyperparameter_tuner.fit(X_train, y_train)

        # Train the model with the best hyperparameters
        self.logger.info("Training the model with the best parameters...")
        self.model.set_params(**best_params)
        self.model.fit(X_train, y_train)

        # Evaluate the model on the test data
        self.logger.info("Evaluating the model...")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)

        metrics = compute_metrics(self.model, X_test, y_test, self.problem_type)

        roc_auc_figure = RocAucFigure.create_figure(y_test, y_prob, self.model.classes_)
        cm_figure = ConfusionMatrixFigure.create_figure(
            y_test, y_pred, self.model.classes_
        )

        # Save the model, metrics, and parameters
        self.object_connector.put_object(self.model, "model.pkl")
        self.object_connector.put_object(metrics, "metrics.json")
        self.object_connector.put_object(roc_auc_figure, "roc_auc_plot.png")
        self.object_connector.put_object(cm_figure, "confusion_matrix.png")
        self.object_connector.put_object(best_params, "params.json")
        self.object_connector.put_object(processor_params, "processor.pkl")

        return metrics
