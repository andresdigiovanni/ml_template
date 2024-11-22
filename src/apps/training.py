from sklearn.ensemble import RandomForestClassifier

from src.configs import load_config
from src.data.connectors import create_data_connector, create_object_connector
from src.data.preprocessors import DataProcessor
from src.ml.evaluation import HyperparameterTuner, Metrics
from src.ml.pipelines import TrainingPipeline
from src.utils import Logger


def main():
    try:
        # Load configuration from file
        config = load_config()

        # Initialize the logger
        logger = Logger(
            log_level=config["logging"]["log_level"],
            log_file=config["logging"]["log_file"],
        )

        logger.info("Starting training...")

        # Initialize data connectors and load data
        data_connector = create_data_connector(config["data"]["type"], logger)
        data = data_connector.get_data(config["data"]["source"])

        # Initialize object connector for saving results
        object_connector = create_object_connector(
            config["object_connector"]["type"],
            logger,
            **config["object_connector"]["params"],
        )

        # Configure model
        direction = config["model"]["direction"]
        model = RandomForestClassifier()

        # Initialize data preprocessor and hyperparameter tuner
        data_processor = DataProcessor(model, config["model"]["scoring"], direction)
        metrics = Metrics(config["model"]["problem_type"])
        hyperparameter_tuner = HyperparameterTuner(
            model,
            config["model"]["scoring"],
            direction,
        )

        # Create and configure the training pipeline
        pipeline = TrainingPipeline(
            model,
            metrics,
            hyperparameter_tuner,
            data_processor,
            object_connector,
            logger,
        )

        # Run the training pipeline
        pipeline.run(data=data, target_column=config["data"]["target_column"])

        logger.info("Training pipeline execution completed successfully.")

    except Exception as e:
        logger.critical(f"Error during training pipeline execution: {e}")


if __name__ == "__main__":
    main()
