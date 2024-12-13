from sklearn.ensemble import RandomForestClassifier

from src.configs import load_config
from src.data.connectors import create_data_connector, create_object_connector
from src.ml.pipelines import TrainingPipeline
from src.tracking import create_model_tracker
from src.utils.logging import Logger


def main():
    try:
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

        # Initialize model tracker for saving results
        model_tracker = create_model_tracker(
            config["tracking"]["type"], config["tracking"]["experiment_name"]
        )

        # Initialize object connector for saving data snapshots
        object_connector = create_object_connector(
            config["object_connector"]["type"],
            logger,
            **config["object_connector"]["params"],
        )

        # Create and configure the training pipeline
        pipeline = TrainingPipeline(
            model=RandomForestClassifier(),
            model_tracker=model_tracker,
            object_connector=object_connector,
            logger=logger,
            config=config,
        )

        # Run the training pipeline
        pipeline.run(data, config["data"]["target_column"])

        logger.info("Training pipeline execution completed successfully.")

    except Exception as e:
        logger.critical(f"Error during training pipeline execution: {e}")


if __name__ == "__main__":
    main()
