from sklearn.ensemble import RandomForestClassifier

from src.configs import load_config
from src.data.connectors import create_data_connector, create_object_connector
from src.ml.pipelines import TrainingPipeline
from src.tracking import create_model_tracker
from src.utils.logging import Logger


def main():
    """Main function to execute the training pipeline.

    Loads configuration, initializes required components, and executes the
    training pipeline to train a model and save the results.

    Raises:
        Exception: If any error occurs during pipeline execution.
    """
    try:
        # Load configuration
        config = load_config()

        # Initialize the logger
        logger = Logger(
            log_level=config["logging"]["log_level"],
            log_file=config["logging"]["log_file"],
        )
        logger.info("Starting training...")

        # Initialize data connectors and load data
        data_connector = create_data_connector(
            connector_type=config["data"]["type"],
            logger=logger,
        )
        data = data_connector.get_data(source=config["data"]["source"])

        # Initialize model tracker for saving results
        model_tracker = create_model_tracker(
            tracker_type=config["tracking"]["type"],
            experiment_name=config["tracking"]["experiment_name"],
        )

        # Initialize object connector for saving data snapshots
        object_connector = create_object_connector(
            connector_type=config["object_connector"]["type"],
            logger=logger,
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
        target_column = config["data"]["target_column"]
        if target_column not in data.columns:
            logger.error(f"Target column '{target_column}' not found in data.")
            raise ValueError(f"Target column '{target_column}' not found in data.")

        pipeline.run(data=data, target_column=target_column)

        logger.info("Training pipeline execution completed successfully.")

    except Exception as error:
        logger.critical(f"Error during training pipeline execution: {error}")
        raise


if __name__ == "__main__":
    main()
