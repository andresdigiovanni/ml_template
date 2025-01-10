from src.configs import load_config
from src.data.connectors import create_data_connector, create_object_connector
from src.ml.pipelines import InferencePipeline
from src.tracking import create_model_tracker
from src.utils.logging import Logger


def main():
    """Main function to execute the inference pipeline.

    Loads configuration, initializes required components, and executes the
    inference pipeline to generate predictions.

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
        logger.info("Starting inference...")

        # Initialize the data connector and load the data
        data_connector = create_data_connector(
            connector_type=config["data"]["type"],
            logger=logger,
        )
        data = data_connector.get_data(source=config["data"]["source"])

        # Drop the target column as it's not needed for inference
        target_column = config["data"]["target_column"]
        if target_column in data.columns:
            data = data.drop(columns=[target_column])
        else:
            logger.warning(f"Target column '{target_column}' not found in data.")

        # Initialize the object connector
        object_connector = create_object_connector(
            connector_type=config["object_connector"]["type"],
            logger=logger,
            **config["object_connector"]["params"],
        )

        # Initialize the model tracker
        model_tracker = create_model_tracker(
            tracker_type=config["tracking"]["type"],
            experiment_name=config["tracking"]["experiment_name"],
        )

        # Create and configure the inference pipeline
        pipeline = InferencePipeline(
            model_tracker=model_tracker,
            object_connector=object_connector,
            logger=logger,
        )

        # Run the inference pipeline to generate predictions
        pipeline.run(data=data)

        logger.info("Inference pipeline execution completed successfully.")

    except Exception as error:
        logger.critical(f"Error during inference pipeline execution: {error}")
        raise


if __name__ == "__main__":
    main()
