from src.configs import load_config
from src.data.connectors import create_data_connector, create_object_connector
from src.ml.pipelines import InferencePipeline
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

        logger.info("Starting inference...")

        # Initialize data connector and load data
        data_connector = create_data_connector(config["data"]["type"], logger)
        data = data_connector.get_data(config["data"]["source"])

        # Drop target column from the data as it's not needed for inference
        target_column = config["data"]["target_column"]
        data = data.drop(columns=[target_column])

        # Initialize object connector
        object_connector = create_object_connector(
            config["object_connector"]["type"],
            logger,
            **config["object_connector"]["params"],
        )

        # Initialize model tracker
        model_tracker = create_model_tracker(
            config["tracking"]["type"], config["tracking"]["experiment_name"]
        )

        # Create and configure the inference pipeline
        pipeline = InferencePipeline(
            model_tracker=model_tracker,
            object_connector=object_connector,
            logger=logger,
        )

        # Run the inference pipeline to generate predictions
        pipeline.run(data)

        logger.info("Inference pipeline execution completed successfully.")

    except Exception as e:
        logger.critical(f"Error during inference pipeline execution: {e}")


if __name__ == "__main__":
    main()
