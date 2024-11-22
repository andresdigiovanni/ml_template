from src.configs import load_config
from src.data.connectors import create_data_connector, create_object_connector
from src.data.preprocessors import DataProcessor
from src.ml.pipelines import InferencePipeline
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

        logger.info("Starting inference...")

        # Initialize data connector and load data
        data_connector = create_data_connector(config["data"]["type"], logger)
        data = data_connector.get_data(config["data"]["source"])

        # Drop target column from the data as it's not needed for inference
        target_column = config["data"]["target_column"]
        data = data.drop(columns=[target_column])

        # Initialize object connector and load the trained model
        object_connector = create_object_connector(
            config["object_connector"]["type"],
            logger,
            **config["object_connector"]["params"],
        )
        model = object_connector.get_object("model.pkl")

        # Initialize the data processor
        direction = config["model"]["direction"]
        data_processor = DataProcessor(model, config["model"]["scoring"], direction)

        # Create and configure the inference pipeline
        pipeline = InferencePipeline(model, data_processor, object_connector, logger)

        # Run the inference pipeline to generate predictions
        pipeline.run(data)

        logger.info("Inference pipeline execution completed successfully.")

    except Exception as e:
        logger.critical(f"Error during inference pipeline execution: {e}")


if __name__ == "__main__":
    main()
