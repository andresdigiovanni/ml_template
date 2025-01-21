from src.configs import load_config
from src.data.connectors import create_data_connector
from src.models import ModelSelection


def main():
    # Load configuration
    config = load_config()

    # Initialize the data connector and load the data
    data_connector = create_data_connector(connector_type=config["data"]["type"])
    data = data_connector.get_data(source=config["data"]["source"])

    # Drop the target column as it's not needed for inference
    target_column = config["data"]["target_column"]
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Run model selector
    selector = ModelSelection(X, y, task=config["model"]["problem_type"])
    models = selector.evaluate(verbose=True)
    print(models)

    best_model = models["Accuracy"].idxmax()
    print(f"\nBest model: {best_model}")

    cls = next(x for x in selector.models if x.__class__.__name__ == best_model)
    print(cls)


if __name__ == "__main__":
    main()
