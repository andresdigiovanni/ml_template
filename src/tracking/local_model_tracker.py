import json
import os
import pickle
import secrets
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime

from .base_model_tracker import BaseModelTracker


class LocalModelTracker(BaseModelTracker):
    """
    Local implementation of the BaseModelTracker interface.

    This class logs experiment data, parameters, metrics, figures, and models
    to the local file system, allowing easy tracking of machine learning experiments.
    """

    ROOT_EXPERIMENTS_PATH = ".experiments"

    def __init__(self, experiment_name):
        """
        Initialize the LocalModelTracker with an experiment name.

        Args:
            experiment_name (str): The name of the experiment.
        """
        self._experiment_name = experiment_name
        self._experiment_path = [self.ROOT_EXPERIMENTS_PATH, experiment_name]

        self._params = {}
        self._metrics = {}
        self._run_id = self._generate_run_id()

    def _generate_run_id(self):
        """
        Generate a unique run ID using a random 24-character hex string.

        Returns:
            str: The generated run ID.
        """
        length = 24
        length_bytes = (length + 1) // 2
        bytes_rnd = secrets.token_bytes(length_bytes)
        id_hex = bytes_rnd.hex()
        return id_hex[:length]

    def run_id(self):
        """
        Retrieve the current run ID.

        Returns:
            str: The current run ID.
        """
        return self._run_id

    @contextmanager
    def run(self, run_name=None, nested=False):
        """
        Context manager for managing the lifecycle of a run.

        Args:
            run_name (str, optional): The name of the run. Defaults to None.
            nested (bool, optional): Whether the run is nested within a parent run. Defaults to False.
        """
        try:
            self.start_run(run_name, nested)
            yield
        finally:
            self.end_run()

    def start_run(self, run_name=None, nested=False):
        """
        Start a new run, creating a directory for the run and initializing logs.

        Args:
            run_name (str, optional): The name of the run. Defaults to None.
            nested (bool, optional): Whether the run is nested within a parent run. Defaults to False.
        """
        if not nested:
            self._experiment_path = [self.ROOT_EXPERIMENTS_PATH, self._experiment_name]

        run_path = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        run_path += f"__{run_name}" if run_name else ""

        self._experiment_path.append(run_path)

        experiment_path = os.path.join(*self._experiment_path)
        os.makedirs(experiment_path, exist_ok=True)

        run_id_path = os.path.join(*self._experiment_path, "run_id.json")
        with open(run_id_path, "w") as f:
            json.dump({"run_id": self._run_id}, f, indent=4)

        self._params = {}
        self._metrics = {}

    def end_run(self):
        """
        End the current run and save parameters and metrics to files.
        """
        self.log_dict(self._params, "params.json")
        self.log_dict(self._metrics, "metrics.json")

        self._experiment_path = self._experiment_path[:-1]

    def log_param(self, name, value):
        """
        Log a single parameter.

        Args:
            name (str): The parameter name.
            value (Any): The parameter value.
        """
        self._params[name] = value

    def log_params(self, params):
        """
        Log multiple parameters.

        Args:
            params (dict): A dictionary of parameters to log.
        """
        self._params.update(params)

    def log_metric(self, name, value):
        """
        Log a single metric.

        Args:
            name (str): The metric name.
            value (float): The metric value.
        """
        self._metrics[name] = value

    def log_metrics(self, metrics):
        """
        Log multiple metrics.

        Args:
            metrics (dict): A dictionary of metrics to log.
        """
        for name, value in metrics.items():
            self.log_metric(name, value)

    def log_dict(self, obj, name):
        """
        Log a dictionary as a JSON file.

        Args:
            name (str): The name of the file to save.
            obj (dict): The dictionary to log.
        """
        dict_path = os.path.join(*self._experiment_path, name)
        with open(dict_path, "w") as f:
            json.dump(obj, f, indent=4)

    def log_figure(self, fig, name):
        """
        Log a matplotlib figure as an image file.

        Args:
            fig (matplotlib.figure.Figure): The figure to log.
            name (str): The filename to save the figure as.
        """
        img_path = os.path.join(*self._experiment_path, name)
        fig.savefig(img_path, format="png")

    def log_model(self, model, artifact_path):
        """
        Log a model by serializing it to a file.

        Args:
            model (Any): The model to log.
            artifact_path (str): The file path to save the model.
        """
        model_path = os.path.join(*self._experiment_path, artifact_path)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    def log_artifact(self, artifact, artifact_path):
        """
        Log an artifact by serializing it to a file.

        Args:
            artifact (Any): The artifact to log.
            artifact_path (str): The file path to save the artifact.
        """
        artifact_path = os.path.join(*self._experiment_path, artifact_path)

        if artifact_path.endswith(".html"):
            with open(artifact_path, "w", encoding="utf-8") as f:
                f.write(artifact)
        else:
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)

    def search_last_run(self, run_name=None, status="FINISHED"):
        """
        Search for the last run in the current experiment path.

        Args:
            run_name (str, optional): The run name to search for. Defaults to None.
            status (str, optional): The status of the run (e.g., "FINISHED"). Defaults to "FINISHED".

        Returns:
            Namespace: A namespace containing the artifact URI and run ID.
        """
        folder_path = os.path.join(*self._experiment_path)
        subfolders = [
            f.path
            for f in os.scandir(folder_path)
            if f.is_dir() and f.path.endswith(f"__{run_name}" if run_name else "")
        ]
        subfolders.sort(reverse=True)

        artifact_uri = os.path.join(subfolders[0])

        run_id_path = os.path.join(artifact_uri, "run_id.json")
        with open(run_id_path, "r") as f:
            run_id = json.load(f)

        return Namespace(
            info=Namespace(
                artifact_uri=artifact_uri,
                run_id=run_id["run_id"],
            )
        )

    def load_dict(self, artifact_uri):
        """
        Load a dictionary from a JSON file.

        Args:
            artifact_uri (str): The path to the artifact.

        Returns:
            dict: The loaded dictionary.
        """
        with open(artifact_uri, "r") as f:
            return json.load(f)

    def load_model(self, model_uri):
        """
        Load a model from a serialized file.

        Args:
            model_uri (str): The path to the serialized model file.

        Returns:
            Any: The loaded model.
        """
        with open(model_uri, "rb") as f:
            return pickle.load(f)

    def load_artifact(self, artifact_uri):
        """
        Load an artifact from the specified URI.

        Args:
            artifact_uri (str): The URI of the artifact to load.

        Returns:
            Any: The loaded artifact.
        """
        with open(artifact_uri, "rb") as f:
            return pickle.load(f)
