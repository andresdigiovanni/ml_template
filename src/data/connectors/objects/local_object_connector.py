import json
import os
import pickle
from typing import Any, Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging import Logger

from .base_object_connector import BaseObjectConnector


class LocalObjectConnector(BaseObjectConnector):
    """
    Implementation of ObjectConnector for local file storage.

    Supports multiple file formats based on file extensions, including CSV, JSON, Parquet,
    HTML, text files, pickle files, and image files (PNG, JPG, JPEG).
    """

    def __init__(self, base_path: str = "", logger: Optional[Logger] = None) -> None:
        """
        Initializes the local file storage connector.

        Args:
            base_path (str): Base path for all file operations. Defaults to the current directory.
            logger (Optional[Logger]): Logger instance for logging operations.
        """
        self.base_path = base_path
        self.logger = logger

        # Mapping of file extensions to their corresponding read/write methods
        self.readers = {
            ".csv": pd.read_csv,
            ".parquet": pd.read_parquet,
            ".json": self._read_json,
            ".txt": self._read_txt,
            ".html": self._read_html,
            ".pkl": self._read_pickle,
        }
        self.writers = {
            ".csv": lambda data, path: data.to_csv(path, index=False),
            ".parquet": lambda data, path: data.to_parquet(path),
            ".json": self._write_json,
            ".txt": self._write_txt,
            ".html": self._write_html,
            ".pkl": self._write_pickle,
            ".png": self._write_image,
            ".jpg": self._write_image,
            ".jpeg": self._write_image,
        }

    def _log(self, message: str) -> None:
        """Logs a message if a logger is provided."""
        if self.logger:
            self.logger.info(message)

    def _read_json(self, file_path: str) -> Any:
        with open(file_path, "r") as file:
            return json.load(file)

    def _write_json(self, data: Any, file_path: str) -> None:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    def _read_txt(self, file_path: str) -> str:
        with open(file_path, "r") as file:
            return file.read()

    def _write_txt(self, data: str, file_path: str) -> None:
        with open(file_path, "w") as file:
            file.write(data)

    def _read_pickle(self, file_path: str) -> Any:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def _write_pickle(self, data: Any, file_path: str) -> None:
        with open(file_path, "wb") as file:
            pickle.dump(data, file)

    def _write_image(self, data: matplotlib.figure.Figure, file_path: str) -> None:
        if not isinstance(data, matplotlib.figure.Figure):
            raise ValueError("Data must be a matplotlib Figure object for image files.")
        data.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(data)

    def _read_html(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def _write_html(self, data: str, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            file.write(data)

    def get_object(self, object_path: str) -> Any:
        full_path = os.path.join(self.base_path, object_path)
        _, ext = os.path.splitext(full_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")

        if ext not in self.readers:
            raise ValueError(f"Unsupported file extension '{ext}'.")

        self._log(f"Loading object: {object_path}")
        return self.readers[ext](full_path)

    def put_object(self, data: Any, object_path: str) -> None:
        full_path = os.path.join(self.base_path, object_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        _, ext = os.path.splitext(full_path)
        if ext not in self.writers:
            raise ValueError(f"Unsupported file extension '{ext}'.")

        self._log(f"Writing object: {object_path}")
        self.writers[ext](data, full_path)

    def delete_object(self, object_path: str) -> None:
        full_path = os.path.join(self.base_path, object_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")

        self._log(f"Deleting object: {object_path}")
        os.remove(full_path)
