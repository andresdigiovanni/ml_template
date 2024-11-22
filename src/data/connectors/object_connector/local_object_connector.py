import json
import os
import pickle
from typing import Any, Optional

import pandas as pd

from src.utils import Logger

from .base_object_connector import BaseObjectConnector


class LocalObjectConnector(BaseObjectConnector):
    """
    Implementation of ObjectConnector for local file storage.

    Supports multiple file formats based on file extensions, including CSV, JSON, Parquet,
    HTML, text files, and pickle files.
    """

    def __init__(self, base_path: str = "", logger: Optional[Logger] = None) -> None:
        """
        Initializes the local file storage connector.

        Args:
            base_path (str): Base path for all file operations. Defaults to the current directory.
        """
        self.base_path = base_path
        self.logger = logger

        # Mapping of file extensions to their corresponding read/write methods
        self.readers = {
            ".csv": pd.read_csv,
            ".parquet": pd.read_parquet,
            ".json": self._read_json,
            ".txt": self._read_txt,
            ".html": pd.read_html,
            ".pkl": self._read_pickle,
        }
        self.writers = {
            ".csv": lambda data, path: data.to_csv(path, index=False),
            ".parquet": lambda data, path: data.to_parquet(path),
            ".json": self._write_json,
            ".txt": self._write_txt,
            ".html": lambda data, path: data.to_html(path),
            ".pkl": self._write_pickle,
        }

    def _read_json(self, file_path: str) -> Any:
        """
        Reads a JSON file and returns its content.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: Parsed JSON content.
        """
        with open(file_path, "r") as file:
            return json.load(file)

    def _write_json(self, data: Any, file_path: str) -> None:
        """
        Writes data to a JSON file.

        Args:
            data (Any): Data to be written (must be JSON serializable).
            file_path (str): Destination path for the JSON file.
        """
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    def _read_txt(self, file_path: str) -> str:
        """
        Reads a text file and returns its content as a string.

        Args:
            file_path (str): Path to the text file.

        Returns:
            str: Content of the text file.
        """
        with open(file_path, "r") as file:
            return file.read()

    def _write_txt(self, data: str, file_path: str) -> None:
        """
        Writes a string to a text file.

        Args:
            data (str): String content to be written.
            file_path (str): Destination path for the text file.
        """
        with open(file_path, "w") as file:
            file.write(data)

    def _read_pickle(self, file_path: str) -> Any:
        """
        Reads a pickle file and returns its deserialized content.

        Args:
            file_path (str): Path to the pickle file.

        Returns:
            Any: Deserialized object from the pickle file.
        """
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def _write_pickle(self, data: Any, file_path: str) -> None:
        """
        Serializes and writes data to a pickle file.

        Args:
            data (Any): Data to be serialized.
            file_path (str): Destination path for the pickle file.
        """
        with open(file_path, "wb") as file:
            pickle.dump(data, file)

    def get_object(self, object_path: str) -> Any:
        """
        Reads a file from local storage and returns its content.

        The method uses the file extension to determine the appropriate
        reader function.

        Args:
            object_path (str): Path to the file to be read.

        Returns:
            Any: Content of the file (e.g., DataFrame, dict, str, etc.).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        if self.logger:
            self.logger.info(f"Loading object: {object_path}")

        full_path = os.path.join(self.base_path, object_path)
        _, ext = os.path.splitext(full_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")

        if ext not in self.readers:
            raise ValueError(f"Unsupported file extension '{ext}'.")

        return self.readers[ext](full_path)

    def put_object(self, data: Any, object_path: str) -> None:
        """
        Writes data to local storage based on the file extension.

        The method uses the file extension to determine the appropriate
        writer function.

        Args:
            data (Any): Data to be saved (e.g., DataFrame, dict, str, etc.).
            object_path (str): Path to the destination file.

        Raises:
            ValueError: If the file extension is not supported.
        """
        if self.logger:
            self.logger.info(f"Writing object: {object_path}")

        full_path = os.path.join(self.base_path, object_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        _, ext = os.path.splitext(full_path)
        if ext not in self.writers:
            raise ValueError(f"Unsupported file extension '{ext}'.")

        self.writers[ext](data, full_path)
