import argparse
import sys
from typing import Any, Dict

import yaml


def load_config(file_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from a YAML file and merges it with command-line arguments.

    This function reads a base configuration file in YAML format, parses any command-line
    overrides provided in the form of key-value pairs (e.g., `key.subkey=value`), and
    merges them recursively.

    Args:
        file_path (str): Path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        Dict[str, Any]: A dictionary containing the merged configuration.
    """
    base_config = _load_yaml(file_path)
    args_config = _parse_command_line_args()
    return _merge_dicts(base_config, args_config)


def _load_yaml(file_path: str) -> Dict[str, Any]:
    """Loads a YAML file into a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: The loaded YAML content as a dictionary.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file) or {}

    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file '{file_path}' was not found.")

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file '{file_path}': {e}")


def _parse_command_line_args() -> Dict[str, Any]:
    """Parses command-line arguments and converts them into a nested dictionary.

    Command-line arguments should be provided as key-value pairs, where nested keys are
    separated by dots (e.g., `key.subkey=value`).

    Returns:
        Dict[str, Any]: Parsed arguments as a nested dictionary.
    """
    parser = argparse.ArgumentParser(
        description="Command-line argument parser for overriding configuration."
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Key-value pairs for configuration (e.g., key.subkey=value).",
    )
    args, _ = parser.parse_known_args(sys.argv)

    nested_config = {}
    for arg in args.args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            _set_nested_value(nested_config, key.split("."), value)

    return nested_config


def _set_nested_value(dictionary: Dict[str, Any], keys: list[str], value: Any) -> None:
    """Sets a value in a nested dictionary based on a list of keys.

    Args:
        dictionary (Dict[str, Any]): The dictionary to modify.
        keys (list[str]): A list of keys representing the path to the value.
        value (Any): The value to set.
    """
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})

    dictionary[keys[-1]] = value


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges two dictionaries, prioritizing override values.

    Args:
        base (Dict[str, Any]): The base dictionary.
        overrides (Dict[str, Any]): The dictionary with override values.

    Returns:
        Dict[str, Any]: The merged dictionary.
    """
    merged = base.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merged[key] = _merge_dicts(base[key], value)
        else:
            merged[key] = value

    return merged
