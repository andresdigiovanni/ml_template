from typing import Union

import numpy as np
import pandas as pd


def subset_data(
    data: Union[pd.DataFrame, pd.Series, np.ndarray], indices: np.ndarray
) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Extract a subset of data using the given indices.

    Args:
        data (Union[pd.DataFrame, pd.Series, np.ndarray]): Input data.
        indices (np.ndarray): Indices to extract.

    Returns:
        Union[pd.DataFrame, pd.Series, np.ndarray]: Extracted subset of data.

    Raises:
        TypeError: If the data type is not compatible.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[indices]

    if isinstance(data, np.ndarray):
        return data[indices]

    raise TypeError(f"Unsupported data type: {type(data)}")
