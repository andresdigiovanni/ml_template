from collections import defaultdict
from typing import Dict, List


def average_metrics(scores: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Averages a list of metric dictionaries.

    Args:
        scores (List[Dict[str, float]]): A list of dictionaries where each dictionary
            contains metric names as keys and their respective scores as values.

    Returns:
        Dict[str, float]: A dictionary with the averaged metrics.
    """
    results = defaultdict(float)
    for metrics in scores:
        for key in metrics:
            results[key] += metrics[key]

    return {key: value / len(scores) for key, value in results.items()}
