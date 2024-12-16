from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class FeatureImportanceExplainer:
    """
    A class to generate and visualize feature importance for a model.
    """

    @staticmethod
    def create_figure(
        feature_names: List[str],
        feature_importances: List[float],
        top_n: Optional[int] = None,
    ) -> plt.Figure:
        """
        Generates the feature importance plot for a model.

        Args:
            feature_names (List[str]): List of feature names.
            feature_importances (List[float]): List of feature importances.
            top_n (Optional[int]): Number of top features to select. Selects all if None.

        Returns:
            plt.Figure: The generated figure.
        """
        if len(feature_names) != len(feature_importances):
            raise ValueError(
                "feature_names and feature_importances must have the same length."
            )

        sorted_features, sorted_importances = (
            FeatureImportanceExplainer._get_sorted_features(
                feature_names, feature_importances, top_n
            )
        )
        return FeatureImportanceExplainer._plot_feature_importance(
            sorted_features, sorted_importances
        )

    @staticmethod
    def _get_sorted_features(
        feature_names: List[str],
        feature_importances: List[float],
        top_n: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sorts the features by importance and returns the top_n features.

        Args:
            feature_names (List[str]): List of feature names.
            feature_importances (List[float]): List of feature importances.
            top_n (Optional[int]): Number of top features to select. Selects all if None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sorted feature names and importances.
        """
        indices = np.argsort(feature_importances)[::-1]
        sorted_features = np.array(feature_names)[indices]
        sorted_importances = np.array(feature_importances)[indices]

        if top_n is not None:
            sorted_features = sorted_features[:top_n]
            sorted_importances = sorted_importances[:top_n]

        return sorted_features, sorted_importances

    @staticmethod
    def _plot_feature_importance(
        sorted_features: np.ndarray, sorted_importances: np.ndarray
    ) -> plt.Figure:
        """
        Generates the feature importance plot.

        Args:
            sorted_features (np.ndarray): Sorted feature names.
            sorted_importances (np.ndarray): Sorted feature importances.

        Returns:
            plt.Figure: Figure object containing the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.bar(range(len(sorted_importances)), sorted_importances, align="center")
        ax.set_xticks(range(len(sorted_importances)))
        ax.set_xticklabels(sorted_features, rotation=45, ha="right")
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importances")
        plt.tight_layout()

        return fig
