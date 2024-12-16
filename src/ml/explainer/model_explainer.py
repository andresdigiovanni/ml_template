from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import shap


class ModelExplainer:
    """
    A class to generate model explainability visualizations using SHAP (SHapley Additive exPlanations).

    Args:
        model (Any): The machine learning model to explain.
        data (pd.DataFrame): The dataset used to generate explanations.
    """

    def __init__(self, model: Any, data: pd.DataFrame) -> None:
        self.data = data
        self.explainer = shap.Explainer(model)

    def _generate_plot(self, plot_func: Any, *args, **kwargs) -> plt.Figure:
        """
        Generates a SHAP plot and returns the figure.

        Args:
            plot_func (Any): SHAP function used to generate the plot.
            *args: Positional arguments for the SHAP plotting function.
            **kwargs: Additional keyword arguments for the SHAP plotting function.

        Returns:
            plt.Figure: The generated figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_func(*args, **kwargs)
        plt.close(fig)
        return fig

    def generate_summary_plot(
        self, num_samples: int = 100
    ) -> List[Tuple[str, plt.Figure]]:
        """
        Generates global explanations and returns SHAP summary plots.
        Handles both binary and multi-class classification cases.

        Args:
            num_samples (int): The number of samples to compute SHAP explanations.

        Returns:
            List[Tuple[str, plt.Figure]]: A list of tuples containing the filename and the corresponding figure.
        """
        shap_values = self.explainer(self.data[:num_samples])
        figures = []

        if len(shap_values.values.shape) == 3:
            # Multi-class classification case
            num_classes = shap_values.values.shape[2]

            for class_index in range(num_classes):
                fig = self._generate_plot(
                    shap.summary_plot,
                    shap_values[:, :, class_index],
                    self.data[:num_samples],
                    show=False,
                )
                figures.append((f"shap_summary_plot_class_{class_index}.png", fig))

        else:
            # Binary classification or regression case
            fig = self._generate_plot(
                shap.summary_plot,
                shap_values,
                self.data[:num_samples],
                show=False,
            )
            figures.append(("shap_summary_plot.png", fig))

        return figures
