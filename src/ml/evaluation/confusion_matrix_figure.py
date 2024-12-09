import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ConfusionMatrixFigure:
    """
    Class to generate a matplotlib.figure.Figure object with a confusion matrix.
    Supports binary and multi-class classification problems.

    Methods:
        create_figure(y_true, y_pred, model_classes, normalize=False, title="Confusion Matrix"):
            Generates the confusion matrix plot.
    """

    @staticmethod
    def create_figure(
        y_true, y_pred, model_classes, normalize=False, title="Confusion Matrix"
    ):
        """
        Creates a matplotlib.figure.Figure object with a confusion matrix.

        Args:
            y_true (array-like): Ground truth target values.
            y_pred (array-like): Predicted target values.
            model_classes (array-like): List of class labels in the model.
            normalize (bool): Whether to normalize the confusion matrix.
            title (str): Title of the plot.

        Returns:
            matplotlib.figure.Figure: Figure object containing the plot.
        """
        # Compute the confusion matrix
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=model_classes,
            normalize="true" if normalize else None,
        )

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the confusion matrix as a heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=model_classes,
            yticklabels=model_classes,
            cbar=True,
            ax=ax,
        )
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title(title)

        return fig
