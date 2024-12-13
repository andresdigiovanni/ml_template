import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


class RocAucFigure:
    """
    Class to generate ROC-AUC plots for binary and multiclass classification problems.
    """

    @staticmethod
    def create_figure(y_true, y_prob, model_classes, title="ROC Curve"):
        """
        Creates a matplotlib.figure.Figure object with ROC curves and AUC values.

        Args:
            y_true (array-like): Ground truth target values.
            y_prob (array-like): Predicted probabilities for each class.
            model_classes (array-like): Classes as ordered by the model (model.classes_).
            title (str): Title of the plot.

        Returns:
            matplotlib.figure.Figure: Figure object containing the plot.
        """
        n_classes = len(model_classes)

        # Handle binary and multiclass cases
        if n_classes == 2:
            return RocAucFigure._plot_binary_roc(y_true, y_prob, title)

        else:
            return RocAucFigure._plot_multiclass_roc(
                y_true, y_prob, model_classes, title
            )

    @staticmethod
    def _plot_binary_roc(y_true, y_prob, title):
        """
        Generates the ROC plot for binary classification.

        Args:
            y_true (array-like): Ground truth target values.
            y_prob (array-like): Predicted probabilities for the positive class.
            title (str): Title of the plot.

        Returns:
            matplotlib.figure.Figure: Figure object containing the plot.
        """
        # Extract probabilities for the positive class
        y_prob_positive = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob_positive)
        roc_auc = auc(fpr, tpr)

        # Create and configure the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
        RocAucFigure._add_base_plot(ax, title)

        return fig

    @staticmethod
    def _plot_multiclass_roc(y_true, y_prob, model_classes, title):
        """
        Generates the ROC plot for multiclass classification.

        Args:
            y_true (array-like): Ground truth target values.
            y_prob (array-like): Predicted probabilities for each class.
            model_classes (array-like): Classes as ordered by the model (model.classes_).
            title (str): Title of the plot.

        Returns:
            matplotlib.figure.Figure: Figure object containing the plot.
        """
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=model_classes)

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        fpr, tpr, roc_auc = {}, {}, {}

        # Compute ROC curve and AUC for each class
        for i, class_label in enumerate(model_classes):
            fpr[class_label], tpr[class_label], _ = roc_curve(
                y_true_bin[:, i], y_prob[:, i]
            )
            roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
            ax.plot(
                fpr[class_label],
                tpr[class_label],
                lw=2,
                label=f"Class {class_label} (AUC = {roc_auc[class_label]:.2f})",
            )

        # Compute macro-average AUC
        RocAucFigure._add_macro_avg(ax, fpr, tpr, model_classes)

        RocAucFigure._add_base_plot(ax, title)

        return fig

    @staticmethod
    def _add_macro_avg(ax, fpr, tpr, model_classes):
        """
        Adds macro-average ROC curve to the plot.

        Args:
            ax (matplotlib.axes.Axes): Plot axes.
            fpr (dict): False positive rates for each class.
            tpr (dict): True positive rates for each class.
            model_classes (array-like): Classes as ordered by the model (model.classes_).
        """
        all_fpr = np.unique(
            np.concatenate([fpr[class_label] for class_label in model_classes])
        )
        mean_tpr = np.zeros_like(all_fpr)

        for class_label in model_classes:
            mean_tpr += np.interp(all_fpr, fpr[class_label], tpr[class_label])

        mean_tpr /= len(model_classes)
        macro_auc = auc(all_fpr, mean_tpr)

        ax.plot(
            all_fpr,
            mean_tpr,
            color="navy",
            lw=2,
            linestyle="--",
            label=f"Macro Avg (AUC = {macro_auc:.2f})",
        )

    @staticmethod
    def _add_base_plot(ax, title):
        """
        Configures the base plot (axes, labels, and diagonal line).

        Args:
            ax (matplotlib.axes.Axes): Plot axes.
            title (str): Title of the plot.
        """
        ax.plot(
            [0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Guess"
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
