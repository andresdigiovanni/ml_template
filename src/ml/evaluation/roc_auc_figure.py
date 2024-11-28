import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize


def create_roc_auc_figure(y_true, y_prob, title="ROC Curve"):
    """
    Creates a matplotlib.figure.Figure object with ROC curves and AUC values.
    Supports both binary and multi-class problems.

    Args:
        y_true (array-like): Ground truth target values.
        y_prob (array-like): Predicted probabilities for each class.
        title (str): Title of the plot.

    Returns:
        matplotlib.figure.Figure: Figure object containing the plot.
    """
    # Automatically determine the number of classes
    n_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 2

    # Binarize labels for multi-class cases
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute ROC curves and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    # Compute macro-average AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    macro_roc_auc = auc(all_fpr, mean_tpr)

    # Add macro-average to the plot
    ax.plot(
        all_fpr,
        mean_tpr,
        color="navy",
        lw=2,
        linestyle="--",
        label=f"Macro Avg (AUC = {macro_roc_auc:.2f})",
    )

    # Add reference line
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return fig
