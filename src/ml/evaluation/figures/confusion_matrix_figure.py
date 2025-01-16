import matplotlib.pyplot as plt
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

        # Plot the confusion matrix as an image
        cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(cax, ax=ax)

        # Add text annotations for each cell
        for i in range(len(model_classes)):
            for j in range(len(model_classes)):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:d}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

        # Set axis labels and title
        ax.set_xticks(range(len(model_classes)))
        ax.set_yticks(range(len(model_classes)))
        ax.set_xticklabels(model_classes)
        ax.set_yticklabels(model_classes)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title(title)

        # Adjust layout
        plt.tight_layout()

        return fig


# Example of use
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Load data
    data = load_iris()
    X, y = data.data, data.target
    class_names = data.target_names

    # Divide data train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train a model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Get predictions
    y_pred = clf.predict(X_test)

    # Create ConfusionMatrix figure
    fig = ConfusionMatrixFigure.create_figure(
        y_true=y_test,
        y_pred=y_pred,
        model_classes=np.arange(len(class_names)),
    )
    plt.show()
