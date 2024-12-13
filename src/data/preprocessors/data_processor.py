from cross import CrossTransformer, auto_transform


class DataProcessor:
    """
    Class for automating data preprocessing and feature transformations using CrossTransformer.

    This class provides an interface for fitting transformations, applying them,
    and customizing the pipeline for feature engineering and preprocessing.

    Attributes:
        model: The machine learning model used for optimization in auto-transformations.
        scoring (str): Scoring metric to evaluate transformations.
        direction (str): Optimization direction, typically 'maximize' or 'minimize'.
        transformer (CrossTransformer): Instance of the CrossTransformer class for managing transformations.
    """

    def __init__(self, model, scoring: str = None, direction: str = None) -> None:
        """
        Initializes the DataProcessor with a model, scoring metric, and optimization direction.

        Args:
            model: The machine learning model to guide the transformation process.
            scoring (str): Scoring metric for evaluating transformations (e.g., 'accuracy', 'r2').
            direction (str): Direction of optimization ('maximize' or 'minimize').
        """
        self.model = model
        self.scoring = scoring
        self.direction = direction
        self.transformer = None

    def get_params(self) -> dict:
        """
        Retrieves the current transformation parameters.

        Returns:
            dict: Parameters of the current transformation pipeline.

        Raises:
            RuntimeError: If no transformations are set (i.e., fit() has not been called).
        """
        if self.transformer is None:
            raise RuntimeError("No transformations are set. Call fit() first.")

        return self.transformer.get_params()

    def set_params(self, transformations: dict) -> None:
        """
        Manually sets the transformation pipeline.

        Args:
            transformations (dict): A dictionary defining the transformations to apply.

        Example:
            transformations = {
                "scaling": {"method": "standard"},
                "feature_selection": {"threshold": 0.1},
            }
        """
        self.transformer = CrossTransformer()
        self.transformer.set_params(**transformations)

    def fit(self, X, y=None) -> "DataProcessor":
        """
        Automatically fits transformations to the provided data using the given model and scoring.

        Args:
            X (pd.DataFrame or np.ndarray): Feature data.
            y (pd.Series or np.ndarray, optional): Target data.

        Returns:
            DataProcessor: The fitted DataProcessor instance.
        """
        transformations = auto_transform(X, y, self.model, self.scoring, self.direction)
        self.transformer = CrossTransformer(transformations)
        self.transformer.fit(X, y)
        return self

    def transform(self, X, y=None):
        """
        Applies the fitted transformations to the data.

        Args:
            X (pd.DataFrame or np.ndarray): Feature data.
            y (pd.Series or np.ndarray, optional): Target data.

        Returns:
            Transformed data (pd.DataFrame or np.ndarray).

        Raises:
            RuntimeError: If fit() has not been called before transform().
        """
        if self.transformer is None:
            raise RuntimeError("fit() must be called before applying transform().")

        return self.transformer.transform(X, y)

    def fit_transform(self, X, y=None):
        """
        Fits and transforms the data in a single step.

        Args:
            X (pd.DataFrame or np.ndarray): Feature data.
            y (pd.Series or np.ndarray, optional): Target data.

        Returns:
            Transformed data (pd.DataFrame or np.ndarray).
        """
        return self.fit(X, y).transform(X, y)
