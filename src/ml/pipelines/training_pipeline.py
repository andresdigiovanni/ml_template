import os

from sklearn.model_selection import train_test_split

from src.data.preprocessors import DataProcessor
from src.ml.evaluation import ConfusionMatrixFigure, RocAucFigure, compute_metrics
from src.ml.explainer import FeatureImportanceExplainer, ModelExplainer
from src.ml.tuning import HyperparameterTuner
from src.utils.data import create_predictions_dataframe


class TrainingPipeline:
    """
    A class to manage the complete machine learning training pipeline,
    including preprocessing, hyperparameter tuning, model training,
    evaluation, and saving the model and results.

    Attributes:
        model: The machine learning model to train.
        hyperparameter_tuner: A hyperparameter optimization tool.
        data_processor: A data transformation and preprocessing tool.
        model_tracker: A model tracker connector for saving models and results.
        object_connector: An object storage connector for saving data snapshots.
        logger: The logger for logging messages.
    """

    def __init__(self, model, model_tracker, object_connector, logger, config):
        self.logger = logger
        self.model_tracker = model_tracker
        self.object_connector = object_connector
        self.problem_type = config["model"]["problem_type"]

        # Inicializa el modelo y el tuner con la configuración proporcionada
        self.model = model
        self.hyperparameter_tuner = HyperparameterTuner(
            self.model,
            config["model"]["scoring"],
            config["model"]["direction"],
            model_tracker,
        )
        self.data_processor = DataProcessor(
            self.model, config["model"]["scoring"], config["model"]["direction"]
        )

    def run(self, data, target_column, test_size=0.2, random_state=42):
        with self.model_tracker.run(run_name="training"):
            self.logger.info("Splitting data into training and test sets...")
            X = data.drop(columns=[target_column])
            y = data[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=True
            )

            # Save a snapshot of the training data
            self.logger.info("Saving snapshot of training data...")
            train_snapshot = X_train.copy()
            train_snapshot[target_column] = y_train
            self.object_connector.put_object(
                train_snapshot,
                os.path.join(
                    "training",
                    f"{self.model_tracker.run_id()}_data_snapshot.parquet",
                ),
            )

            # Procesamiento de datos
            self.logger.info("Processing training data...")
            X_train = self.data_processor.fit_transform(X_train, y_train)
            X_test = self.data_processor.transform(X_test)
            processor_params = self.data_processor.get_params()

            # Ajuste de hiperparámetros
            self.logger.info("Starting hyperparameter tuning...")
            best_params = self.hyperparameter_tuner.fit(X_train, y_train)

            # Entrenamiento del modelo
            self.logger.info("Training the model with the best parameters...")
            self.model.set_params(**best_params)
            self.model.fit(X_train, y_train)

            # Evaluación
            self.logger.info("Evaluating the model...")
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)

            metrics = compute_metrics(self.model, X_test, y_test, self.problem_type)

            roc_auc_figure = RocAucFigure.create_figure(
                y_test, y_prob, self.model.classes_
            )
            cm_figure = ConfusionMatrixFigure.create_figure(
                y_test, y_pred, self.model.classes_
            )

            model_explainer = ModelExplainer(self.model, X_train)
            model_explainer_figures = model_explainer.generate_summary_plot()

            fi_figure = FeatureImportanceExplainer().create_figure(
                X_train.columns, self.model.feature_importances_
            )

            # Guardar resultados
            self.model_tracker.log_model(self.model, "model.pkl")
            self.model_tracker.log_artifact(processor_params, "processor.pkl")
            self.model_tracker.log_metrics(metrics)
            self.model_tracker.log_params(best_params)
            self.model_tracker.log_figure(roc_auc_figure, "roc_auc_plot.png")
            self.model_tracker.log_figure(cm_figure, "confusion_matrix.png")

            for filename, fig in model_explainer_figures:
                self.model_tracker.log_figure(fig, filename)

            self.model_tracker.log_figure(fi_figure, "feature_importance.png")

            # Save a snapshot of predicted data
            self.logger.info("Saving snapshot of predicted data...")
            predictions_snapshot = create_predictions_dataframe(y_pred, y_prob)

            self.object_connector.put_object(
                predictions_snapshot,
                os.path.join(
                    "training",
                    f"{self.model_tracker.run_id()}_predictions_snapshot.parquet",
                ),
            )
