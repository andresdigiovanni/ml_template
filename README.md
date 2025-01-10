# Machine Learning Project Template

## Overview

This template is a robust framework for developing machine learning projects, from preprocessing and training to inference and monitoring. Designed with reproducibility, scalability, and maintainability in mind, it supports a wide range of tools and integrations to accelerate your workflow.

---

## Features

- **Configuration Management:** Flexible and dynamic configuration using YAML files and command-line overrides.
- **Data and Object Connectors:** Supports various data sources (e.g., local, scikit-learn, S3, GCS) and file formats (CSV, JSON, Parquet).
- **Model Tracking:** Compatible with MLflow, Weights & Biases, and local storage for experiment tracking.
- **Hyperparameter Tuning:** Built-in integration with Optuna for efficient optimization.
- **Data Processing:** Advanced feature engineering powered by CrossTransformer.
- **Training and Inference Pipelines:** Modular pipelines for seamless execution.
- **Monitoring Tools:** Integrated data drift and model drift analysis using Evidently.
- **Visualization:** Generate ROC-AUC plots, confusion matrices, and SHAP-based explainability insights.

---

## Project Structure

```plaintext
src/
├── apps/
│   ├── inference.py
│   ├── training.py
├── configs/
├── data/
│   ├── connectors/
│   ├── preprocessors/
├── drift/
│   ├── data_monitoring_tool.py
│   ├── model_monitoring_tool.py
├── ml/
│   ├── evaluation/
│   ├── explainer/
│   ├── pipelines/
│   ├── tuning/
├── tracking/
├── utils/
│   ├── metrics/
│   ├── data/
│   ├── logging/
```

- **`apps/`**: Entry points for training and inference.
- **`configs/`**: Configuration management files.
- **`data/`**: Data connectors and preprocessors.
- **`drift/`**: Tools for monitoring data and model drift.
- **`ml/`**: Core machine learning components, including pipelines, tuning, evaluation, and explainers.
- **`tracking/`**: Components for tracking experiments and models.
- **`utils/`**: Helper functions for metrics, logging, and data manipulation.

---

## How to Use

### 1. Configure the Project
Update the `config.yaml` file to match your dataset, tracking system, and other parameters.

### 2. Run Training
Execute the training pipeline to preprocess data, tune hyperparameters, and train the model:

```bash
python src/apps/training.py
```

### 3. Run Inference
Use the trained model to generate predictions and monitor performance:

```bash
python src/apps/inference.py
```

---

## Configuration Example (`config.yaml`)

Here’s an example of the `config.yaml` file for a scikit-learn dataset:

```yaml
data:
    type: "sklearn"            # Supported types: "local", "sklearn", "s3", "gcs"
    source: "wine"             # Dataset name (e.g., "iris", "wine", etc.)
    target_column: "target"    # Column to predict

logging:
    log_file: app.log          # Log file location
    log_level: 20              # Logging level: 10=DEBUG, 20=INFO, etc.

model:
    problem_type: "classification"  # "classification" or "regression"
    scoring: "roc_auc_ovr"          # Metric for optimization
    direction: "maximize"           # Direction of optimization

tracking:
    type: "local"                  # Supported: "local", "mlflow", "wandb"
    experiment_name: "sklearn_wine"

object_connector:
    type: "local"                  # Object storage type
    params:
        base_path: "./data/"       # Base path for storage
```

---

## Components

### Configuration Loader (`src/configs/config_loader.py`)
Loads and merges configurations from YAML files and command-line arguments for flexible execution.

### Training Pipeline (`src/ml/pipelines/training_pipeline.py`)
Manages the end-to-end training workflow:
- Preprocessing
- Hyperparameter tuning
- Model training
- Evaluation
- Saving models and metrics

### Inference Pipeline (`src/ml/pipelines/inference_pipeline.py`)
Generates predictions, monitors data drift, and logs outputs.

### Hyperparameter Tuner (`src/ml/tuning/hyperparameter_tuner.py`)
Optimizes model parameters using Optuna and supports cross-validation.

### Monitoring Tools
- **Data Monitoring Tool:** Checks data quality and drift.
- **Model Monitoring Tool:** Monitors model drift in production.

---

## Contribution Guidelines

We welcome contributions to improve this template. Please fork the repository, create a feature branch, and submit a pull request.

---

## License

[MIT License](LICENSE)
