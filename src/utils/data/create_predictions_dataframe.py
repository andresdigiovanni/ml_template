import pandas as pd


def create_predictions_dataframe(y_pred, y_prob):
    predictions_snapshot = pd.DataFrame({"predictions": y_pred})

    if y_prob.shape[1] > 1:
        for i, class_prob in enumerate(y_prob.T):
            predictions_snapshot[f"predictions_prob_class_{i}"] = class_prob
    else:
        predictions_snapshot["predictions_prob"] = y_prob.flatten()

    return predictions_snapshot
