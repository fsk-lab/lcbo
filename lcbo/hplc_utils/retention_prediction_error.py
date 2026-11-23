from typing import Optional
import numpy as np

from .hplc_experiment import HPLCExperiment


def retention_prediction_error(predicted: HPLCExperiment, observed: HPLCExperiment) -> Optional[float]:
    """
    Evaluates the prediction errors of a retention model by comparing predicted and observed retention times. Retention
    error is calcualted as the RMS log ratio for positive finite observations. A false non-elution prediction receives
    at least unit error.

    Args:
        predicted (np.ndarray): The predicted retention times from the model.
        observed (np.ndarray): The observed retention times from the experiment.

    Returns:
        float: The retention prediction error (RMS log ratio) or None if no valid comparisons exist.
    """
    true_retention_times, predicted_retention_times = observed.retention_times, predicted.retention_times
    errors = []
    for true, prediction in zip(true_retention_times, predicted_retention_times):
        if np.isfinite(true) and true > 0:
            if np.isfinite(prediction) and prediction > 0:
                errors.append(float(np.log(prediction / true)))
            else:
                errors.append(max(1.0, float(np.log(observed.gradient.run_time / true))))
        elif np.isposinf(true) and np.isfinite(prediction) and 0 < prediction < observed.gradient.run_time:
            errors.append(float(np.log(observed.gradient.run_time / prediction)))

    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None