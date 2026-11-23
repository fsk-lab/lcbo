from typing import Optional
import numpy as np
from scipy.optimize import differential_evolution

from moses.datatypes import LCGradient
from moses.simulator import Simulator

from ..data.config import HPLCOptimizationConfig, HPLCExperimentConfig
from ..hplc_utils.analyte import AnalyteSet
from ..hplc_utils.hplc_experiment import HPLCExperiment
from ..hplc_utils.multilinear_gradient import LCGradientParameters
from ..bo_utils.random import sample_from_polytope
from .experiment import run_simulation_experiment


def recommend_retention_modeling(
        simulator: Simulator,
        compounds: AnalyteSet,
        config: HPLCOptimizationConfig,
        experiment_config: Optional[HPLCExperimentConfig] = None,
        random_seed: int = 42
) -> LCGradient:
    """
    Recommends a new eluent program (LCGradient) by optimizing the fitted retention model's predicted separation.
    Optimization is performed using constrained differential evolution.

    Args:
        simulator (Simulator): The MoSeS simulator instance for running experiments.
        compounds (AnalyteSet): The set of analytes for which the retention model is fitted.
        config (HPLCOptimizationConfig): The optimization configuration with model and acquisition settings.
        experiment_config (HPLCExperimentConfig, optional): Configuration object for the HPLC experiment. If provided,
                                                            specifies eluent program parameters that are not used in
                                                            simulation, but are relevant for real experiments.
        random_seed (int): Random seed for reproducibility.

    Returns:
        LCGradient: The recommended next gradient to test.

    """
    gradient_params = {}
    if experiment_config is not None:
        gradient_params["equilibration_time"] = experiment_config.equilibration_time
        gradient_params["eluents"] = experiment_config.eluents

    initial_params = sample_from_polytope(
        config.bounds,
        config.constraints,
        max(5, 5 * config.bounds.shape[1]),
        seed=random_seed
    ).numpy()

    def loss_fn(params: np.ndarray) -> float:
        """
        Uses the fitted retention model to predict the separation score of a candidate eluent program (LCGradient).

        Args:
            params (np.ndarray): Candidate parameters for the eluent program.

        Returns:
            float: The negative separation score of the predicted experiment.
        """
        gradient = LCGradientParameters(params, config.segment_based).to_gradient(config.flow_rate, **gradient_params)
        experiment = run_simulation_experiment(
            simulator,
            compounds,
            gradient,
            config.resolution_threshold,
            advanced_peak_detection=config.advanced_peak_detection
        )
        experiment.segment_based = config.segment_based
        return -config.calculate_experiment_score(experiment)

    result = differential_evolution(
        func=loss_fn,
        bounds=config.bounds_scipy,
        constraints=config.constraints_scipy,
        init=initial_params,
        seed=random_seed,
        maxiter=25,
        polish=False,
        tol=1E-6
    )

    if not np.isfinite(result.fun):
        raise RuntimeError("No feasible retention-model recommendation was found.")
    return LCGradientParameters(result.x, config.segment_based).to_gradient(config.flow_rate, **gradient_params)


def prediction_errors(predicted: HPLCExperiment, observed: HPLCExperiment) -> tuple[Optional[float], Optional[float]]:
    """
    Evaluates the prediction errors of a retention model by comparing predicted and observed retention times and peak
    widths.
      - Retention error is the RMS log ratio for positive finite observations. A false non-elution prediction receives
        at least unit error.
      - Width error is the RMS log FWHM ratio.
    Unknown observations (NaN) are excluded from both metrics.

    Args:
        predicted (HPLCExperiment): The predicted experiment containing retention times and FWHMs.
        observed (HPLCExperiment): The observed experiment containing retention times and FWHMs.

    Returns:
        Optional[float]: The retention error (RMS log ratio) or None if no valid observations exist.
        Optional[float]: The width error (RMS log FWHM ratio) or None if no valid observations exist.
    """
    true_times, pred_times = observed.retention_times, predicted.retention_times
    errors = []
    for actual, prediction in zip(true_times, pred_times):
        if np.isfinite(actual) and actual > 0:
            if np.isfinite(prediction) and prediction > 0:
                errors.append(float(np.log(prediction / actual)))
            else:
                errors.append(max(1.0, float(np.log(observed.gradient.run_time / actual))))
        elif np.isposinf(actual) and np.isfinite(prediction) and 0 < prediction < observed.gradient.run_time:
            errors.append(float(np.log(observed.gradient.run_time / prediction)))
    retention_error = float(np.sqrt(np.mean(np.square(errors)))) if errors else None

    valid_widths = np.isfinite(observed.fwhms) & (observed.fwhms > 0)
    width_errors = []
    for actual, prediction in zip(observed.fwhms[valid_widths], predicted.fwhms[valid_widths]):
        if np.isfinite(prediction) and prediction > 0:
            width_errors.append(float(np.log(prediction / actual)))
        else:
            width_errors.append(1.0)  # Assign a default error for non-finite or non-positive predictions
    width_error = float(np.sqrt(np.mean(np.square(width_errors)))) if width_errors else None

    return retention_error, width_error
