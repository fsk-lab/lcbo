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
            simulator=simulator,
            compound_set=compounds,
            gradient=gradient,
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
