from typing import Optional
import numpy as np
from moses.datatypes import LCGradient

from ..data.config import HPLCOptimizationConfig, HPLCExperimentConfig
from ..hplc_utils.multilinear_gradient import LCGradientParameters
from ..bo_utils.random import sample_from_polytope


def generate_seed_experiments(
        config: HPLCOptimizationConfig,
        experiment_config: Optional[HPLCExperimentConfig] = None,
        random_seed: int = None
) -> list[LCGradient]:
    """
    Generates a set of initial HPLC gradients based on the specified strategy.

    Currently implemented:
        - "random": Randomly samples gradients from the feasible space defined by the number of segments and maximum
                    run time. Takes into account constraints on time points and fraction values to ensure valid
                    HPLC gradients.
        - "isocratic": Generates gradients with constant organic modifier content, linearly spaced between 0% and 100%.

    Args:
        config (HPLCOptimizationConfig): Configuration object containing parameters for generating seed experiments.
        experiment_config (HPLCExperimentConfig, optional): Configuration object for the HPLC experiment. If provided,
                                                            specifies eluent program parameters that are not used in
                                                            simulation, but are relevant for real experiments.
        random_seed (int, optional): The random seed to use for reproducibility. Defaults to None.

    Returns:
        list[LCGradient]: A list of generated LCGradient objects based on the specified strategy.
    """
    gradient_params = {}
    if experiment_config is not None:
        gradient_params["eluents"] = experiment_config.eluents
        gradient_params["equilibration_time"] = experiment_config.equilibration_time

    if config.seed_strategy == "random":
        bounds, constraints = config.bounds, config.constraints
        sampled_parameters = sample_from_polytope(bounds, constraints, config.seed_budget, seed=random_seed)
        gradients = []
        for params in sampled_parameters:
            param_obj = LCGradientParameters(params, segment_based=config.segment_based)
            gradients.append(param_obj.to_gradient(flow_rate=config.flow_rate, **gradient_params))

    elif config.seed_strategy == "isocratic":
        fractions = np.linspace(0.05, 0.95, config.seed_budget)
        gradients = []
        for frac in fractions:
            _time_steps = np.linspace(0.0, config.max_run_time, config.num_segments + 1)[1:-1]
            _fractions = np.full((config.num_segments + 1,), frac)

            if config.segment_based is False:
                param_obj = LCGradientParameters(
                    np.concatenate([_time_steps, [config.max_run_time], _fractions]),
                    segment_based=False
                )
            else:
                durations = np.diff(np.concatenate([[0.0], _time_steps, [config.max_run_time]]))
                param_obj = LCGradientParameters(
                    np.concatenate([durations, _fractions]),
                    segment_based=True
                )

            gradients.append(param_obj.to_gradient(flow_rate=config.flow_rate, **gradient_params))

    else:
        raise ValueError(f"Unknown strategy: {config.seed_strategy}")

    return gradients
