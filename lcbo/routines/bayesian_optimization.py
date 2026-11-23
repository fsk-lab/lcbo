from typing import Optional
import numpy as np
import torch
from botorch.models import ModelList

from moses.datatypes import LCGradient

from ..data.campaign import HPLCOptimizationCampaign
from ..data.config import HPLCOptimizationConfig, HPLCExperimentConfig
from ..hplc_utils.multilinear_gradient import LCGradientParameters
from ..bo_utils.random import sample_from_polytope
from ..bo_utils.surrogate import train_surrogate_model
from ..bo_utils.acquisition import optimize_acquisition_function


def recommend_bayesian_optimization(
        campaign: HPLCOptimizationCampaign,
        config: HPLCOptimizationConfig,
        experiment_config: Optional[HPLCExperimentConfig] = None,
        random_seed: int = 42
) -> tuple[bool, LCGradient]:
    """
    Recommend a new eluent program (LCGradient) for the next experiment in the campaign using Bayesian optimization.
    Proceeds in two steps:
        1. Fit the surrogate model to the existing campaign data.
        2. Optimize the acquisition function to select the next gradient.

    Args:
        campaign (HPLCOptimizationCampaign): The current campaign containing past experiments.
        config (HPLCOptimizationConfig): The optimization configuration with model and acquisition settings.
        experiment_config (HPLCExperimentConfig, optional): Configuration object for the HPLC experiment. If provided,
                                                            specifies eluent program parameters that are not used in
                                                            simulation, but are relevant for real experiments.
        random_seed (int): Random seed for reproducibility.

    Returns:
        bool: Indicates if high-fidelity gradient is recommended (True) or low-fidelity (False).
        LCGradient: The recommended next gradient to test.
    """
    torch.manual_seed(random_seed)

    gradient_params = {}
    if experiment_config is not None:
        gradient_params["equilibration_time"] = experiment_config.equilibration_time
        gradient_params["eluents"] = experiment_config.eluents

    if config.acquisition_function.lower() == "random":
        params = sample_from_polytope(config.bounds, config.constraints, 1, seed=random_seed)[0].numpy()
        return True, LCGradientParameters(params, config.segment_based).to_gradient(config.flow_rate, **gradient_params)

    fallback = False

    ####################################################################################################################
    # Option 1: Train single surrogate model on [detection_fraction, critical_resolution] as outputs
    ####################################################################################################################

    if config.model_type == "cumulative":

        train_x_detection, train_f_detection = campaign.get_gradient_parameters(float(config.single_fidelity))
        if "NumPeaks" in config.objective:
            train_y_detection = campaign.get_num_peaks(float(config.single_fidelity))
        elif "SinglePeak" in config.objective:
            train_y_detection = campaign.get_single_compound_elution(config.compound_idx, float(config.single_fidelity))
        else:
            raise ValueError(f"Unknown objective: {config.objective}")

        train_x_resolution, train_f_resolution = campaign.get_gradient_parameters(float(config.single_fidelity))
        if config.objective == "NumPeaks-CritRes":
            train_y_resolution = campaign.get_critical_resolutions(float(config.single_fidelity))
        elif config.objective == "NumPeaks-NumResPeaks":
            train_y_resolution = campaign.get_num_resolved_peaks(float(config.single_fidelity))
        elif config.objective == "SinglePeak-Resolution":
            train_y_resolution = campaign.get_single_compound_resolutions(config.compound_idx, float(config.single_fidelity))
        else:
            raise ValueError(f"Unknown objective: {config.objective}")

        try:
            model_detection = train_surrogate_model(config, train_x_detection, train_y_detection, train_f_detection)
            model_resolution = train_surrogate_model(config, train_x_resolution, train_y_resolution, train_f_resolution)
            model = ModelList(model_detection, model_resolution)

        except Exception as e:
            config.logger.warning(f"Surrogate model training failed with error: {e}. "
                                  f"Falling back to random sampling.")
            fallback = True

    ####################################################################################################################
    # Option 2: Train surrogate model on retention times as outputs
    ####################################################################################################################

    elif config.model_type == "compoundwise":

        train_x, train_f = campaign.get_gradient_parameters(float(config.single_fidelity))
        train_retention_times = campaign.get_retention_times(float(config.single_fidelity))
        train_fwhms = campaign.get_peak_fwhms(float(config.single_fidelity))
        train_fwhms[~np.isfinite(train_fwhms) | (train_fwhms <= 0)] = np.nan
        train_run_times = campaign.get_gradient_run_times(float(config.single_fidelity))

        try:
            surrogate_retention = train_surrogate_model(
                config,
                train_x,
                train_retention_times,
                train_f,
                retention_run_times=train_run_times
            )
            default_log_fwhm = float(np.log(0.25 / 4 * 2.354820045))
            surrogate_widths = train_surrogate_model(config, train_x, np.log(train_fwhms), train_f, default_value=default_log_fwhm)
            model = ModelList(surrogate_retention, surrogate_widths)

        except Exception as e:
            config.logger.warning(f"Surrogate model training failed with error: {e}. "
                                  f"Falling back to random sampling.")
            fallback = True

    ####################################################################################################################
    # Unknown objective type
    ####################################################################################################################

    else:
        raise ValueError(f"Unknown objective type: {config.model_type}")

    ####################################################################################################################
    # Optimize acquisition function or fallback to random sampling
    ####################################################################################################################

    if not fallback:
        try:
            torch.manual_seed(random_seed)
            high_fidelity, next_params = optimize_acquisition_function(model, campaign, config)
        except Exception as e:
            config.logger.warning(f"Acquisition function optimization failed with error: {e}. "
                                  f"Falling back to random sampling.")
            fallback = True

    if fallback:
        next_params = sample_from_polytope(config.bounds, config.constraints, 1, seed=random_seed)
        high_fidelity = True if config.single_fidelity else (np.random.rand() < 0.5)

    ####################################################################################################################
    # Convert recommended parameters to LCGradient
    ####################################################################################################################

    next_params_obj = LCGradientParameters(next_params.numpy().flatten(), segment_based=config.segment_based)
    return high_fidelity, next_params_obj.to_gradient(flow_rate=config.flow_rate, **gradient_params)
