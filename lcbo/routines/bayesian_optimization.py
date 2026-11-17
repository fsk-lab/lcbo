from typing import Optional
from pathlib import Path
import copy
import time
import numpy as np
import torch
from botorch.models import ModelList

from moses.datatypes import LCGradient, Compound
from moses.controller import HPLCMSController
from moses.analyzer import PeakTrack
from moses.simulator import Simulator


from ..data.campaign import HPLCOptimizationCampaign
from ..data.config import HPLCOptimizationConfig, HPLCExperimentConfig
from ..hplc_utils.multilinear_gradient import LCGradientParameters
from ..hplc_utils.analyte import AnalyteSet
from ..hplc_utils.simulator import get_simulator_from_data
from ..bo_utils.random import sample_from_polytope
from ..bo_utils.surrogate import train_surrogate_model
from ..bo_utils.acquisition import optimize_acquisition_function
from ..utils.numpy_torch import replace_nonfinite

from .experiment import run_experiment, run_simulation_experiment
from .initial_experiments import generate_seed_experiments


def recommend_next_gradient(
        campaign: HPLCOptimizationCampaign,
        config: HPLCOptimizationConfig,
        experiment_config: Optional[HPLCExperimentConfig] = None,
        random_seed: int = 42
) -> tuple[bool, LCGradient]:
    """
    Recommend the next gradient to test based on the current campaign data and optimization configuration. Proceeds in
    two steps:
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

    if config.single_fidelity:
        fidelity_threshold = 1.0
    elif config.only_current_fidelity:
        fidelity_threshold = campaign.expected_fidelity_level
    else:
        fidelity_threshold = 0.0

    train_x_detection, train_f_detection = campaign.get_gradient_parameters(fidelity_threshold=fidelity_threshold)
    if "NumPeaks" in config.objective:
        train_y_detection = campaign.get_num_peaks(fidelity_threshold=fidelity_threshold)
    elif "SinglePeak" in config.objective:
        train_y_detection = campaign.get_single_compound_elution(compound_idx=config.compound_idx, fidelity_threshold=fidelity_threshold)
    else:
        raise ValueError(f"Unknown objective: {config.objective}")
    retention_times_detection = campaign.get_retention_times(fidelity_threshold=fidelity_threshold)

    train_x_resolution, train_f_resolution = campaign.get_gradient_parameters(fidelity_threshold=fidelity_threshold)
    if config.objective == "NumPeaks-CritRes":
        train_y_resolution = campaign.get_critical_resolutions(fidelity_threshold=fidelity_threshold)
    elif config.objective == "NumPeaks-NumResPeaks":
        train_y_resolution = campaign.get_num_resolved_peaks(fidelity_threshold=fidelity_threshold)
    elif config.objective == "SinglePeak-Resolution":
        train_y_resolution = campaign.get_single_compound_resolutions(compound_idx=config.compound_idx, fidelity_threshold=fidelity_threshold)
    else:
        raise ValueError(f"Unknown objective: {config.objective}")

    fallback = False

    ####################################################################################################################
    # Option 1: Train single surrogate model on [detection_fraction, critical_resolution] as outputs
    ####################################################################################################################

    if config.objective_type == "cumulative":

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

    elif config.objective_type == "compoundwise":

        train_y = replace_nonfinite(retention_times_detection, nan_value=0.0, inf_value=1.5 * config.max_run_time)

        try:
            surrogate_retention = train_surrogate_model(config, train_x_detection, train_y, train_f_detection)
            model = ModelList(surrogate_retention)

        except Exception as e:
            config.logger.warning(f"Surrogate model training failed with error: {e}. "
                                  f"Falling back to random sampling.")
            fallback = True

    ####################################################################################################################
    # Unknown objective type
    ####################################################################################################################

    else:
        raise ValueError(f"Unknown objective type: {config.objective_type}")

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

    gradient_params = {}
    if experiment_config is not None:
        gradient_params["equilibration_time"] = experiment_config.equilibration_time
        gradient_params["eluents"] = experiment_config.eluents

    next_params_obj = LCGradientParameters(next_params.numpy().flatten(), segment_based=config.segment_based)
    return high_fidelity, next_params_obj.to_gradient(flow_rate=config.flow_rate, **gradient_params)


def run_optimization_campaign(
        simulation: bool,
        instrument: HPLCMSController | Simulator,
        optimization_config: HPLCOptimizationConfig,
        analytes: AnalyteSet | list[Compound] | list[PeakTrack],
        data_dir: Path,
        sample_settings: Optional[dict] = None,
        experiment_config: Optional[HPLCExperimentConfig] = None,
        random_seed: int = 42
) -> HPLCOptimizationCampaign:
    """
    Runs a full optimization campaign in simulation or on a real HPLC-MS instrument.

    Args:
        simulation (bool): Whether to run the campaign in simulation mode or on a real instrument.
        instrument (HPLCMSController | Simulator): If simulation is True, this should be a Simulator object; otherwise,
                                                   it should be an HPLCMSController object.
        optimization_config (HPLCOptimizationConfig): The configuration for the optimization campaign.
        analytes (AnalyteSet | list[Compound] | list[PeakTrack]): If simulation is True, this should be an AnalyteSet
                                                                  object with retention parameters for all analytes;
                                                                  otherwise, it should be a list of
                                                                  moses.datatypes.Compound objects or a list of
                                                                  moses.analyzer.PeakTrack objects to be analyzed.
        data_dir (Path): The directory to save experimental data.
        sample_settings (dict, optional): The settings for the sample injections. Required if simulation is False.
        experiment_config (HPLCExperimentConfig, optional): The configuration for individual experiments. Required if
                                                            simulation is False.
        random_seed (int): Random seed for reproducibility.

    Returns:
        HPLCOptimizationCampaign: The completed optimization campaign with all experiments and results.
    """
    if simulation is False:
        if not isinstance(instrument, HPLCMSController):
            raise ValueError("For real experiments, 'instrument' must be an instance of HPLCMSController.")
        if sample_settings is None:
            raise ValueError("For real experiments, 'sample_settings' must be provided.")
        if experiment_config is None:
            raise ValueError("For real experiments, 'experiment_config' must be provided.")

    optimization_config.initialize_torch()

    def log_message(message: str):
        if simulation:
            optimization_config.logger.debug(message)
        else:
            optimization_config.logger.info(message)

    ####################################################################################################################
    # Step 1: Instantiate an optimization campaign. Check for an existing campaign to resume from. If not found, create
    # a new campaign and run initial seed experiments.
    ####################################################################################################################

    try:
        campaign = HPLCOptimizationCampaign.from_pkl(data_dir / "campaign.pkl")
        optimization_config.logger.info(f"Loaded existing campaign from {data_dir / 'campaign.pkl'}. Resuming optimization with {campaign.num_high_fidelity_experiments} already completed.")
    except FileNotFoundError:
        optimization_config.logger.info(f"Started new optimization campaign.")
        campaign = HPLCOptimizationCampaign(config=optimization_config)

        initial_eluent_programs = generate_seed_experiments(
            config=optimization_config,
            experiment_config=experiment_config,
            random_seed=random_seed,
        )

        for eluent_program in initial_eluent_programs:

            if simulation:
                experiment = run_simulation_experiment(
                    simulator=instrument,
                    compound_set=analytes,
                    gradient=eluent_program,
                    resolution_threshold=optimization_config.resolution_threshold,
                    advanced_peak_detection=optimization_config.advanced_peak_detection,
                    fidelity_level=1.0
                )
            else:
                time_stamp = time.strftime("%Y-%m-%d_%H-%M")
                _sample = copy.deepcopy(sample_settings)
                _sample["sample_name"] = f"Optimization_{campaign.num_high_fidelity_experiments + 1}_{sample_settings['sample_name']}_{time_stamp}"

                experiment = run_experiment(
                    instrument=instrument,
                    config=experiment_config,
                    gradient=eluent_program,
                    sample_settings=_sample,
                    targets=analytes,
                    data_dir=data_dir / f"Experiment_{campaign.num_high_fidelity_experiments + 1}",
                    pre_flush=True
                )

            experiment.segment_based = optimization_config.segment_based
            experiment.score = optimization_config.calculate_experiment_score(experiment)
            campaign.append(experiment)
            log_message(f"Completed experiment {campaign.num_high_fidelity_experiments} with score {experiment.score:.2f}")

        if (campaign.scores > 1.0).flatten().sum() < 1:
            raise RuntimeError("Some compounds were not detected in any of the initial experiments.")

    ####################################################################################################################
    # Step 2: Run the main optimization loop until the budget of high-fidelity experiments is reached. In each
    # iteration, recommend the next gradient, run the experiment (either in simulation or on the real instrument),
    # and update the campaign with the new results.
    ####################################################################################################################

    lf_simulator, lf_compounds_fitted, lf_train_size, current_lf_count = None, None, 0, 0

    while campaign.num_high_fidelity_experiments < optimization_config.budget:

        high_fidelity, next_eluent_program = recommend_next_gradient(
            campaign=campaign,
            config=optimization_config,
            experiment_config=experiment_config,
        )

        log_message(f"Recommended next gradient at {'high' if high_fidelity else 'low'} fidelity.")

        if not high_fidelity and current_lf_count >= optimization_config.max_consecutive_low_fidelity:
            high_fidelity = True
            current_lf_count = 0

        if not high_fidelity:

            if lf_train_size != campaign.num_high_fidelity_experiments:
                lf_simulator, lf_compounds_fitted = get_simulator_from_data(
                    data=campaign,
                    column_length=instrument.parameters["column_length"] if simulation else optimization_config.column_length,
                    column_inner_diameter=optimization_config.column_inner_diameter,
                    simulation_interval=0.01,
                    previous_simulator=lf_simulator,
                    previous_compound_params=lf_compounds_fitted
                )
                lf_train_size = campaign.num_high_fidelity_experiments

            experiment = run_simulation_experiment(
                simulator=lf_simulator,
                compound_set=lf_compounds_fitted,
                gradient=next_eluent_program,
                resolution_threshold=optimization_config.resolution_threshold,
                advanced_peak_detection=optimization_config.advanced_peak_detection,
                fidelity_level=campaign.expected_fidelity_level
            )

        else:

            if simulation:
                experiment = run_simulation_experiment(
                    simulator=instrument,
                    compound_set=analytes,
                    gradient=next_eluent_program,
                    resolution_threshold=optimization_config.resolution_threshold,
                    advanced_peak_detection=optimization_config.advanced_peak_detection,
                    fidelity_level=1.0
                )
            else:
                time_stamp = time.strftime("%Y-%m-%d_%H-%M")
                _sample = copy.deepcopy(sample_settings)
                _sample["sample_name"] = f"Optimization_{campaign.num_high_fidelity_experiments + 1}_{sample_settings['sample_name']}_{time_stamp}"

                experiment = run_experiment(
                    instrument=instrument,
                    config=experiment_config,
                    gradient=next_eluent_program,
                    sample_settings=_sample,
                    targets=analytes,
                    data_dir=data_dir / f"Experiment_{campaign.num_high_fidelity_experiments + 1}",
                    pre_flush=True
                )
            current_lf_count = 0

        experiment.segment_based = optimization_config.segment_based
        experiment.score = optimization_config.calculate_experiment_score(experiment)
        campaign.append(experiment)
        log_message(f"Completed experiment {campaign.num_high_fidelity_experiments} with score {experiment.score:.2f}")

        if campaign.num_high_fidelity_experiments % 10 == 0 or campaign.num_high_fidelity_experiments == optimization_config.budget:
            campaign.to_pkl(data_dir / "campaign.pkl")

    if simulation:
        best_experiment = campaign.optimum_experiment
        _ = run_simulation_experiment(
            simulator=instrument,
            compound_set=analytes,
            gradient=best_experiment.gradient,
            resolution_threshold=optimization_config.resolution_threshold,
            advanced_peak_detection=optimization_config.advanced_peak_detection,
            fidelity_level=1.0,
            chromatogram_file=data_dir / f"Optimum_Gradient.png"
        )

    return campaign
