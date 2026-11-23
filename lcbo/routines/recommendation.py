from typing import Optional
from pathlib import Path
import copy
import time
import numpy as np
import torch

from moses.datatypes import LCGradient, Compound
from moses.controller import HPLCMSController
from moses.analyzer import PeakTrack
from moses.simulator import Simulator

from ..data.campaign import HPLCOptimizationCampaign
from ..data.config import HPLCOptimizationConfig, HPLCExperimentConfig
from ..hplc_utils.analyte import AnalyteSet
from ..hplc_utils.multilinear_gradient import LCGradientParameters
from ..hplc_utils.simulator import get_simulator_from_data
from ..bo_utils.random import sample_from_polytope

from .experiment import run_experiment, run_simulation_experiment
from .initial_experiments import generate_seed_experiments
from .bayesian_optimization import recommend_bayesian_optimization
from .retention_modeling_optimization import recommend_retention_modeling


def recommend_next_eluent_program(
        campaign: HPLCOptimizationCampaign,
        config: HPLCOptimizationConfig,
        experiment_config: Optional[HPLCExperimentConfig] = None,
        analytes: Optional[AnalyteSet] = None,
        simulator: Optional[Simulator] = None,
        random_seed: int = 42
) -> tuple[bool, LCGradient]:
    """
    Recommends the next eluent program to run in the optimization campaign.

    Args:
        campaign (HPLCOptimizationCampaign): The current optimization campaign.
        config (HPLCOptimizationConfig): The configuration for the optimization campaign.
        experiment_config (HPLCExperimentConfig, optional): Configuration object for the HPLC experiment. If provided,
                                                            specifies eluent program parameters that are not used in
                                                            simulation, but are relevant for real experiments.
        analytes (AnalyteSet, optional): Fitted analytes for retention-model optimization.
        simulator (Simulator, optional): The fitted retention-model simulator.
        random_seed (int): Random seed for reproducibility.

    Returns:
        bool: True if the next experiment should be high-fidelity (measured), False if it should be low-fidelity
              (simulated).
        LCGradient: The recommended next eluent program.
    """
    if config.optimization_strategy == "retention_model":
        if simulator is None or analytes is None:
            gradient_params = {}
            if experiment_config is not None:
                gradient_params["equilibration_time"] = experiment_config.equilibration_time
                gradient_params["eluents"] = experiment_config.eluents
            params = sample_from_polytope(config.bounds, config.constraints, 1, seed=random_seed)[0].numpy()
            next_eluent_program = LCGradientParameters(params, config.segment_based).to_gradient(config.flow_rate, **gradient_params)
        else:
            next_eluent_program = recommend_retention_modeling(
                simulator=simulator,
                compounds=analytes,
                config=config,
                experiment_config=experiment_config,
                random_seed=random_seed
            )
        high_fidelity = True
    else:
        high_fidelity, next_eluent_program = recommend_bayesian_optimization(
            campaign=campaign,
            config=config,
            experiment_config=experiment_config,
            random_seed=random_seed
        )

    return high_fidelity, next_eluent_program


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
    data_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load or initialize the optimization campaign (resume if a previous campaign exists)
    try:
        campaign = HPLCOptimizationCampaign.from_pkl(data_dir / "campaign.pkl")
        previous_settings, new_settings = campaign.config.to_dict(), optimization_config.to_dict()
        for attribute in ("budget", "log_level", "working_dir"):
            previous_settings.pop(attribute), new_settings.pop(attribute)
        if previous_settings != new_settings:
            raise ValueError("Cannot resume a campaign with different scientific settings.")
        campaign.config = optimization_config
        optimization_config.logger.info(
            f"Resuming campaign with {campaign.num_high_fidelity_experiments} measured experiments."
        )
    except FileNotFoundError:
        campaign = HPLCOptimizationCampaign(config=optimization_config)

    # Generate initial seed experiments if the campaign has not yet reached the seed budget
    if campaign.num_high_fidelity_experiments < optimization_config.seed_budget:
        initial_eluent_programs = generate_seed_experiments(optimization_config, experiment_config, random_seed)
        for eluent_program in initial_eluent_programs[campaign.num_high_fidelity_experiments:]:

            if simulation:
                experiment = run_simulation_experiment(
                    simulator=instrument,
                    compound_set=analytes,
                    gradient=eluent_program,
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
            campaign.to_pkl(data_dir / "campaign.pkl")

    # Reload the retention model state for restarting the campaign (if applicable)
    lf_simulator, lf_compounds_fitted, lf_train_size = None, None, -1
    if campaign.retention_model_state is not None:
        state = campaign.retention_model_state
        lf_simulator = Simulator(**state["simulator"])
        lf_compounds_fitted = AnalyteSet.from_dicts(state["compounds"])
        lf_train_size = state["training_size"]
    current_lf_count = 0
    for experiment in reversed(campaign.experiments):
        if experiment.high_fidelity:
            break
        current_lf_count += 1

    while campaign.num_high_fidelity_experiments < optimization_config.budget:
        iteration_seed = random_seed + len(campaign.experiments)
        np.random.seed(iteration_seed), torch.manual_seed(iteration_seed)

        if optimization_config.needs_retention_model and lf_train_size != campaign.num_high_fidelity_experiments:
            try:
                lf_simulator, lf_compounds_fitted = get_simulator_from_data(
                    data=campaign,
                    column_length=instrument.parameters["column_length"] if simulation else optimization_config.column_length,
                    column_inner_diameter=optimization_config.column_inner_diameter,
                    simulation_interval=optimization_config.simulation_interval,
                    retention_model=optimization_config.retention_model,
                    previous_simulator=lf_simulator,
                    previous_compound_params=lf_compounds_fitted
                )
            except (ValueError, RuntimeError) as error:
                optimization_config.logger.warning(f"Retention fitting failed; using measured experiments: {error}")
                lf_simulator, lf_compounds_fitted = None, None
            lf_train_size = campaign.num_high_fidelity_experiments
            campaign.retention_model_state = {"simulator": lf_simulator.parameters, "compounds": lf_compounds_fitted.to_dicts(), "training_size": lf_train_size} if lf_simulator is not None else None

        high_fidelity, next_eluent_program = recommend_next_eluent_program(
            campaign=campaign,
            config=optimization_config,
            experiment_config=experiment_config,
            analytes=lf_compounds_fitted,
            simulator=lf_simulator,
            random_seed=iteration_seed
        )

        if current_lf_count >= optimization_config.max_consecutive_low_fidelity:
            high_fidelity = True
        if optimization_config.multi_fidelity and (lf_simulator is None or campaign.expected_fidelity_level <= 0):
            high_fidelity = True

        if high_fidelity:
            if simulation:
                experiment = run_simulation_experiment(
                    simulator=instrument,
                    compound_set=analytes,
                    gradient=next_eluent_program,
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
        else:
            experiment = run_simulation_experiment(
                simulator=lf_simulator,
                compound_set=lf_compounds_fitted,
                gradient=next_eluent_program,
                advanced_peak_detection=optimization_config.advanced_peak_detection,
                fidelity_level=campaign.expected_fidelity_level
            )
            experiment.simulation_training_size = lf_train_size
            current_lf_count += 1

        experiment.segment_based = optimization_config.segment_based
        experiment.score = optimization_config.calculate_experiment_score(experiment)
        campaign.append(experiment)
        optimization_config.logger.debug(f"Completed {'measured' if high_fidelity else 'simulated'} experiment with score {experiment.score:.2f}.")
        campaign.to_pkl(data_dir / "campaign.pkl")

    # Save the final campaign state and run a final simulation of the optimum gradient if applicable
    campaign.to_pkl(data_dir / "campaign.pkl")
    if simulation:
        run_simulation_experiment(
            simulator=instrument,
            compound_set=analytes,
            gradient=campaign.optimum_experiment.gradient,
            advanced_peak_detection=optimization_config.advanced_peak_detection,
            chromatogram_file=data_dir / "Optimum_Gradient.png"
        )

    return campaign
