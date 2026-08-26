from typing import Optional
from pathlib import Path
import json
import numpy as np

from moses.simulator import Simulator

from lcbo.data.campaign import HPLCOptimizationCampaign
from ..hplc_utils.analyte import AnalyteSet, Analyte


def get_simulator(config_file: Path) -> Simulator:
    """
    Initialize and return a Simulator object based on the provided configuration file and retention model.

    Args:
        config_file (Path): Path to the JSON configuration file containing simulator settings.

    Returns:
        Simulator: An instance of the Simulator class initialized with the specified settings.
    """
    with open(config_file, "r") as f:
        config = json.load(f)

    return Simulator(**config)


def get_simulator_from_data(
        data: HPLCOptimizationCampaign,
        column_length: float,
        simulation_interval: float,
        retention_model: str = "linear",
        previous_simulator: Optional[Simulator] = None,
        previous_compound_params: Optional[AnalyteSet] = None,
) -> tuple[Simulator, AnalyteSet]:
    """
    Fit a simulator based on the provided dataset and retention model. Can be used as a low-fidelity experiment.
    """
    runs = []
    valid_counts = [0 for _ in range(data.config.num_compounds)]

    for experiment in data.high_fidelity_experiments:

        for cmpd_idx in range(experiment.num_compounds):
            if not all(np.isfinite(i) for i in (experiment.retention_times[cmpd_idx], experiment.fwhms[cmpd_idx], experiment.peak_areas[cmpd_idx])):
                continue
            valid_counts[cmpd_idx] += 1

        runs.append(
            {
                "gradient": experiment.gradient,
                "retention_times": experiment.retention_times,
                "fwhms": experiment.fwhms,
                "peak_areas": experiment.peak_areas
            }
        )

    if not all(valid_counts):
        # Could be adapted to a minimum count of 2 for proper fitting?
        raise ValueError("Not all compounds have valid data for fitting the simulator.")

    data.config.logger.info("Deriving low-fidelity simulator based on existing experimental data. Data was validated "
                            "successfully for all compounds. Now fitting the retention model...")

    if previous_simulator is not None:
        simulator_initial_guess = previous_simulator.parameters
    else:
        simulator_initial_guess = None

    if previous_compound_params is not None:
        compounds_initial_guess = previous_compound_params.to_dicts()
    else:
        compounds_initial_guess = None

    try:
        simulator, cmpd_params = Simulator.from_data(
            retention_model,
            column_length,
            simulation_interval,
            runs=runs,
            simulator_initial_guess=simulator_initial_guess,
            compounds_initial_guess=compounds_initial_guess,
        )
    except RuntimeError as e:
        data.config.logger.error("Failed to fit low-fidelity simulator from data.")
        import pickle
        with open("simulator_fit_error_data.pkl", "wb") as f:
            pickle.dump(
                {
                    "retention_model": retention_model,
                    "column_length": column_length,
                    "simulation_interval": simulation_interval,
                    "runs": runs,
                },
                f
            )
        raise e

    cmpds_fitted = AnalyteSet()
    for params in cmpd_params:
        cmpds_fitted.append(Analyte(**params))

    data.config.logger.info("Low-fidelity simulator fitted successfully.")

    return simulator, cmpds_fitted
