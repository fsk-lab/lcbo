from typing import Optional
from pathlib import Path
import pickle
import numpy as np

from moses.simulator import Simulator
from lcbo.hplc_utils.analyte import AnalyteSet
from lcbo.hplc_utils.retention_prediction_error import retention_prediction_error

from lcbo.hplc_utils.hplc_experiment import HPLCExperiment
from lcbo.data.config import HPLCOptimizationConfig, HPLCOptimizationBenchmarkConfig


class HPLCOptimizationCampaign(object):
    """
    Class to manage a campaign of HPLC experiments, storing and analyzing multiple experiments.

    Args:
        config (HPLCOptimizationConfig): Configuration for the HPLC optimization campaign.
        experiments (list[HPLCExperiment], optional): List of HPLCExperiments (simulations or measured) in the campaign.
        retention_model_state (dict, optional): State of the retention model, if applicable.
    """

    def __init__(
            self,
            config: HPLCOptimizationConfig,
            experiments: Optional[list[HPLCExperiment]] = None,
            retention_model_state: Optional[dict] = None
    ):
        self.config = config
        self.experiments = experiments if experiments is not None else []
        self.retention_model_state = retention_model_state

    def append(self, experiment: HPLCExperiment):
        if experiment.num_compounds != self.config.num_compounds:
            raise ValueError("Experiment num_compounds does not match campaign num_compounds.")
        self.experiments.append(experiment)

    @property
    def num_high_fidelity_experiments(self) -> int:
        """
        Returns the number of high-fidelity experiments in the campaign.

        Returns:
            int: Number of high-fidelity experiments.
        """
        return len(self.high_fidelity_experiments)

    def experiments_above_fidelity(self, fidelity_threshold: float) -> list[HPLCExperiment]:
        """
        Returns experiments above the fidelity threshold in campaign order. Only low-fidelity results after
        the latest high-fidelity experiment belong to the current fit.

        Args:
            fidelity_threshold (float): The fidelity level threshold.

        Returns:
            list[HPLCExperiment]: List of experiments above the fidelity threshold.
        """
        experiments = []
        after_latest_high_fidelity = True
        for experiment in reversed(self.experiments):
            if experiment.high_fidelity:
                after_latest_high_fidelity = False
            elif not after_latest_high_fidelity:
                continue
            if experiment.fidelity_level >= fidelity_threshold - 1E-3:
                experiments.append(experiment)
        experiments.reverse()
        return experiments

    @property
    def high_fidelity_experiments(self) -> list[HPLCExperiment]:
        """
        Returns a list of high-fidelity experiments in the campaign.

        Returns:
            list[HPLCExperiment]: List of high-fidelity experiments.
        """
        return [exp for exp in self.experiments if exp.high_fidelity]

    @property
    def scores(self) -> np.ndarray:
        """
        Returns an array of scores from all experiments in the campaign.

        Returns:
            np.ndarray: Array of scores (shape: (num_experiments, 1)).
        """
        return np.array([exp.score for exp in self.high_fidelity_experiments]).reshape(-1, 1)

    @property
    def score_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the trajectory of scores over the course of the campaign, i.e., the maximum score observed up to each
        experiment, as well as the indices of the experiments that correspond to these maximum scores.

        Returns:
            np.ndarray: Array of maximum scores up to each experiment (shape: (num_experiments, 1)).
            np.ndarray: Array of indices of the experiments corresponding to the maximum scores
                        (shape: (num_experiments, 1)).
        """
        traj = np.maximum.accumulate(self.scores.flatten())
        eq = self.scores.flatten() == traj
        idx_candidates = np.where(eq, np.arange(len(self.high_fidelity_experiments)), 0)
        indices = np.maximum.accumulate(idx_candidates)
        return traj.reshape(-1, 1), indices.reshape(-1, 1)

    def get_gradient_parameters(self, fidelity_threshold: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns an array of gradient parameters from all experiments in the campaign.

        Args:
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of gradient parameters (shape: (num_hf_experiments, num_params) if high_fidelity_only is
                        True, else (num_experiments, num_params)).
            np.ndarray: Array of fidelity levels (shape: (num_experiments, 1)).

        """
        params, fidelities = [], []

        for exp in self.experiments_above_fidelity(fidelity_threshold):
            params.append(exp.gradient_parameters.parameters.reshape(1, -1))
            fidelities.append(exp.fidelity_level)

        if not params:
            return np.empty((0, self.config.num_optimization_variables)), np.empty((0, 1))
        return np.concatenate(params, axis=0), np.array(fidelities).reshape(-1, 1)

    def get_gradient_run_times(self, fidelity_threshold: float = 1.0) -> np.ndarray:
        """
        Returns an array of gradient run times from all experiments in the campaign.

        Args:
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of gradient run times (shape: (num_experiments, 1)).
        """
        experiments = self.experiments_above_fidelity(fidelity_threshold)
        return np.array([exp.gradient.run_time for exp in experiments]).reshape(-1, 1)

    def get_retention_times(self, fidelity_threshold: float = 1.0) -> np.ndarray:
        """
        Returns an array of compound-wise retention times from all experiments in the campaign.

        Args:
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of retention times (shape: (num_experiments, num_compounds)).
        """
        experiments = self.experiments_above_fidelity(fidelity_threshold)
        return np.array([exp.retention_times for exp in experiments])

    def get_peak_fwhms(self, fidelity_threshold: float = 1.0) -> np.ndarray:
        """
        Returns an array of compound-wise peak widths from all experiments in the campaign.

        Args:
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of peak widths (shape: (num_experiments, num_compounds)).
        """
        experiments = self.experiments_above_fidelity(fidelity_threshold)
        return np.array([exp.fwhms for exp in experiments])

    def get_num_peaks(self, fidelity_threshold: float = 1.0) -> np.ndarray:
        """
        Returns an array of number of detected peaks from all experiments in the campaign.

        Args:
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of number of detected peaks (shape: (num_experiments, 1)).
        """
        experiments = self.experiments_above_fidelity(fidelity_threshold)
        return np.array([exp.num_peaks for exp in experiments]).reshape(-1, 1)

    @property
    def num_peaks_trajectory(self) -> np.ndarray:
        """
        Returns the trajectory of number of detected peaks over the course of the campaign, i.e., the number of peaks
        at the "best" experiment up to each point in the campaign. The best experiment is given by the highest score.

        Returns:
            np.ndarray: Array of maximum number of peaks up to each experiment (shape: (num_experiments, 1)).
        """
        _, indices = self.score_trajectory
        return self.get_num_peaks(fidelity_threshold=1.0)[indices.flatten()]

    def get_critical_resolutions(self, fidelity_threshold: float = 1.0) -> np.ndarray:
        """
        Returns an array of critical resolutions from all experiments in the campaign.

        Args:
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of critical resolutions (shape: (num_experiments, 1)).
        """
        experiments = self.experiments_above_fidelity(fidelity_threshold)
        return np.array([exp.critical_resolution for exp in experiments]).reshape(-1, 1)

    @property
    def critical_resolution_trajectory(self) -> np.ndarray:
        """
        Returns the trajectory of critical resolutions over the course of the campaign, i.e., the critical resolution
        at the "best" experiment up to each point in the campaign. The best experiment is given by the highest score.

        Returns:
            np.ndarray: Array of maximum critical resolutions up to each experiment (shape: (num_experiments, 1)).
        """
        _, indices = self.score_trajectory
        return self.get_critical_resolutions(fidelity_threshold=1.0)[indices.flatten()]

    def get_num_resolved_peaks(self, fidelity_threshold: float = 1.0) -> np.ndarray:
        """
        Returns an array of number of resolved peaks from all experiments in the campaign.

        Args:
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of number of resolved peaks (shape: (num_experiments, 1)).
        """
        experiments = self.experiments_above_fidelity(fidelity_threshold)
        return np.array([exp.num_resolved_peaks(self.config.resolution_threshold) for exp in experiments]).reshape(-1, 1)

    @property
    def num_resolved_peaks_trajectory(self) -> np.ndarray:
        """
        Returns the trajectory of number of resolved peaks over the course of the campaign, i.e., the number of resolved
        peaks at the "best" experiment up to each point in the campaign. The best experiment is given by the highest
        score.

        Returns:
            np.ndarray: Array of maximum number of resolved peaks up to each experiment (shape: (num_experiments, 1)).
        """
        _, indices = self.score_trajectory
        return self.get_num_resolved_peaks(fidelity_threshold=1.0)[indices.flatten()]

    def get_single_compound_elution(self, compound_idx: int, fidelity_threshold: float = 1.0) -> np.ndarray:
        """
        Returns an array of elution status (0: not eluted, 1: eluted) for a specific compound across all experiments in
        the campaign.

        Args:
            compound_idx (int): The index of the compound to evaluate.
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of elution status for the specified compound (shape: (num_experiments, 1)).
        """
        experiments = self.experiments_above_fidelity(fidelity_threshold)
        return np.array([exp.single_peak_elution_resolution(compound_idx)[0] for exp in experiments]).reshape(-1, 1)

    def single_compound_elution_trajectory(self, compound_idx: int) -> np.ndarray:
        """
        Returns the trajectory of elution status for a specific compound over the course of the campaign, i.e., the 
        elution status at the "best" experiment up to each point in the campaign. The best experiment is given by the 
        highest score.
        
        Args:
            compound_idx (int): The index of the compound to evaluate.
            
        Returns:
            np.ndarray: Array of elution status for the specified compound at the best experiments up to each experiment
                        (shape: (num_experiments, 1)).
        """
        _, indices = self.score_trajectory
        return self.get_single_compound_elution(compound_idx=compound_idx, fidelity_threshold=1.0)[indices.flatten()]

    def get_single_compound_resolutions(self, compound_idx: int, fidelity_threshold: float = 1.0) -> np.ndarray:
        """
        Returns an array of resolutions for a specific compound across all experiments in the campaign. If the compound
        is not eluted in an experiment, the resolution is set to 0.

        Args:
            compound_idx (int): The index of the compound to evaluate.
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of resolutions for the specified compound (shape: (num_experiments, 1)).
        """
        experiments = self.experiments_above_fidelity(fidelity_threshold)
        return np.array([exp.single_peak_elution_resolution(compound_idx)[1] for exp in experiments]).reshape(-1, 1)

    def single_compound_resolution_trajectory(self, compound_idx: int) -> np.ndarray:
        """
        Returns the trajectory of resolutions for a specific compound over the course of the campaign, i.e., the
        resolution at the "best" experiment up to each point in the campaign. The best experiment is given by the
        highest score.

        Args:
            compound_idx (int): The index of the compound to evaluate.

        Returns:
            np.ndarray: Array of resolutions for the specified compound at the best experiments up to each experiment
                        (shape: (num_experiments, 1)).
        """
        _, indices = self.score_trajectory
        return self.get_single_compound_resolutions(compound_idx=compound_idx, fidelity_threshold=1.0)[indices.flatten()]

    def get_run_times(self, fidelity_threshold: float = 1.0) -> np.ndarray:
        """
        Returns an array of run times from all experiments in the campaign.

        Args:
            fidelity_threshold (float): Includes all experiments with fidelity level >= this threshold.

        Returns:
            np.ndarray: Array of run times (shape: (num_experiments, 1)).
        """
        experiments = self.experiments_above_fidelity(fidelity_threshold)
        return np.array([exp.gradient.run_time for exp in experiments]).reshape(-1, 1)

    @property
    def run_time_trajectory(self) -> np.ndarray:
        """
        Returns the trajectory of run times over the course of the campaign, i.e., the run time at the "best" experiment
        up to each point in the campaign. The best experiment is given by the highest score.

        Returns:
            np.ndarray: Array of run times at the best experiments up to each experiment (shape: (num_experiments,)).
        """
        _, indices = self.score_trajectory
        return self.get_run_times(fidelity_threshold=1.0)[indices.flatten()]

    @property
    def expected_fidelity_level(self) -> float:
        """
        Returns the expected fidelity level of the fitted simulator. The prediction_error strategy evaluates the
        current fit on all measured gradients and uses the resulting in-sample retention error. The experiment_count
        strategy uses an exponential saturation model.

        s = max_simulation_fidelity * (1 - exp(-simulation_fidelity_decay * n_hf))

        where n_hf is the number of high-fidelity experiments in the campaign.

        Returns:
            float: Expected simulation fidelity for a multi-fidelity campaign.
        """
        if self.config.single_fidelity:
            raise ValueError("Expected fidelity level is not defined for single-fidelity campaigns.")

        if self.config.fidelity_strategy == "prediction_error":
            experiments = self.high_fidelity_experiments
            if not experiments or self.retention_model_state is None:
                return 0.0

            from lcbo.routines.experiment import run_simulation_experiment

            simulator = Simulator(**self.retention_model_state["simulator"])
            compounds = AnalyteSet.from_dicts(self.retention_model_state["compounds"])
            errors = []
            for observed in experiments:
                predicted = run_simulation_experiment(
                    simulator, compounds, observed.gradient, advanced_peak_detection=self.config.advanced_peak_detection
                )
                error = retention_prediction_error(predicted, observed)
                if error is not None and np.isfinite(error):
                    errors.append(error)

            if not errors:
                return 0.0

            rms_error = np.sqrt(np.mean(np.square(errors)))
            return float(self.config.max_simulation_fidelity * np.exp(-rms_error / self.config.retention_error_scale))

        elif self.config.fidelity_strategy == "experiment_count":
            n_hf = self.num_high_fidelity_experiments
            return self.config.max_simulation_fidelity * (1 - np.exp(-self.config.simulation_fidelity_decay * n_hf))

        else:
            raise ValueError(f"Unknown fidelity strategy: {self.config.fidelity_strategy}")

    @property
    def optimum_experiment(self) -> HPLCExperiment:
        """
        Returns the experiment with the highest score in the campaign.

        Returns:
            HPLCExperiment: The experiment with the highest score.
        """
        max_idx = np.argmax(self.scores)
        return self.high_fidelity_experiments[max_idx]

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "experiments": [exp.to_dict() for exp in self.experiments],
            "retention_model_state": self.retention_model_state
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HPLCOptimizationCampaign":
        config = HPLCOptimizationConfig.from_dict(data["config"])
        experiments = [HPLCExperiment.from_dict(exp_data) for exp_data in data["experiments"]]
        return cls(config=config, experiments=experiments, retention_model_state=data.get("retention_model_state"))

    def to_pkl(self, file: Path) -> None:
        with open(file, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def from_pkl(cls, file: Path) -> "HPLCOptimizationCampaign":
        with open(file, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)


class HPLCOptimizationBenchmark(object):
    """
    Class to hold benchmark results for multiple HPLC optimization campaigns.
    """
    def __init__(
            self,
            config: HPLCOptimizationBenchmarkConfig,
            campaigns: Optional[list[list[HPLCOptimizationCampaign]]] = None
    ):

        self.config = config

        if campaigns:
            if len(campaigns) != config.num_compound_sets:
                raise ValueError("Number of campaign sets must match number of compound sets in config.")
            for campaign_set in campaigns:
                if len(campaign_set) != config.iterations_per_compound_set:
                    raise ValueError("Number of campaigns per set must match iterations_per_compound_set in config.")
            self.campaigns = campaigns
        else:
            self.campaigns = [[] for _ in range(config.num_compound_sets)]

    def add_campaign(self, compound_set_index: int, campaign: HPLCOptimizationCampaign):
        """
        Add a new campaign to the benchmark for a specific compound set index.

        Args:
            compound_set_index (int): Index of the compound set to which the campaign belongs.
            campaign (HPLCOptimizationCampaign): The campaign to add.

        Raises:
            IndexError: If the compound_set_index is out of range.
            ValueError: If the number of campaigns for the compound set exceeds iterations_per_compound_set.
        """
        if compound_set_index < 0 or compound_set_index >= self.config.num_compound_sets:
            raise IndexError("Analyte set index out of range.")
        if len(self.campaigns[compound_set_index]) >= self.config.iterations_per_compound_set:
            raise ValueError("Cannot add more campaigns to this compound set; limit reached.")
        self.campaigns[compound_set_index].append(campaign)

    def get_score_trajectories(self, compound_wise: bool = False) -> np.ndarray | list[np.ndarray]:
        """
        Get the score trajectories for all campaigns in the benchmark.

        Args:
            compound_wise (bool): If True, return trajectories for each compound set separately.
                                  If False, return a combined trajectory across all compound sets.

        Returns:
            np.ndarray: Single array of score trajectories (if `compound_wise` is False).
                        Shape: (num_campaigns, max_trajectory_length).
            list[np.ndarray]: List of arrays of score trajectories for each compound set (if `compound_wise` is True).
                              Each array has shape: (iterations_per_compound_set, max_trajectory_length).
        """
        trajectories = []
        for campaign_set in self.campaigns:
            set_trajectories = []
            for campaign in campaign_set:
                traj, _ = campaign.score_trajectory
                set_trajectories.append(traj.flatten())
            trajectories.append(np.array(set_trajectories))

        if compound_wise:
            return trajectories
        else:
            return np.concatenate(trajectories, axis=0)

    def get_num_peaks_trajectories(self, compound_wise: bool = False) -> np.ndarray | list[np.ndarray]:
        """
        Get the number of peaks trajectories for all campaigns in the benchmark.

        Args:
            compound_wise (bool): If True, return trajectories for each compound set separately.
                                  If False, return a combined trajectory across all compound sets.

        Returns:
            np.ndarray: Single array of number of peaks trajectories (if `compound_wise` is False).
                        Shape: (num_campaigns, max_trajectory_length).
            list[np.ndarray]: List of arrays of number of peaks trajectories for each compound set (if `compound_wise` is True).
                              Each array has shape: (iterations_per_compound_set, max_trajectory_length).
        """
        trajectories = []
        for campaign_set in self.campaigns:
            set_trajectories = []
            for campaign in campaign_set:
                traj = campaign.num_peaks_trajectory
                set_trajectories.append(traj.flatten())
            trajectories.append(np.array(set_trajectories))

        if compound_wise:
            return trajectories
        else:
            return np.concatenate(trajectories, axis=0)

    def get_critical_resolution_trajectories(self, compound_wise: bool = False) -> np.ndarray | list[np.ndarray]:
        """
        Get the critical resolution trajectories for all campaigns in the benchmark.

        Args:
            compound_wise (bool): If True, return trajectories for each compound set separately.
                                  If False, return a combined trajectory across all compound sets.

        Returns:
            np.ndarray: Single array of critical resolution trajectories (if `compound_wise` is False).
                        Shape: (num_campaigns, max_trajectory_length).
            list[np.ndarray]: List of arrays of critical resolution trajectories for each compound set (if `compound_wise` is True).
                              Each array has shape: (iterations_per_compound_set, max_trajectory_length).
        """
        trajectories = []
        for campaign_set in self.campaigns:
            set_trajectories = []
            for campaign in campaign_set:
                traj = campaign.critical_resolution_trajectory
                set_trajectories.append(traj.flatten())
            trajectories.append(np.array(set_trajectories))

        if compound_wise:
            return trajectories
        else:
            return np.concatenate(trajectories, axis=0)

    def get_num_resolved_peaks_trajectories(self, compound_wise: bool = False) -> np.ndarray | list[np.ndarray]:
        """
        Get the number of resolved peaks trajectories for all campaigns in the benchmark.

        Args:
            compound_wise (bool): If True, return trajectories for each compound set separately.
                                  If False, return a combined trajectory across all compound sets.

        Returns:
            np.ndarray: Single array of number of resolved peaks trajectories (if `compound_wise` is False).
                        Shape: (num_campaigns, max_trajectory_length).
            list[np.ndarray]: List of arrays of number of resolved peaks trajectories for each compound set (if `compound_wise` is True).
                              Each array has shape: (iterations_per_compound_set, max_trajectory_length).
        """
        trajectories = []
        for campaign_set in self.campaigns:
            set_trajectories = []
            for campaign in campaign_set:
                traj = campaign.num_resolved_peaks_trajectory
                set_trajectories.append(traj.flatten())
            trajectories.append(np.array(set_trajectories))

        if compound_wise:
            return trajectories
        else:
            return np.concatenate(trajectories, axis=0)

    def get_run_time_trajectories(self, compound_wise: bool = False) -> np.ndarray | list[np.ndarray]:
        """
        Get the run time trajectories for all campaigns in the benchmark.

        Args:
            compound_wise (bool): If True, return trajectories for each compound set separately.
                                  If False, return a combined trajectory across all compound sets.

        Returns:
            np.ndarray: Single array of run time trajectories (if `compound_wise` is False).
                        Shape: (num_campaigns, max_trajectory_length).
            list[np.ndarray]: List of arrays of run time trajectories for each compound set (if `compound_wise` is True).
                              Each array has shape: (iterations_per_compound_set, max_trajectory_length).
        """
        trajectories = []
        for campaign_set in self.campaigns:
            set_trajectories = []
            for campaign in campaign_set:
                traj = campaign.run_time_trajectory
                set_trajectories.append(traj.flatten())
            trajectories.append(np.array(set_trajectories))

        if compound_wise:
            return trajectories
        else:
            return np.concatenate(trajectories, axis=0)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "campaigns": [
                [campaign.to_dict() for campaign in campaign_set]
                for campaign_set in self.campaigns
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HPLCOptimizationBenchmark":

        config = HPLCOptimizationBenchmarkConfig.from_dict(data["config"])

        campaigns = []
        for _campaign_set in data["campaigns"]:
            campaign_set = []
            for campaign in _campaign_set:
                campaign_set.append(HPLCOptimizationCampaign.from_dict(campaign))
            campaigns.append(campaign_set)

        return HPLCOptimizationBenchmark(config=config, campaigns=campaigns)

    def to_pkl(self, file: Path) -> None:
        with open(file, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def from_pkl(cls, file: Path) -> "HPLCOptimizationBenchmark":
        with open(file, "rb") as f:
            data = pickle.load(f)

        return cls.from_dict(data)
