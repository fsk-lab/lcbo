from pathlib import Path
import copy
from tqdm import tqdm
import json
import argparse
import numpy as np
import torch
from moses.simulator import Simulator
from moses.datatypes import LCGradient
from lcbo.data.config import HPLCOptimizationConfig
from lcbo.hplc_utils.analyte import Analyte, AnalyteSet
from lcbo.routines.initial_experiments import generate_seed_experiments
from lcbo.routines.experiment import run_simulation_experiment


def sample_analyte_set(
        num_compounds: int,
        ref_simulator: Simulator,
        max_run_time: float,
        log_k0_mean: float = 8.0,
        log_k0_std: float = 3.0,
        log_k0_bounds: tuple[float, float] = (2.0, 16.0),
        log_k1_mean: float = -1.0,
        log_k1_bounds: tuple[float, float] = (-5.0, 1.0),
        s_1_relative_std: float = 0.2,
        s_1_bounds: tuple[float, float] = (1.0, 100.0),
        s_2_mean: float = 1.0,
        s_2_relative_std: float = 0.5,
        s_2_bounds: tuple[float, float] = (0.0, 3.0),
        random_seed: int = 0,
        flow_rate: float = 0.4,
        max_attempts: int = 10000
) -> AnalyteSet:
    """
    Sample separable analytes from a synthetic Neue-Kuss prior.

    Proceeds as follows:
      1. Sample log_k0 from a bounded normal distribution.
      2. Sample S_2 independently from a bounded log-normal distribution.
      3. Sample S_1 from a log-normal distribution centered at (1 + S_2) * (log_k0 + 2 * log(1 + S_2) - log_k1_mean).
      4. Accept a compound if the following criteria are met:
         - The compound elutes within half of max_run_time at 100% B.
         - For each of the existing compounds, there is at least one isocratic gradient that separates the new compound
           from the existing compound by a resolution of >1.5 within half of max_run_time.

    Args:
        num_compounds (int): Number of analytes to sample.
        ref_simulator (Simulator): Reference simulator using the Neue-Kuss retention model.
        max_run_time (float): Maximum chromatographic run time in minutes.
        log_k0_mean (float): Mean of the normal proposal for log(k at 0% B).
        log_k0_std (float): Standard deviation of that proposal.
        log_k0_bounds (tuple[float, float]): Accepted log(k at 0% B) range.
        log_k1_mean (float): Mean of the normal proposal for log(k at 100% B).
        log_k1_bounds (tuple[float, float]): Accepted log(k at 100% B) range.
        s_1_relative_std (float): Coefficient of variation of the log-normal S_1 proposal.
        s_1_bounds (tuple[float, float]): Accepted S_1 value range.
        s_2_mean (float): Mean of the S_2 proposal..
        s_2_relative_std (float): Coefficient of variation of the log-normal S_2 proposal.
        s_2_bounds (tuple[float, float]): Accepted S_2 value range.
        random_seed (int): Random seed for reproducible sampling.
        flow_rate (float): Flow rate in mL/min for the separability checks.
        max_attempts (int): Maximum total candidate draws before reporting an infeasible sampling request.

    Returns:
        AnalyteSet: Sampled analytes satisfying parameter and pairwise separability checks.
    """
    all_isocratic_gradients = [
        LCGradient(f"{i}", [0, max_run_time], [i / 100, i / 100], flow_rate, eluents=("A", "B"))
        for i in range(101)
    ]
    retention_behavior = np.zeros((num_compounds, len(all_isocratic_gradients), 2))  # elution time, peak width

    cmpd_set = AnalyteSet()
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    rejections = dict(log_k0=0, S_2=0, S_1=0, log_k1=0, elution=0, resolution=0)

    def sample_positive_log_normal(mean: float, relative_std: float) -> float:
        sigma = np.sqrt(np.log1p(relative_std**2))
        z = torch.randn((), generator=generator, dtype=torch.float64).item()
        return float(mean * np.exp(sigma * z - 0.5 * sigma**2))

    for _ in range(max_attempts):

        # sample log_k_0 from the bounded normal distribution
        log_k0 = log_k0_mean + log_k0_std * torch.randn((), generator=generator, dtype=torch.float64).item()
        if not log_k0_bounds[0] <= log_k0 <= log_k0_bounds[1]:
            rejections["log_k0"] += 1
            continue

        # sample S_2 from the bounded log-normal distribution
        S_2 = sample_positive_log_normal(s_2_mean, s_2_relative_std)
        if not s_2_bounds[0] <= S_2 <= s_2_bounds[1]:
            rejections["S_2"] += 1
            continue

        # sample S_1 from a bounded distribution conditional on the sampled S_1 and S_2 values
        mean_S_1 = (1 + S_2) * (log_k0 + 2 * np.log1p(S_2) - log_k1_mean)
        if mean_S_1 <= 0:
            rejections["S_1"] += 1
            continue
        S_1 = sample_positive_log_normal(mean_S_1, s_1_relative_std)
        if not s_1_bounds[0] <= S_1 <= s_1_bounds[1] or S_1 < 2 * S_2 * (1 + S_2):
            rejections["S_1"] += 1
            continue
        log_k1 = log_k0 + 2 * np.log1p(S_2) - S_1 / (1 + S_2)
        if not log_k1_bounds[0] <= log_k1 <= log_k1_bounds[1]:
            rejections["log_k1"] += 1
            continue

        area = 20.0 + 30.0 * torch.rand(1, generator=generator, dtype=torch.float64).item()

        # calculate retention behavior for all isocratic gradients
        temp_cmpd = Analyte(log_k0=log_k0, S_1=S_1, S_2=S_2, area=area)
        for grad_idx, grad in enumerate(all_isocratic_gradients):
            ref_simulator.gradient = grad
            t_red, sigma, _ = ref_simulator.calculate_peak(**temp_cmpd.to_dict())
            retention_behavior[len(cmpd_set), grad_idx, 0] = t_red.item()
            retention_behavior[len(cmpd_set), grad_idx, 1] = sigma.item()

        # Validate elution time under 100% B
        if not np.all(np.isfinite(retention_behavior[len(cmpd_set), -1])) or retention_behavior[len(cmpd_set), -1, 0] > 0.5 * max_run_time:
            rejections["elution"] += 1
            continue

        # Validate resolution against existing compounds
        valid_resolution = []
        for existing_idx in range(len(cmpd_set)):
            with np.errstate(invalid="ignore"):
                diff = retention_behavior[len(cmpd_set), :, 0] - retention_behavior[existing_idx, :, 0]
            denom = 2 * (retention_behavior[len(cmpd_set), :, 1] + retention_behavior[existing_idx, :, 1])
            resolutions = np.abs(diff) / denom
            valid = (resolutions > 1.5) & (retention_behavior[len(cmpd_set), :, 0] < 0.5 * max_run_time) & (retention_behavior[existing_idx, :, 0] < 0.5 * max_run_time)
            valid_resolution.append(np.any(valid))
        if not all(valid_resolution):
            rejections["resolution"] += 1
            continue

        # If all criteria are met, add the compound
        cmpd_set.append(temp_cmpd)
        if len(cmpd_set) == num_compounds:
            return cmpd_set

    raise RuntimeError(f"Only sampled {len(cmpd_set)} of {num_compounds} analytes after {max_attempts} attempts. Rejections: {rejections}")


def test_separability(
        cmpd_set: AnalyteSet,
        simulator: Simulator,
        config: HPLCOptimizationConfig,
        num_gradients: int = 10000
) -> np.ndarray:
    """
    Test the separability of the given compound set using 10,000 random gradients and return the performance scores.

    Args:
        cmpd_set (AnalyteSet): The set of analytes to test.
        simulator (Simulator): The HPLC experiment emulator to simulate experiments.
        config (HPLCOptimizationConfig): The configuration for the optimization benchmark.
        num_gradients (int): The number of random gradients to generate and test.

    Returns:
        np.ndarray: An array of performance scores for each of the 10,000 random gradients.
    """
    config = copy.deepcopy(config)
    config.budget = num_gradients
    config.seed_budget = num_gradients
    gradients = generate_seed_experiments(config=config, random_seed=42)

    scores = np.empty(len(gradients), dtype=np.float32)

    for idx, grad in enumerate(gradients):

        experiment = run_simulation_experiment(
            simulator=simulator,
            compound_set=cmpd_set,
            gradient=grad,
            resolution_threshold=config.resolution_threshold,
            advanced_peak_detection=config.advanced_peak_detection,
            fidelity_level=1.0,
        )
        experiment.segment_based = config.segment_based
        scores[idx] = config.calculate_experiment_score(experiment)

    return scores


if __name__ == "__main__":

    VERBOSE = True
    NUM_COMPOUND_SETS = 500
    NUM_COMPOUND_SETS_TO_KEEP = 50
    NUM_COMPOUNDS = (5, 10, 15, 20, 25)
    MAX_RUN_TIMES = (10, 20, 30, 45, 60)

    NUM_GRADIENTS = 10000
    BUDGET = 100
    PERFORMANCE_THRESHOLD = 2.2

    parser = argparse.ArgumentParser(description="Generate analyte sets for HPLC optimization benchmarks.")
    parser.add_argument("--num_compounds", type=int, help="Numbers of compounds to sample.")
    parser.add_argument("--max_run_time", type=int, help="Maximum run time for the HPLC experiment in minutes.")
    args = parser.parse_args()
    num_cmpds, max_run_time = args.num_compounds, args.max_run_time

    config_file = Path.cwd() / "configs" / "simulator_config.json"

    with open(config_file, "r") as f:
        simulator = Simulator(**json.load(f))

    stats = []

    grad_config = HPLCOptimizationConfig(
        num_compounds=num_cmpds,
        num_segments=5,
        min_segment_length=0.5,
        max_run_time=max_run_time,
        flow_rate=0.4,
        advanced_peak_detection=True,
        seed_strategy="random"
    )

    cmpd_sets_to_keep = []
    seed = 0
    too_easy, too_difficult = 0, 0

    pbar = tqdm(total=NUM_COMPOUND_SETS, desc=f"Sampling {NUM_COMPOUND_SETS} sets of {num_cmpds} compounds for {max_run_time} min run time")

    for cmpd_set_idx in range(NUM_COMPOUND_SETS):

        pbar.set_postfix({"Accepted": len(cmpd_sets_to_keep), "Too Easy": too_easy, "Too Difficult": too_difficult})
        seed += 1

        try:
            cmpd_set = sample_analyte_set(
                num_compounds=num_cmpds,
                ref_simulator=simulator,
                max_run_time=max_run_time,
                random_seed=seed
            )
            scores = test_separability(cmpd_set, simulator, grad_config, num_gradients=NUM_GRADIENTS)
            stats.append(scores)

        except RuntimeError as e:
            print(f"Failed to sample compound set {cmpd_set_idx + 1}: {e}")
            stats.append(np.full(NUM_GRADIENTS, np.nan, dtype=np.float32))
            pbar.update(1)
            continue

        if len(cmpd_sets_to_keep) < NUM_COMPOUND_SETS_TO_KEEP:
            if np.all(scores < PERFORMANCE_THRESHOLD):
                too_difficult += 1
                pbar.update(1)
                continue
            if np.sum(scores > PERFORMANCE_THRESHOLD) > NUM_GRADIENTS / BUDGET:
                too_easy += 1
                pbar.update(1)
                continue
            cmpd_sets_to_keep.append(cmpd_set)

    pbar.close()

    stats = np.array(stats)

    # Store the generated compound sets

    hash_val = hash(tuple(cmpd_sets_to_keep)) % (10 ** 6)
    data_dir = Path.cwd() / "compound_sets" / f"{len(cmpd_sets_to_keep)}x{num_cmpds}C_{max_run_time}min_{hash_val}"
    data_dir.mkdir(parents=True, exist_ok=True)

    for idx, cmpd_set in enumerate(cmpd_sets_to_keep):
        file_path = data_dir / f"compound_set_{idx}.json"
        cmpd_set.to_json(file_path)

    print(f"Created compound set '{data_dir.name}.")

    np.savez(data_dir / "compound_set_stats.npz", stats=stats, too_easy=too_easy, too_difficult=too_difficult)
