from pathlib import Path
from tqdm import tqdm
import json
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
        log_k0_mean: float = 12.0,
        log_k0_std: float = 4.0,
        log_k0_distance_threshold: float = 8.0,
        s_1_relative_std: float = 0.2,
        s_2_relative_std: float = 0.1,
        random_seed: int = 0
) -> AnalyteSet:
    """
    Largely empirical method to sample a set of analytes (following a Neue-Kuss retention model) such that all
    compounds are likely separable.

    This is done as follows:
      1. Sample log_k0 values from a normal distribution.
      2. Sample S_1 from a normal distribution centered at 35.0 + 8.0 * log_k0 (empirically derived).
      3. Sample S_2 from a normal distribution centered at 4.05 + 0.19 * log_k0 (empirically derived).
      4. Accept a compound if the following criteria are met:
           - log_k0 is within 5.0 to 19.0
           - The log_k0 value is sufficiently distant from all previously accepted compounds, as defined by a
             logistic distance criterion: d = 3.0 - 2.5 / (1 + exp(-(log_k0 - log_k0_distance_threshold))). This ensures
             that for small log_k0 values, compounds are reasonably spaced, since otherwise, they elute too closely.
           - The compound elutes within 0.5 * max_run_time under an isocratic gradient at 100% B.
           - For each of the existing compounds, there is at least one isocratic gradient that separates the new
             compound from the existing compounds by a resolution of at least 1.5 and within 0.5 * max_run_time.

    Args:
        num_compounds (int): The number of compounds to sample.
        ref_simulator (Simulator): A reference simulator to use for elution time and resolution calculations.
        max_run_time (float): The maximum run time for the chromatographic method.
        log_k0_mean (float): The mean of the normal distribution for log_k0 sampling.
        log_k0_std (float): The standard deviation of the normal distribution for log_k0 sampling.
        log_k0_distance_threshold (float): The distance threshold parameter for the logistic distance criterion.
        s_1_relative_std (float): The relative standard deviation for S_1 sampling.
        s_2_relative_std (float): The relative standard deviation for S_2 sampling.
        random_seed (int): The random seed for reproducibility.

    Returns:
        AnalyteSet: The sampled set of analytes.
    """
    all_isocratic_gradients = [
        LCGradient(f"{i}", [0, max_run_time], [i / 100, i / 100], 0.4, eluents=("A", "B"))
        for i in range(101)
    ]
    retention_behavior = np.zeros((num_compounds, len(all_isocratic_gradients), 2))  # elution time, peak width

    cmpd_set = AnalyteSet()
    generator = torch.Generator(device="cpu").manual_seed(random_seed)

    while len(cmpd_set) < num_compounds:

        log_k0 = torch.normal(
            torch.tensor(log_k0_mean),
            torch.tensor(log_k0_std),
            generator=generator
        ).item()

        # Validate log_k0 range
        if not (5.0 <= log_k0 <= 19.0):
            continue

        # Validate log_k0 distance
        valid_dist = []
        for cmpd in cmpd_set:
            dist, ref_log_k0 = abs(log_k0 - cmpd.log_k0), min(cmpd.log_k0, log_k0)
            thresh = 2.0 - 1.75 / (1 + torch.exp(-1. * (torch.tensor(ref_log_k0) - log_k0_distance_threshold))).item()
            valid_dist.append(dist > thresh)
        if not all(valid_dist):
            continue

        # Sample S_1 and S_2
        mean_S_1 = 35.0 + 8.5 * log_k0
        S_1 = mean_S_1 * torch.normal(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(s_1_relative_std, dtype=torch.float64),
            generator=generator
        ).item()

        mean_S_2 = 4.05 + 0.19 * log_k0
        S_2 = mean_S_2 * torch.normal(
            torch.tensor(1.0, dtype=torch.float64),
            torch.tensor(s_2_relative_std, dtype=torch.float64),
            generator=generator
        ).item()

        area = 20.0 + 30.0 * torch.rand(1, generator=generator, dtype=torch.float64).item()

        # calculate retention behavior for all isocratic gradients
        temp_cmpd = Analyte(log_k0=log_k0, S_1=S_1, S_2=S_2, area=area)
        for grad_idx, grad in enumerate(all_isocratic_gradients):
            ref_simulator.gradient = grad
            t_red, sigma, _ = ref_simulator.calculate_peak(**temp_cmpd.to_dict())
            retention_behavior[len(cmpd_set), grad_idx, 0] = t_red.item()
            retention_behavior[len(cmpd_set), grad_idx, 1] = sigma.item()

        # Validate elution time under 100% B
        if retention_behavior[len(cmpd_set), -1, 0] > 0.5 * max_run_time:
            continue

        # Validate resolution against existing compounds
        valid_resolution = []
        for existing_idx in range(len(cmpd_set)):
            diff = retention_behavior[len(cmpd_set), :, 0] - retention_behavior[existing_idx, :, 0]
            denom = 4 * (retention_behavior[len(cmpd_set), :, 1] + retention_behavior[existing_idx, :, 1])
            resolutions = np.abs(diff) / denom
            valid = (resolutions > 1.5) & (retention_behavior[len(cmpd_set), :, 0] < 0.5 * max_run_time) & (retention_behavior[existing_idx, :, 0] < 0.5 * max_run_time)
            valid_resolution.append(np.any(valid))
        if not all(valid_resolution):
            continue

        # If all criteria are met, add the compound
        cmpd_set.append(temp_cmpd)

    return cmpd_set


def test_separability(
        cmpd_set: AnalyteSet,
        simulator: Simulator,
        config: HPLCOptimizationConfig,
        max_random_performance: float = 2.0,
        min_total_performance: float = 2.0
) -> tuple[float, float]:
    """
    Test the separability of the given compound set using the provided configuration using random gradients:
      - budget = 100  (random separatability)
      - budget = 10000 (general separability)

    Args:
        cmpd_set (AnalyteSet): The set of analytes to test.
        simulator (Simulator): The HPLC experiment emulator to simulate experiments.
        config (HPLCOptimizationConfig): The configuration for the optimization benchmark.
        max_random_performance (float): The maximum acceptable performance score for random gradients.
        min_total_performance (float): The minimum acceptable performance score for the best gradients.

    Returns:
        float: The performance score for the best gradients from 100 random gradients.
        float: The performance score for the best gradients from 10,000 random gradients.
    """
    config.budget = 100
    config.seed_budget = 100
    gradients_rand = generate_seed_experiments(config=config, random_seed=42)
    score_rand = float('-inf')

    for grad in gradients_rand:

        experiment = run_simulation_experiment(
            simulator=simulator,
            compound_set=cmpd_set,
            gradient=grad,
            resolution_threshold=config.resolution_threshold,
            advanced_peak_detection=config.advanced_peak_detection,
            fidelity_level=1.0,
        )
        experiment.segment_based = config.segment_based
        experiment.score = config.calculate_experiment_score(experiment)

        if experiment.score > score_rand:
            score_rand = experiment.score

    if score_rand > max_random_performance:
        return score_rand, float("inf")

    ################################################################################
    config.budget = 10000
    config.seed_budget = 10000
    gradients = generate_seed_experiments(config=config, random_seed=42)
    optimum_score = float('-inf')

    for grad in gradients:

        experiment = run_simulation_experiment(
            simulator=simulator,
            compound_set=cmpd_set,
            gradient=grad,
            resolution_threshold=config.resolution_threshold,
            advanced_peak_detection=config.advanced_peak_detection,
            fidelity_level=1.0,
        )
        experiment.segment_based = config.segment_based
        experiment.score = config.calculate_experiment_score(experiment)

        if experiment.score > optimum_score:
            optimum_score = experiment.score

        if optimum_score >= min_total_performance:
            break

    return score_rand, optimum_score


if __name__ == "__main__":

    NUM_COMPOUND_SETS = (5, )
    NUM_COMPOUNDS = (10, )
    MAX_RUN_TIMES = (20, )

    MAX_RANDOM_PERFORMANCE = 1.8
    MIN_TOTAL_PERFORMANCE = 2.2

    config_file = Path.cwd() / "configs" / "simulator_config.json"

    with open(config_file, "r") as f:
        simulator = Simulator(**json.load(f))

    for num_cmpd_sets in NUM_COMPOUND_SETS:
        for num_cmpds in NUM_COMPOUNDS:

            for max_run_time in MAX_RUN_TIMES:

                grad_config = HPLCOptimizationConfig(
                    num_compounds=num_cmpds,
                    num_segments=5,
                    min_segment_length=0.5,
                    max_run_time=max_run_time,
                    flow_rate=0.4,
                    advanced_peak_detection=True,
                    seed_strategy="random"
                )

                cmpd_sets, seed, too_easy, too_difficult = [], 0, 0, 0
                pbar = tqdm(total=num_cmpd_sets, desc=f"Generating {num_cmpd_sets} sets of {num_cmpds} compounds")

                while len(cmpd_sets) < num_cmpd_sets:

                    pbar.set_postfix({"Accepted": len(cmpd_sets), "Too Easy": too_easy, "Too Difficult": too_difficult})
                    seed += 1

                    cmpd_set = sample_analyte_set(
                        num_compounds=num_cmpds,
                        ref_simulator=simulator,
                        max_run_time=max_run_time,
                        random_seed=seed
                    )

                    random_sep, sep = test_separability(cmpd_set, simulator, grad_config, MAX_RANDOM_PERFORMANCE, MIN_TOTAL_PERFORMANCE)

                    if random_sep > MAX_RANDOM_PERFORMANCE:
                        too_easy += 1
                        continue
                    elif sep < MIN_TOTAL_PERFORMANCE:
                        too_difficult += 1
                        continue
                    else:
                        cmpd_sets.append(cmpd_set)
                        pbar.update(1)

                pbar.close()

                hash_val = hash(tuple(cmpd_sets)) % (10 ** 6)

                data_dir = Path.cwd() / "compound_sets" / f"{num_cmpd_sets}x{num_cmpds}C_{max_run_time}min_{hash_val}"
                data_dir.mkdir(parents=True, exist_ok=True)

                for idx, cmpd_set in enumerate(cmpd_sets):
                    file_path = data_dir / f"compound_set_{idx}.json"
                    cmpd_set.to_json(file_path)

                print(f"Created compound set '{data_dir.name}.")