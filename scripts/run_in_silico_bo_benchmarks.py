import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
from joblib import Parallel, delayed
from lcbo.data.config import HPLCOptimizationBenchmarkConfig
from lcbo.data.campaign import HPLCOptimizationBenchmark, HPLCOptimizationCampaign
from lcbo.routines import run_optimization_campaign
from lcbo.hplc_utils.simulator import get_simulator, Simulator
from lcbo.hplc_utils.analyte import AnalyteSet


def run_benchmarks_for_compound_set(
        emulator: Simulator,
        config: HPLCOptimizationBenchmarkConfig,
        compound_set_idx: int,
) -> list[HPLCOptimizationCampaign]:
    """
    Runs a series of optimization campaigns for a specific compound set index, using the provided experiment emulator
    and configuration. Each campaign is executed independently for a number of iterations defined in the configuration.

    Args:
        emulator (Simulator): The HPLC experiment emulator to simulate experiments.
        config (HPLCOptimizationBenchmarkConfig): The configuration for the optimization benchmark.
        compound_set_idx (int): The index of the compound set to be optimized.
    """
    torch.set_num_threads(1)

    results = []

    config.logger.info(f"Analyte Set {compound_set_idx + 1} / {config.num_compound_sets}. Performing "
                       f"{config.iterations_per_compound_set} independent optimization runs with a budget of "
                       f"{config.budget} experiments each.")

    compounds = AnalyteSet.from_json(config.compound_sets_folder / f"compound_set_{compound_set_idx}.json")

    for iter_idx in range(config.iterations_per_compound_set):
        data_dir = config.working_dir / f"CmpdSet-{compound_set_idx}_Iter-{iter_idx}"
        data_dir.mkdir(parents=True, exist_ok=True)

        campaign = run_optimization_campaign(
            simulation=True,
            instrument=emulator,
            optimization_config=config.optimization_config,
            analytes=compounds,
            data_dir=data_dir,
            random_seed=iter_idx
        )
        results.append(campÏaign)

    config.logger.info(f"Completed all optimization runs for Analyte Set {compound_set_idx + 1}.\n")

    return results


if __name__ == "__main__":

    data_dir = Path.cwd() / "results" / "in_silico_bo_benchmarks"

    num_jobs = 25  # number of parallel CPU jobs to run
    config = HPLCOptimizationBenchmarkConfig.from_argparse()
    config.optimization_config.working_dir = data_dir / config.tag
    results = HPLCOptimizationBenchmark(config)
    emulator = get_simulator(config.experiment_config)

    config.logger.info(f"Starting {config.tag}")

    parallelizer = Parallel(n_jobs=num_jobs, backend="loky")
    campaigns = parallelizer(
        delayed(run_benchmarks_for_compound_set)(
            emulator=emulator,
            config=config,
            compound_set_idx=cmpd_set_idx
        ) for cmpd_set_idx in range(config.num_compound_sets)
    )

    for cmpd_set_idx, cmpd_set_campaigns in enumerate(campaigns):
        for campaign in cmpd_set_campaigns:
            results.add_campaign(cmpd_set_idx, campaign)

    results.to_pkl(config.working_dir / "results.pkl")

    config.logger.info("Benchmarking completed. Saving results.\n\n")
