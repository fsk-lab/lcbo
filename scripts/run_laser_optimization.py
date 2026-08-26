import time
from pathlib import Path
from typing import Optional
import json
import numpy as np

from moses.datatypes import LCMSRecord, Compound, MSFragment
from moses.analyzer import DataAnalyzer, PeakTracker, PeakTrack
from lcbo.data.campaign import HPLCOptimizationCampaign
from lcbo.hplc_utils.hplc_experiment import HPLCExperiment
from lcbo.data.config import HPLCExperimentConfig, HPLCOptimizationConfig
from lcbo.routines import run_optimization_campaign
from lcbo.routines.plot_chromatogram import plot_chromatogram


def load_initial_experiments(
        campaign: HPLCOptimizationCampaign,
        experiment_config: HPLCExperimentConfig,
        sample_settings: dict,
        initial_experiments: list[Path]
) -> tuple[HPLCOptimizationCampaign, list[PeakTrack]]:
    """
    Loads initial experiments from the specified file paths and adds them to the campaign. Performs the following
    intialization steps:
        1. For each file, loads the data and performs initial analysis to identify peaks.
        2. Calculates peak tracks across the initial experiments.
        3. Annotates the initial experiments with peak tracks, and creates HPLCExperiment objects to add to the campaign
        4. Returns the updated campaign and the list of peak tracks.

    Args:
        campaign (HPLCOptimizationCampaign): The optimization campaign to which the experiments will be added.
        experiment_config (HPLCExperimentConfig): The configuration for analyzing the experiments.
        sample_settings (dict): The settings for the sample injections (used for naming and metadata).
        initial_experiments (list[Path]): A list of file paths pointing to the initial experiment data.

    Returns:
        HPLCOptimizationCampaign: The updated campaign with the loaded experiments.
        list[PeakTrack]: A list of PeakTrack objects fitted on the loaded experiments.
    """
    tracker = PeakTracker(
        weights={"retention": 0.2, "dad": 0.0, "ms": 0.8, "area": 0.0},
        min_edge_similarity=0.5,
        mutual_top_k=5,
        refinement_rounds=8
    )

    all_data, all_peaks, all_peaks_for_tracking = [], [], []
    for i, experiment_file in enumerate(initial_experiments):
        data = LCMSRecord.from_file(experiment_file)
        chrom = DataAnalyzer.extract_uv_chromatogram(lcms_data=data, **experiment_config.uv_extraction_settings)
        peaks = DataAnalyzer.find_peaks(chromatogram=chrom, lcms_data=data, **experiment_config.uv_peak_detection_settings)
        peaks = DataAnalyzer.fit_peaks(chrom, peaks, data, **experiment_config.uv_peak_fitting_settings)

        peaks_for_tracking = tracker.build_peaks_for_tracking(
            raw_data=data,
            peaks=peaks,
            use_peak_coherent_ms=True,
            resolution=1E-3,
            top_k_masses=50
        )
        all_data.append(data)
        all_peaks.append(peaks)
        all_peaks_for_tracking.append(peaks_for_tracking)

        data_dir = campaign.config.working_dir / f"Experiment_{i + 1}"
        data_dir.mkdir(parents=True, exist_ok=True)
        data.save(data_dir / f"{sample_settings['sample_name']}_data.npz")
        analysis_table = peaks.to_df(*experiment_config.reporting_properties)
        analysis_table.to_csv(data_dir / f"{sample_settings['sample_name']}_analysis.csv", index=False)
        plot_chromatogram(chrom, data_dir / f"{sample_settings['sample_name']}_chromatogram.png", peaks=peaks, gradient=data.metadata.gradient)

    peak_tracks = tracker.track_peaks(all_peaks_for_tracking)
    campaign.config.num_compounds = len(peak_tracks)

    for i, (data, peaks) in enumerate(zip(all_data, all_peaks)):
        retention_times, peak_resolutions, fwhms, peak_areas = [], [], [], []
        peaks = tracker.annotate_chromatograms_with_tracks(peak_tracks, data, peaks, **experiment_config.ms_peak_annotation_settings)
        track_peaks = [peaks.find_species(track.track_id, k=10) for track in peak_tracks]
        track_resolutions = peaks.peak_resolutions([track.track_id for track in peak_tracks], fitted=True)

        for peak, resolution in zip(track_peaks, track_resolutions):
            if peak is not None:
                retention_times.append(peak.peak_time_fitted)
                peak_resolutions.append(resolution)
                fwhms.append(peak.fwhm_fitted)
                peak_areas.append(peak.area_fitted)
            else:
                retention_times.append(np.inf)
                peak_resolutions.append(np.nan)
                fwhms.append(np.nan)
                peak_areas.append(np.nan)

        experiment = HPLCExperiment(
            gradient=data.metadata.gradient,
            num_compounds=len(track_peaks),
            retention_times=np.array(retention_times),
            peak_resolutions=np.array(peak_resolutions),
            fwhms=np.array(fwhms),
            peak_areas=np.array(peak_areas),
            fidelity_level=1.0,
        )
        campaign.append(experiment)

    return campaign, peak_tracks


def find_target_track(
        peak_tracks: list[PeakTrack],
        target_compound: Compound,
        ionization_mode: Optional[MSFragment] = None
) -> int:
    """
    Identifies the index of the peak track corresponding to the target compound based on mass spectral similarity.

    Args:
        peak_tracks (list[PeakTrack]): The list of peak tracks obtained from the initial experiments.
        target_compound (Compound): The target compound for which the track index needs to be identified.
        ionization_mode (MSFragment): The ionization mode used for the experiments.

    Returns:
        int: The index of the peak track corresponding to the target compound. Returns -1 if no suitable track is found.
    """
    if ionization_mode is None:
        ionization_mode = MSFragment("H", 1)  # Default to [+H]+ ionization mode
    target_ion = target_compound + ionization_mode

    best_idx, best_similarity = None, -np.inf
    for track_idx, track in enumerate(peak_tracks):
        polarity = "positive scan" if target_ion.charge > 0 else "negative scan"
        prototype_ms = track.peak_prototype.ms.get_mass_spectrum(polarity=polarity)
        similarity = target_ion.ms_similarity(prototype_ms)
        if similarity > best_similarity:
            best_similarity = similarity
            best_idx = track_idx

    if best_idx is None:
        raise ValueError("No suitable track found for the target compound.")

    return best_idx


if __name__ == "__main__":

    config_dir = Path.cwd() / "configs"
    data_dir = Path.cwd() / "results" / "laser_optimization"

    time_stamp = time.strftime("%Y-%m-%d_%H-%M")
    campaign_name = f"Optimization_Campaign_{time_stamp}"

    sample_settings = {
        "sample_name": "Laser_Crude",
        "folder": f"Felix/LCBO/{campaign_name}",
        "sample_position": "R:D9",
        "injection_volume": 2.0,
        "sample_solvent": "Acetonitrile"
    }

    target_compound = Compound.from_smiles(name="Target", smiles="CC1=CC=C(C)C(C2=CC3=C(C=CC(C4=CC(N(C5=CC=CC=C5)C6=C7C=CC=C6)=C7C=C4)=C3)O2)=C1")

    with open(config_dir / "eluent_optimization_config_single_compound.json", "r") as f:
        optimization_config = HPLCOptimizationConfig.from_dict(json.load(f))
    optimization_config.working_dir = data_dir / campaign_name

    with open(optimization_config.experiment_config, "r") as f:
        experiment_config = HPLCExperimentConfig.from_dict(json.load(f))
    experiment_config.working_dir = optimization_config.working_dir

    instrument = experiment_config.get_instrument()

    with open(data_dir / "initial_experiments.json", "r") as f:
        initial_experiments = json.load(f)
        initial_experiments = [Path(exp) for exp in initial_experiments]

    try:
        campaign = HPLCOptimizationCampaign(config=optimization_config)

        campaign, peak_tracks = load_initial_experiments(
            campaign=campaign,
            experiment_config=experiment_config,
            sample_settings=sample_settings,
            initial_experiments=initial_experiments
        )

        target_track_idx = find_target_track(peak_tracks, target_compound)
        optimization_config.compound_idx = target_track_idx

        for i, experiment in enumerate(campaign.experiments):
            experiment.score = campaign.config.calculate_experiment_score(experiment)
            data_dir = campaign.config.working_dir / f"Experiment_{i + 1}"
            with open(data_dir / f"{sample_settings['sample_name']}_experiment.json", "w") as f:
                json.dump(experiment.to_dict(), f, indent=4)
        campaign.to_pkl(campaign.config.working_dir / "campaign.pkl")

        campaign = run_optimization_campaign(
            simulation=False,
            instrument=instrument,
            optimization_config=optimization_config,
            analytes=peak_tracks,
            data_dir=optimization_config.working_dir,
            sample_settings=sample_settings,
            experiment_config=experiment_config
        )

        with open(optimization_config.working_dir / "campaign_summary.json", "w") as f:
            json.dump(campaign.to_dict(), f, indent=4)

    finally:
        instrument.standby()