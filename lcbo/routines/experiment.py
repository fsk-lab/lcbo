from typing import Optional
import json
from pathlib import Path
import numpy as np

from moses.datatypes import LCGradient, Compound
from moses.controller import HPLCMSController
from moses.simulator import Simulator
from moses.analyzer import DataAnalyzer
from moses.analyzer.peak_tracking import PeakTrack, PeakTracker

from ..hplc_utils.hplc_experiment import HPLCExperiment
from ..hplc_utils.analyte import AnalyteSet
from ..data.config import HPLCExperimentConfig

from .plot_chromatogram import plot_chromatogram


def run_simulation_experiment(
        simulator: Simulator,
        compound_set: AnalyteSet,
        gradient: LCGradient,
        resolution_threshold,
        advanced_peak_detection: bool = False,
        fidelity_level: float = 1.0,
        chromatogram_file: Optional[Path] = None,
) -> HPLCExperiment:
    """
    Run a single HPLC simulation experiment using the provided simulator and compound set. Analyze the resulting
    chromatogram to extract relevant metrics and return an HPLCExperiment object.

    Args:
        simulator (Simulator): The simulator object to use for the experiment.
        compound_set (AnalyteSet): The set of compounds to simulate.
        gradient (LCGradient): The gradient to use for the simulation.
        resolution_threshold (float): The resolution threshold for calculating resolution fraction.
        advanced_peak_detection (bool): Whether to use advanced peak detection methods.
        fidelity_level (float): The fidelity level of the simulation (between 0 and 1).
        chromatogram_file (Optional[Path]): If provided, save the chromatogram plot to this file.

    """
    simulator.gradient = gradient
    chrom, peaks, cmpd_data = simulator.simulate_chromatogram(*compound_set.to_dicts())
    if chromatogram_file is not None:
        plot_chromatogram(chrom, chromatogram_file, peaks=peaks, gradient=gradient)

    # Calculate theoretical compound-wise properties (for compound-wise peak detection / peak tracking)
    retention_times = [d["retention_time"] for d in cmpd_data]
    sigmas = [d["sigma"] for d in cmpd_data]
    fwhms = [sigma * 2.3548 for sigma in sigmas]
    areas = [d["pre_exp_factor"] * d["sigma"] * np.sqrt(2 * np.pi) for d in cmpd_data]

    peak_resolutions = [np.nan for _ in range(len(retention_times))]
    valid_indices = [i for i in np.argsort(retention_times) if np.isfinite(retention_times[i]) and np.isfinite(sigmas[i])]
    for i, original_idx in enumerate(valid_indices):
        if i == 0:
            resolution_backward = np.inf
        else:
            previous_idx = valid_indices[i - 1]
            retention_time_diff = retention_times[original_idx] - retention_times[previous_idx]
            sigma_sum = sigmas[original_idx] + sigmas[previous_idx]
            resolution_backward = retention_time_diff / (2 * sigma_sum)

        if i == len(valid_indices) - 1:
            resolution_forward = np.inf
        else:
            next_idx = valid_indices[i + 1]
            retention_time_diff = retention_times[next_idx] - retention_times[original_idx]
            sigma_sum = sigmas[next_idx] + sigmas[original_idx]
            resolution_forward = retention_time_diff / (2 * sigma_sum)

        peak_resolutions[original_idx] = min(resolution_backward, resolution_forward)

    return HPLCExperiment(
        gradient=gradient,
        num_compounds=len(compound_set),
        retention_times=np.array(retention_times),
        peak_resolutions=np.array(peak_resolutions),
        fwhms=np.array(fwhms),
        peak_areas=np.array(areas),
        fidelity_level=fidelity_level,
    )


def run_experiment(
        instrument: HPLCMSController,
        gradient: LCGradient,
        sample_settings: dict,
        config: HPLCExperimentConfig,
        targets: Optional[list[Compound] | list[PeakTrack]] = None,
        data_dir: Optional[Path] = None,
        pre_flush: bool = True,
        post_flush: bool = False
) -> HPLCExperiment:
    """
    Run a single HPLC experiment using the provided instrument and gradient. Analyze the resulting
    chromatogram to extract relevant metrics and return an HPLCExperiment object.

    Args:
        instrument (HPLCMSController): The HPLC-MS instrument controller to use for the experiment.
        gradient (LCGradient): The gradient to use for the experiment.
        sample_settings (dict): Settings for the sample injection. Required keys: "folder", "sample_name",
                                "sample_position", "injection_volume".
        config (HPLCExperimentConfig): Configuration for data analysis and reporting.
        targets (Optional[list[Compound] | list[PeakTrack]]): Optional list of target compounds or peak tracks for
                                                              annotation.
        data_dir (Optional[Path]): If provided, save the chromatogram plot to this file.
        pre_flush (bool): Whether to perform a pre-flush injection before the main experiment.
        post_flush (bool): Whether to perform a post-flush injection after the main experiment.
    """
    if pre_flush:
        flush_method = LCGradient(
            name="Flush Gradient",
            eluents=gradient.eluents,
            flow_rate=gradient.flow_rate,
            time_steps=[0.0, 1.0],
            proportions_b=[0.95, gradient.gradient_table[gradient.eluent_b].tolist()[0]],
            equilibration_time=3.0,
        )
        instrument.set_acquisition_method(gradient=flush_method, ms_settings=config.ms_settings)
        instrument.start_run(
            sample_name="Pre_Flush",
            folder=sample_settings["folder"],
            sample_position=config.blank,
            injection_volume=1.0,
        )

    # Run the experiment on the instrument
    instrument.set_acquisition_method(gradient=gradient, ms_settings=config.ms_settings)
    data = instrument.start_run(**sample_settings)

    # Perform data analysis
    chrom = DataAnalyzer.extract_uv_chromatogram(lcms_data=data, **config.uv_extraction_settings)
    peaks = DataAnalyzer.find_peaks(chromatogram=chrom, lcms_data=data, **config.uv_peak_detection_settings)
    peaks = DataAnalyzer.fit_peaks(chrom, peaks, data, **config.uv_peak_fitting_settings)
    if targets is not None:
        if isinstance(targets[0], Compound):
            peaks = DataAnalyzer.annotate_peaks(data, peaks, targets, **config.ms_peak_annotation_settings)
        elif isinstance(targets[0], PeakTrack):
            peaks = PeakTracker.annotate_chromatograms_with_tracks(targets, data, peaks, **config.ms_peak_annotation_settings)

    # Save data and analysis results if a directory is provided
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        data.save(data_dir / f"{sample_settings['sample_name']}_data.npz")

        analysis_table = peaks.to_df(*config.reporting_properties)
        analysis_table.to_csv(data_dir / f"{sample_settings['sample_name']}_analysis.csv", index=False)

        plot_chromatogram(chrom, data_dir / f"{sample_settings['sample_name']}_chromatogram.png", peaks=peaks, gradient=gradient)


    # Calculate experiment metrics
    retention_times, peak_resolutions, fwhms, peak_areas = [], [], [], []

    if targets is not None:
        if isinstance(targets[0], Compound):
            target_peaks = [peaks.find_species(cmpd, k=10) for cmpd in targets]
            target_resolutions = peaks.peak_resolutions(targets, fitted=True)
        elif isinstance(targets[0], PeakTrack):
            target_peaks = [peaks.find_species(track.track_id, k=10) for track in targets]
            target_resolutions = peaks.peak_resolutions([track.track_id for track in targets], fitted=True)

        for peak, resolution in zip(target_peaks, target_resolutions):
            if peak is not None:
                retention_times.append(peak.peak_time_fitted)
                peak_resolutions.append(resolution)
                fwhms.append(peak.fwhm_fitted)
                peak_areas.append(peak.area_fitted)
            else:
                retention_times.append(np.inf)  # peak probably elutes after run time
                peak_resolutions.append(np.nan)
                fwhms.append(np.nan)
                peak_areas.append(np.nan)

    res = HPLCExperiment(
        gradient=gradient,
        num_compounds=len(targets) if targets is not None else 0,
        retention_times=np.array(retention_times),
        peak_resolutions=np.array(peak_resolutions),
        fwhms=np.array(fwhms),
        peak_areas=np.array(peak_areas),
        fidelity_level=1.0,
    )

    if data_dir is not None:
        res_dict = res.to_dict()
        with open(data_dir / f"{sample_settings['sample_name']}_experiment.json", "w") as f:
            json.dump(res_dict, f, indent=4)

    if post_flush:
        flush_method = LCGradient(
            name="Flush Gradient",
            eluents=gradient.eluents,
            flow_rate=gradient.flow_rate,
            time_steps=[0.0, 1.0],
            proportions_b=[1.00, 1.00],
            equilibration_time=3.0,
        )
        instrument.set_acquisition_method(gradient=flush_method, ms_settings=config.ms_settings)
        instrument.start_run(
            sample_name="Post_Flush",
            folder=sample_settings["folder"],
            sample_position=config.blank,
            injection_volume=1.0,
        )

    return res
