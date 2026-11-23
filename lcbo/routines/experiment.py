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
        advanced_peak_detection: bool = True,
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
        advanced_peak_detection (bool): Whether to use advanced peak detection methods.
        fidelity_level (float): The fidelity level of the simulation (between 0 and 1).
        chromatogram_file (Optional[Path]): If provided, save the chromatogram plot to this file.

    """
    simulator.gradient = gradient
    compounds = compound_set.to_dicts()

    if chromatogram_file is not None or not advanced_peak_detection:
        chrom, peaks, compound_data = simulator.simulate_chromatogram(*compounds)
        peak_params = [(data["retention_time"], data["sigma"]) for data in compound_data]
    else:
        peak_params = [(simulator.calculate_peak(**compound))[:2] for compound in compounds]

    # Parse the simulation results and filter out invalid values
    retention_times, sigmas = np.asarray([(time.item(), sigma.item()) for time, sigma in peak_params]).reshape(-1, 2).T
    areas = np.asarray([compound["area"] for compound in compounds])
    valid_retention_times = np.isfinite(retention_times) & (retention_times > 0)
    valid_sigmas = np.isfinite(sigmas) & (sigmas > 0)
    valid_areas = np.isfinite(areas) & (areas > 0)
    valid = valid_retention_times & valid_sigmas & valid_areas
    valid_indices = np.flatnonzero(valid)

    # Calculate peak resolutions for valid compounds
    if advanced_peak_detection:
        peak_resolutions = np.full(len(compounds), np.nan)
        indices = valid_indices[np.argsort(retention_times[valid_indices])]
        if indices.size:
            sorted_sigmas = sigmas[indices]
            adjacent_diffs = np.diff(retention_times[indices]) / (2 * (sorted_sigmas[:-1] + sorted_sigmas[1:]))
            peak_resolutions[indices] = np.minimum(np.r_[np.inf, adjacent_diffs], np.r_[adjacent_diffs, np.inf])

        retention_times = np.where(valid, retention_times, np.nan)
        fwhms = np.where(valid, sigmas * (2 * np.sqrt(2 * np.log(2))), np.nan)
        areas = np.where(valid, areas, np.nan)

    else:
        peaks = DataAnalyzer.find_peaks(chrom, signal_to_noise=0.0, abs_threshold=0.0, min_fwhm=0.0)
        observed = np.full((len(compounds), 4), np.nan)
        available = set(valid_indices)

        for peak, resolution in zip(peaks, peak.peak_resolutions(None, fitted=False)):
            candidates = [i for i in available if peak.start_time <= retention_times[i] <= peak.end_time]
            idx = min(candidates, key=lambda i: abs(retention_times[i] - peak.peak_time), default=None)
            if idx is not None:
                observed[idx] = [peak.peak_time, peak.fwhm, peak.area, resolution]
                available.remove(idx)

        retention_times, fwhms, areas, peak_resolutions = observed.T

    # Plot the chromatogram if a file path is provided
    if chromatogram_file is not None:
        plot_chromatogram(chrom, chromatogram_file, peaks=peaks, gradient=gradient)

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
    if targets:
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
        else:
            raise ValueError("Targets must be a list of Compound or PeakTrack objects.")

        for peak, resolution in zip(target_peaks, target_resolutions):
            if peak is not None:
                retention_times.append(peak.peak_time_fitted)
                peak_resolutions.append(resolution)
                fwhms.append(peak.fwhm_fitted)
                peak_areas.append(peak.area_fitted)
            else:
                retention_times.append(np.nan)
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
