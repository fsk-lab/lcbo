import logging
import hashlib
import math
from typing import Literal, Optional, get_args
from pathlib import Path
from argparse import ArgumentParser
import json
import warnings
import numpy as np

from scipy.optimize import LinearConstraint
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionObjective
from botorch.exceptions.warnings import BotorchWarning, OptimizationWarning
from linear_operator.utils.warnings import NumericalWarning

from moses.datatypes import MSFragment
from moses.controller import HPLCMSController

from ..hplc_utils.multilinear_gradient import LCGradientConstraints
from ..hplc_utils.optimization_objectives import get_hplc_objective
from ..hplc_utils.hplc_experiment import HPLCExperiment


InstrumentVendor = Literal["ThermoFisher", "Agilent"]
BaselineAlgorithm = Literal["rolling_ball", "snip"]
PeakDeconvolutionModel = Literal["Gaussian", "SkewedGaussian", "ExponNorm"]
ReportingProperty = Literal[
    "Peak Time", "Peak Time (fitted)", "Start Time", "End Time", "Area", "Area (fitted)", "Height", "Height (fitted)",
    "FWHM", "FWHM (fitted)", "Asymmetry Factor", "Asymmetry Factor (fitted)", "Tailing Factor", "Tailing Factor (fitted)",
    "UV Spectrum", "Mass Spectrum", "Peak Purity (UV)", "Peak Purity (MS+)", "Peak Purity (MS-)", "Annotation",
    "Resolution", "Resolution (fitted)"
]
LogLevel = Literal["NOTSET", "DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"]

OptimizationStrategy = Literal["bayesian", "retention_model"]
RetentionModel = Literal["linear", "quadratic", "neue-kuss"]
FidelityStrategy = Literal["prediction_error", "experiment_count"]
SurrogateModel = Literal["GP", "NN", "DNN"]
AcquisitionFunction = Literal["EI", "LogEI", "UCB", "UCBExp", "EHVI", "STGap", "random"]
SeedStrategy = Literal["random", "isocratic"]
OptimizationObjective = Literal["NumPeaks-CritRes", "NumPeaks-NumResPeaks", "SinglePeak-Resolution"]
OptimizationModelType = Literal["cumulative", "compoundwise"]


class HPLCExperimentConfig(object):
    """
    Configuration class that contains all parameters for running and analyzing real-life HPLC experiments.
    """

    __instrument_vendor: InstrumentVendor = "ThermoFisher"
    __eluents: tuple[str, str] = ("Eluent A", "Eluent B")
    __blank: Optional[str] = None
    __equilibration_time: float = 1.0  # minutes

    __uv_wavelength_range : tuple[float, float] = (250., 300.)
    __uv_baseline_correction: bool = True
    __uv_baseline_algorithm: BaselineAlgorithm = "rolling_ball"
    __uv_min_peak_fwhm: float = 0.01  # minutes

    __uv_peak_detection_threshold: float = 10.0

    __uv_peak_deconvolution: bool = True
    __uv_peak_deconvolution_model: PeakDeconvolutionModel = "SkewedGaussian"

    __ms_mode: str = "positive_negative"
    __ms_ionization_modes: list[dict] = [{"formula": "H", "charge": 1, "addition": True}]
    __ms_peak_detection_threshold: float = 1E5
    __ms_rel_tolerance: float = 10.0

    __reporting_properties: list[ReportingProperty] = [
        "Peak Time", "Start Time", "End Time", "Area", "Asymmetry Factor", "Peak Purity (UV)"
    ]

    __scoring_resolution_threshold: float = 1.5

    __log_level: LogLevel = "INFO"

    def __init__(
            self,
            instrument_vendor: Optional[InstrumentVendor] = None,
            instrument_config_file: Optional[Path | str] = None,
            eluents: Optional[tuple[str, str]] = None,
            blank: Optional[str] = None,
            equilibration_time: Optional[float] = None,
            uv_wavelength_range: Optional[tuple[float, float]] = None,
            uv_baseline_correction: Optional[bool] = None,
            uv_baseline_algorithm: Optional[BaselineAlgorithm] = None,
            uv_peak_detection_threshold: Optional[float] = None,
            uv_min_peak_fwhm: Optional[float] = None,
            uv_peak_deconvolution: Optional[bool] = None,
            uv_peak_deconvolution_model: Optional[PeakDeconvolutionModel] = None,
            ms_mode: Optional[str] = None,
            ms_ionization_modes: Optional[list[dict]] = None,
            ms_peak_detection_threshold: Optional[float] = None,
            ms_rel_tolerance: Optional[float] = None,
            reporting_properties: Optional[list[ReportingProperty]] = None,
            scoring_resolution_threshold: Optional[float] = None,
            working_dir: Optional[Path] = None,
            log_level: Optional[LogLevel] = None,
    ):
        self.instrument_vendor = instrument_vendor if instrument_vendor is not None else self.__instrument_vendor
        self.instrument_config_file = Path(instrument_config_file) if instrument_config_file else None
        self.eluents = eluents if eluents is not None else self.__eluents
        self.blank = blank if blank is not None else self.__blank
        self.equilibration_time = equilibration_time if equilibration_time is not None else self.__equilibration_time
        self.uv_wavelength_range = uv_wavelength_range if uv_wavelength_range is not None else self.__uv_wavelength_range
        self.uv_baseline_correction = uv_baseline_correction if uv_baseline_correction is not None else self.__uv_baseline_correction
        self.uv_baseline_algorithm = uv_baseline_algorithm if uv_baseline_algorithm is not None else self.__uv_baseline_algorithm
        self.uv_peak_detection_threshold = uv_peak_detection_threshold if uv_peak_detection_threshold is not None else self.__uv_peak_detection_threshold
        self.uv_min_peak_fwhm = uv_min_peak_fwhm if uv_min_peak_fwhm is not None else self.__uv_min_peak_fwhm
        self.uv_peak_deconvolution = uv_peak_deconvolution if uv_peak_deconvolution is not None else self.__uv_peak_deconvolution
        self.uv_peak_deconvolution_model = uv_peak_deconvolution_model if uv_peak_deconvolution_model is not None else self.__uv_peak_deconvolution_model
        self.ms_mode = ms_mode if ms_mode is not None else self.__ms_mode
        self.ms_ionization_modes = [MSFragment(**mode) for mode in ms_ionization_modes] if ms_ionization_modes else [MSFragment(**mode) for mode in self.__ms_ionization_modes]
        self.ms_peak_detection_threshold = ms_peak_detection_threshold if ms_peak_detection_threshold is not None else self.__ms_peak_detection_threshold
        self.ms_rel_tolerance = ms_rel_tolerance if ms_rel_tolerance is not None else self.__ms_rel_tolerance
        self.reporting_properties = reporting_properties if reporting_properties is not None else self.__reporting_properties
        self.scoring_resolution_threshold = scoring_resolution_threshold if scoring_resolution_threshold is not None else self.__scoring_resolution_threshold

        self.log_level = log_level.upper() if log_level else self.__log_level
        self._working_dir = working_dir
        self.logger = logging.getLogger("moses")
        self.logger.propagate = False
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        if self._working_dir is not None:
            self._working_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self._working_dir / f"optimization_progress.log")
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.setLevel(self.log_level)

    @property
    def working_dir(self) -> Path:
        return self._working_dir

    @working_dir.setter
    def working_dir(self, path: Path):
        self._working_dir = path
        self._working_dir.mkdir(parents=True, exist_ok=True)

        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
                handler.close()
        # add new file handler
        file_handler = logging.FileHandler(self._working_dir / f"optimization_progress.log")
        file_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_instrument(self) -> HPLCMSController:
        """
        Loads the instrument configuration from the specified JSON file, and initializes the appropriate HPLC-MS
        controller based on the instrument vendor.
        """
        if self.instrument_config_file is None:
            raise ValueError("No instrument configuration file provided.")

        with open(self.instrument_config_file, "r") as f:
            config = json.load(f)

        if self.instrument_vendor == "ThermoFisher":
            from moses.chromeleon import ThermoHPLCMSController
            controller = ThermoHPLCMSController(config)
        elif self.instrument_vendor == "Agilent":
            from moses.agilent import AgilentController
            controller = AgilentController(config)
        else:
            raise ValueError(f"Unsupported instrument vendor: {self.instrument_vendor}")

        return controller

    @property
    def uv_extraction_settings(self) -> dict:
        return {
            "wavelength_range": self.uv_wavelength_range,
            "baseline_correction": self.uv_baseline_correction,
            "baseline_algorithm": self.uv_baseline_algorithm,
        }

    @property
    def uv_peak_detection_settings(self) -> dict:
        return {
            "abs_threshold": self.uv_peak_detection_threshold,
            "min_fwhm": self.uv_min_peak_fwhm,
        }

    @property
    def uv_peak_fitting_settings(self) -> dict:
        return {
            "deconvolution": self.uv_peak_deconvolution,
            "model": self.uv_peak_deconvolution_model,
        }

    @property
    def ms_settings(self) -> dict:
        return {
            "mode": self.ms_mode,
        }

    @property
    def ms_peak_annotation_settings(self) -> dict:
        return {
            "ionizations": self.ms_ionization_modes,
            "rel_mass_tolerance": self.ms_rel_tolerance,
            "eic_abs_threshold": self.ms_peak_detection_threshold,
        }

    def to_dict(self) -> dict:
        return {
            "instrument_vendor": self.instrument_vendor,
            "instrument_config_file": str(self.instrument_config_file) if self.instrument_config_file else None,
            "eluents": list(self.eluents),
            "blank": self.blank,
            "equilibration_time": self.equilibration_time,
            "uv_wavelength_range": self.uv_wavelength_range,
            "uv_baseline_correction": self.uv_baseline_correction,
            "uv_baseline_algorithm": self.uv_baseline_algorithm,
            "uv_peak_detection_threshold": self.uv_peak_detection_threshold,
            "uv_min_peak_fwhm": self.uv_min_peak_fwhm,
            "uv_peak_deconvolution": self.uv_peak_deconvolution,
            "uv_peak_deconvolution_model": self.uv_peak_deconvolution_model,
            "ms_mode": self.ms_mode,
            "ms_ionization_modes": [{"formula": mode.compound.formula, "charge": mode.charge, "addition": mode.add} for mode in self.ms_ionization_modes],
            "ms_peak_detection_threshold": self.ms_peak_detection_threshold,
            "ms_rel_tolerance": self.ms_rel_tolerance,
            "reporting_properties": self.reporting_properties,
            "scoring_resolution_threshold": self.scoring_resolution_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HPLCExperimentConfig":
        return cls(
            instrument_vendor=data.get("instrument_vendor"),
            instrument_config_file=Path(data["instrument_config_file"]) if data.get("instrument_config_file") else None,
            eluents=tuple(data["eluents"]) if data.get("eluents") else None,
            blank=data.get("blank"),
            equilibration_time=data.get("equilibration_time"),
            uv_wavelength_range=tuple(data["uv_wavelength_range"]) if data.get("uv_wavelength_range") else None,
            uv_baseline_correction=data.get("uv_baseline_correction"),
            uv_baseline_algorithm=data.get("uv_baseline_algorithm"),
            uv_peak_detection_threshold=data.get("uv_peak_detection_threshold"),
            uv_min_peak_fwhm=data.get("uv_min_peak_fwhm"),
            uv_peak_deconvolution=data.get("uv_peak_deconvolution"),
            uv_peak_deconvolution_model=data.get("uv_peak_deconvolution_model"),
            ms_mode=data.get("ms_mode"),
            ms_ionization_modes=data.get("ms_ionization_modes"),
            ms_peak_detection_threshold=data.get("ms_peak_detection_threshold"),
            ms_rel_tolerance=data.get("ms_rel_tolerance"),
            reporting_properties=data.get("reporting_properties"),
            scoring_resolution_threshold=data.get("scoring_resolution_threshold"),
        )


class HPLCOptimizationConfig(object):
    """
    Configuration class that contains all parameters for a data-driven optimization of HPLC runs.
    """

    __num_compounds: int = 10

    __num_segments: int = 3
    __min_segment_length: float = 0.5  # minutes
    __segment_based: bool = False  # if True, optimize segment lengths instead of time points
    __max_run_time: float = 20.0  # minutes
    __flow_rate: float = 0.4  # mL/min, fixed for now
    __column_length: float = 0.1  # m
    __column_inner_diameter: float = 0.0021  # m

    __experiment_config: Path = Path(__file__).resolve().parents[2] / "scripts" / "configs" / "simulator_config.json"
    __advanced_peak_detection: bool = True

    __optimization_strategy: OptimizationStrategy = "bayesian"
    __retention_model: RetentionModel = "linear"
    __simulation_interval: float = 0.01
    __fidelity_strategy: FidelityStrategy = "prediction_error"
    __retention_error_scale: float = 0.1
    __retention_miss_probability: float = 0.05
    __retention_observation_sigma: float = 0.25  # minutes

    __surrogate_model: SurrogateModel = "GP"
    __acquisition_function: AcquisitionFunction = "LogEI"
    __budget: int = 100
    __multi_fidelity: bool = False
    __max_simulation_fidelity: float = 0.5
    __simulation_fidelity_decay: float = 0.1
    __low_fidelity_cost: float = 0.25
    __max_consecutive_low_fidelity: int = 10

    __seed_budget: int = 3
    __seed_strategy: SeedStrategy = "random"

    __objective: OptimizationObjective = "NumPeaks-CritRes"
    __model_type: OptimizationModelType = "cumulative"
    __count_threshold: float = 0.5
    __resolution_threshold: float = 1.5
    __run_time_threshold: float = 5.0  # minutes

    __parent_dir: Path = Path.cwd() / "results"
    __log_level: LogLevel = "INFO"

    def __init__(
            self,
            num_compounds: Optional[int] = None,
            compound_idx: Optional[int] = None,
            num_segments: Optional[int] = None,
            min_segment_length: Optional[float] = None,
            segment_based: Optional[bool] = None,
            max_run_time: Optional[float] = None,
            flow_rate: Optional[float] = None,
            column_length: Optional[float] = None,
            column_inner_diameter: Optional[float] = None,
            experiment_config: Optional[Path] = None,
            advanced_peak_detection: Optional[bool] = None,
            optimization_strategy: Optional[OptimizationStrategy] = None,
            retention_model: Optional[RetentionModel] = None,
            simulation_interval: Optional[float] = None,
            fidelity_strategy: Optional[FidelityStrategy] = None,
            retention_error_scale: Optional[float] = None,
            surrogate_model: Optional[SurrogateModel] = None,
            acquisition_function: Optional[AcquisitionFunction] = None,
            budget: Optional[int] = None,
            seed_budget: Optional[int] = None,
            seed_strategy: Optional[SeedStrategy] = None,
            multi_fidelity: Optional[bool] = None,
            max_simulation_fidelity: Optional[float] = None,
            simulation_fidelity_decay: Optional[float] = None,
            low_fidelity_cost: Optional[float] = None,
            max_consecutive_low_fidelity: Optional[int] = None,
            objective: Optional[OptimizationObjective] = None,
            model_type: Optional[OptimizationModelType] = None,
            count_threshold: Optional[float] = None,
            resolution_threshold: Optional[float] = None,
            run_time_threshold: Optional[float] = None,
            working_dir: Optional[Path] = None,
            log_level: Optional[LogLevel] = None,
            retention_miss_probability: Optional[float] = None,
            retention_observation_sigma: Optional[float] = None,
    ):

        self.initialize_torch()

        if objective == "SinglePeak-Resolution" and compound_idx is None:
            raise ValueError("compound_idx must be specified when objective is 'SinglePeak-Resolution'.")
        if compound_idx is not None and objective != "SinglePeak-Resolution":
            raise ValueError("compound_idx should only be specified when objective is 'SinglePeak-Resolution'.")

        self.num_compounds = num_compounds if num_compounds is not None else self.__num_compounds
        self.compound_idx = compound_idx

        self.num_segments = num_segments if num_segments is not None else self.__num_segments
        self.min_segment_length = min_segment_length if min_segment_length is not None else self.__min_segment_length
        self.segment_based = segment_based if segment_based is not None else self.__segment_based
        self.max_run_time = max_run_time if max_run_time is not None else self.__max_run_time
        self.flow_rate = flow_rate if flow_rate is not None else self.__flow_rate
        self.column_length = column_length if column_length is not None else self.__column_length
        self.column_inner_diameter = column_inner_diameter if column_inner_diameter is not None else self.__column_inner_diameter
        if self.column_length <= 0 or self.column_inner_diameter <= 0:
            raise ValueError("Column length and inner diameter must be positive (in meters).")
        self.constraint_generator = LCGradientConstraints(
            num_segments=self.num_segments,
            min_segment_length=self.min_segment_length,
            max_run_time=self.max_run_time,
            segment_based=self.segment_based
        )

        self.experiment_config = experiment_config if experiment_config is not None else self.__experiment_config
        self.advanced_peak_detection = advanced_peak_detection if advanced_peak_detection is not None else self.__advanced_peak_detection

        self.optimization_strategy = optimization_strategy if optimization_strategy is not None else self.__optimization_strategy
        self.retention_model = retention_model if retention_model is not None else self.__retention_model
        self.simulation_interval = simulation_interval if simulation_interval is not None else self.__simulation_interval
        self.fidelity_strategy = fidelity_strategy if fidelity_strategy is not None else self.__fidelity_strategy
        self.retention_error_scale = retention_error_scale if retention_error_scale is not None else self.__retention_error_scale
        self.retention_miss_probability = retention_miss_probability if retention_miss_probability is not None else self.__retention_miss_probability
        self.retention_observation_sigma = retention_observation_sigma if retention_observation_sigma is not None else self.__retention_observation_sigma
        if not 0 <= self.retention_miss_probability < 1:
            raise ValueError("Retention miss probability must lie in [0, 1).")
        if not math.isfinite(self.retention_observation_sigma) or self.retention_observation_sigma <= 0:
            raise ValueError("Retention observation sigma must be positive and finite (in minutes).")

        self.surrogate_model = surrogate_model if surrogate_model is not None else self.__surrogate_model
        self.acquisition_function = acquisition_function if acquisition_function is not None else self.__acquisition_function

        self.budget = budget if budget is not None else self.__budget
        self.seed_budget = seed_budget if seed_budget is not None else self.__seed_budget
        self.seed_strategy = seed_strategy if seed_strategy is not None else self.__seed_strategy

        self.multi_fidelity = multi_fidelity if multi_fidelity is not None else self.__multi_fidelity
        self.max_simulation_fidelity = max_simulation_fidelity if max_simulation_fidelity is not None else self.__max_simulation_fidelity
        self.simulation_fidelity_decay = simulation_fidelity_decay if simulation_fidelity_decay is not None else self.__simulation_fidelity_decay
        self.low_fidelity_cost = low_fidelity_cost if low_fidelity_cost is not None else self.__low_fidelity_cost
        self.max_consecutive_low_fidelity = max_consecutive_low_fidelity if max_consecutive_low_fidelity is not None else self.__max_consecutive_low_fidelity

        self.objective = objective if objective is not None else self.__objective
        self.model_type = model_type if model_type is not None else self.__model_type
        self.count_threshold = count_threshold if count_threshold is not None else self.__count_threshold
        self.resolution_threshold = resolution_threshold if resolution_threshold is not None else self.__resolution_threshold
        self.run_time_threshold = run_time_threshold if run_time_threshold is not None else self.__run_time_threshold

        if self.optimization_strategy not in ("bayesian", "retention_model"):
            raise ValueError("optimization_strategy must be 'bayesian' or 'retention_model'.")
        if self.surrogate_model not in ("GP", "NN", "DNN") or (self.multi_fidelity and self.surrogate_model != "GP"):
            raise ValueError("Use GP, NN or DNN; multi-fidelity optimization requires GP.")
        acquisition_name = self.acquisition_function.lower()
        if acquisition_name not in {name.lower() for name in get_args(AcquisitionFunction)}:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}.")
        if self.multi_fidelity and acquisition_name == "ehvi":
            raise ValueError("Multi-fidelity EHVI is not implemented.")
        if self.retention_model not in ("linear", "quadratic", "neue-kuss"):
            raise ValueError("Unknown retention model.")
        if self.optimization_strategy == "retention_model" and self.multi_fidelity:
            raise ValueError("The retention-model comparator uses measured experiments only; disable multi_fidelity.")
        if self.fidelity_strategy not in ("prediction_error", "experiment_count"):
            raise ValueError("Unknown fidelity strategy.")
        if not 0 <= self.max_simulation_fidelity < 0.99 or not 0 < self.low_fidelity_cost <= 1:
            raise ValueError("Simulation fidelity must be below 0.99 and relative simulation cost must be in (0, 1].")
        if min(self.num_compounds, self.budget, self.seed_budget) < 1 or self.seed_budget > self.budget:
            raise ValueError("Counts must be positive and seed_budget cannot exceed budget.")
        if self.compound_idx is not None and not 0 <= self.compound_idx < self.num_compounds:
            raise ValueError("compound_idx must identify a compound in the mixture.")
        if not all(math.isfinite(value) and value > 0 for value in (self.column_length, self.column_inner_diameter, self.flow_rate, self.simulation_interval, self.retention_error_scale)):
            raise ValueError("Dimensions, flow rate, simulation interval and error scale must be positive.")
        count_upper = self.num_compounds if self.compound_idx is None else 1
        if not 0 <= self.count_threshold < count_upper or not 0 < self.resolution_threshold <= 5 or not 0 < self.run_time_threshold < self.max_run_time:
            raise ValueError("Objective thresholds must lie inside their normalization ranges.")
        if self.max_consecutive_low_fidelity < 0 or self.simulation_fidelity_decay < 0:
            raise ValueError("Low-fidelity limit and fidelity decay must be non-negative.")

        self.log_level = log_level.upper() if log_level else self.__log_level

        # Set up the data directory
        self._working_dir = working_dir
        if working_dir is not None:
            self._working_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger("hplc-bo")
        self.logger.propagate = False
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        if not any(type(handler) is logging.StreamHandler for handler in self.logger.handlers):
            self.logger.addHandler(console_handler)
        if self._working_dir is not None:
            file_handler = logging.FileHandler(self._working_dir / f"optimization_progress.log")
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.setLevel(self.log_level)

    @staticmethod
    def initialize_torch():
        torch.set_default_dtype(torch.float64)
        warnings.filterwarnings("ignore", category=BotorchWarning)
        warnings.filterwarnings("ignore", category=OptimizationWarning)
        warnings.filterwarnings("ignore", category=NumericalWarning)

    @property
    def tag(self) -> str:
        """
        Creates a unique tag for the current configuration based on its parameters.

        Returns:
            str: The unique tag string.
        """
        settings = self.to_dict()
        for name in ("working_dir", "log_level"):
            settings.pop(name)
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        return f"Optimization_{self.optimization_strategy}_{self.surrogate_model}-{self.acquisition_function}_{self.objective}_{'MF' if self.multi_fidelity else 'SF'}_{digest}"

    @property
    def working_dir(self) -> Path:
        return self._working_dir

    @working_dir.setter
    def working_dir(self, path: Path):
        self._working_dir = path
        self._working_dir.mkdir(parents=True, exist_ok=True)

        for handler in list(self.logger.handlers):
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
                handler.close()
        # add new file handler
        file_handler = logging.FileHandler(self._working_dir / f"optimization_progress.log")
        file_handler.setLevel(self.log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    @property
    def num_optimization_variables(self) -> int:
        return 2 * self.num_segments + 1

    @property
    def bounds(self) -> torch.Tensor:
        return self.constraint_generator.bounds

    @property
    def bounds_scipy(self) -> np.ndarray:
        return self.constraint_generator.bounds_scipy

    @property
    def constraints(self) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        return self.constraint_generator.constraints

    @property
    def constraints_scipy(self) -> LinearConstraint:
        return self.constraint_generator.constraints_scipy

    @property
    def optimization_objective(self) -> MCAcquisitionObjective:
        """
        Returns the appropriate multi-objective acquisition function based on the specified objective type and
        thresholds. Uses BoTier for hierarchical multi-objective optimization between detection fraction, resolution
        objective, and run time.

        Returns:
            MCAcquisitionObjective: The configured acquisition objective.
        """
        return get_hplc_objective(
            hierarchical=self.acquisition_function.lower() != "ehvi",
            model_type=self.model_type,
            num_compounds=self.num_compounds,
            compound_idx=self.compound_idx,
            count_threshold=self.count_threshold,
            resolution_objective="count" if "NumResPeaks" in self.objective else "critical",
            resolution_bounds=(0.0, 5.0),
            resolution_threshold=self.resolution_threshold,
            run_time_bounds=(0.0, self.max_run_time),
            run_time_threshold=self.run_time_threshold,
            segment_based=self.segment_based
        )

    @property
    def single_fidelity(self) -> bool:
        """
        Indicates whether the optimization configuration is set up for single-fidelity optimization.

        Returns:
            bool: True if single-fidelity optimization is enabled, False otherwise.
        """
        return not self.multi_fidelity

    @property
    def needs_retention_model(self) -> bool:
        """
        Indicates whether the optimization requires a retention model for optimization.

        Returns:
            bool: True if a retention model is needed, False otherwise.
        """
        return self.optimization_strategy == "retention_model" or self.multi_fidelity

    def calculate_experiment_score(self, experiment: HPLCExperiment) -> float:
        """
        Calculates the score of a given HPLC experiment based on the objective thresholds.

        Args:
            experiment (HPLCExperiment): The HPLC experiment to score.

        Returns:
            float: The calculated score.
        """
        x = torch.tensor(experiment.gradient_parameters.parameters.reshape(1, -1))
        if self.objective == "NumPeaks-CritRes":
            y = torch.tensor([[experiment.num_peaks, experiment.critical_resolution]])
        elif self.objective == "NumPeaks-NumResPeaks":
            y = torch.tensor([[experiment.num_peaks, experiment.num_resolved_peaks(self.resolution_threshold)]])
        elif self.objective == "SinglePeak-Resolution":
            y = torch.tensor([experiment.single_peak_elution_resolution(self.compound_idx)])
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        objective = get_hplc_objective(
            hierarchical=True,
            model_type="cumulative",
            num_compounds=self.num_compounds if self.compound_idx is None else 1,
            compound_idx=self.compound_idx,
            count_threshold=self.count_threshold,
            resolution_objective="count" if "NumResPeaks" in self.objective else "critical",
            resolution_bounds=(0.0, 5.0),
            resolution_threshold=self.resolution_threshold,
            run_time_bounds=(0.0, self.max_run_time),
            run_time_threshold=self.run_time_threshold,
            segment_based=self.segment_based
        )

        return objective(y, x).item()

    def to_dict(self) -> dict:
        return {
            "num_compounds": self.num_compounds,
            "compound_idx": self.compound_idx,
            "num_segments": self.num_segments,
            "min_segment_length": self.min_segment_length,
            "segment_based": self.segment_based,
            "max_run_time": self.max_run_time,
            "flow_rate": self.flow_rate,
            "column_length": self.column_length,
            "column_inner_diameter": self.column_inner_diameter,
            "experiment_config": str(self.experiment_config),
            "advanced_peak_detection": self.advanced_peak_detection,
            "optimization_strategy": self.optimization_strategy,
            "retention_model": self.retention_model,
            "simulation_interval": self.simulation_interval,
            "fidelity_strategy": self.fidelity_strategy,
            "retention_error_scale": self.retention_error_scale,
            "retention_miss_probability": self.retention_miss_probability,
            "retention_observation_sigma": self.retention_observation_sigma,
            "surrogate_model": self.surrogate_model,
            "acquisition_function": self.acquisition_function,
            "budget": self.budget,
            "seed_budget": self.seed_budget,
            "seed_strategy": self.seed_strategy,
            "multi_fidelity": self.multi_fidelity,
            "max_simulation_fidelity": self.max_simulation_fidelity,
            "simulation_fidelity_decay": self.simulation_fidelity_decay,
            "low_fidelity_cost": self.low_fidelity_cost,
            "max_consecutive_low_fidelity": self.max_consecutive_low_fidelity,
            "objective": self.objective,
            "model_type": self.model_type,
            "count_threshold": self.count_threshold,
            "resolution_threshold": self.resolution_threshold,
            "run_time_threshold": self.run_time_threshold,
            "working_dir": str(self.working_dir) if self.working_dir is not None else None,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HPLCOptimizationConfig":

        # make sure that transfer between systems is possible
        working_dir = Path(data["working_dir"]) if data.get("working_dir") else None
        if working_dir:
            if not (working_dir.exists() or working_dir.parent.exists()):
                working_dir = None

        return cls(
            num_compounds=data.get("num_compounds"),
            compound_idx=data.get("compound_idx"),
            num_segments=data.get("num_segments"),
            min_segment_length=data.get("min_segment_length"),
            segment_based=data.get("segment_based"),
            max_run_time=data.get("max_run_time"),
            flow_rate=data.get("flow_rate"),
            column_length=data.get("column_length"),
            column_inner_diameter=data.get("column_inner_diameter"),
            experiment_config=Path(data["experiment_config"]) if data.get("experiment_config") else None,
            advanced_peak_detection=data.get("advanced_peak_detection"),
            optimization_strategy=data.get("optimization_strategy"),
            retention_model=data.get("retention_model"),
            simulation_interval=data.get("simulation_interval"),
            fidelity_strategy=data.get("fidelity_strategy"),
            retention_error_scale=data.get("retention_error_scale"),
            retention_miss_probability=data.get("retention_miss_probability"),
            retention_observation_sigma=data.get("retention_observation_sigma"),
            surrogate_model=data.get("surrogate_model"),
            acquisition_function=data.get("acquisition_function"),
            budget=data.get("budget"),
            seed_budget=data.get("seed_budget"),
            seed_strategy=data.get("seed_strategy"),
            multi_fidelity=data.get("multi_fidelity"),
            max_simulation_fidelity=data.get("max_simulation_fidelity"),
            simulation_fidelity_decay=data.get("simulation_fidelity_decay"),
            low_fidelity_cost=data.get("low_fidelity_cost"),
            max_consecutive_low_fidelity=data.get("max_consecutive_low_fidelity"),
            objective=data.get("objective"),
            model_type=data.get("model_type"),
            count_threshold=data.get("count_threshold"),
            resolution_threshold=data.get("resolution_threshold"),
            run_time_threshold=data.get("run_time_threshold"),
            working_dir=working_dir,
            log_level=data.get("log_level"),
        )


class HPLCOptimizationBenchmarkConfig(object):
    """
    Configuration class that contains all parameters for data-driven optimization of HPLC runs.
    """

    __num_compounds: int = 10
    __num_compound_sets: int = 50
    __iterations_per_compound_set: int = 1
    __compound_set_dir: Path = Path(__file__).resolve().parents[2] / "scripts" / "compound_sets"

    def __init__(
            self,
            optimization_config: HPLCOptimizationConfig,
            num_compounds: Optional[int] = None,
            num_compound_sets: Optional[int] = None,
            compound_set_hash: Optional[str] = None,
            compound_set_dir: Optional[Path] = None,
            iterations_per_compound_set: Optional[int] = None,
    ):

        self.optimization_config = optimization_config

        self.num_compounds = num_compounds if num_compounds is not None else optimization_config.num_compounds
        self.optimization_config.num_compounds = self.num_compounds
        self.num_compound_sets = num_compound_sets if num_compound_sets is not None else self.__num_compound_sets
        self.iterations_per_compound_set = iterations_per_compound_set if iterations_per_compound_set is not None else self.__iterations_per_compound_set

        for attr in ("num_segments", "min_segment_length", "max_run_time", "segment_based", "flow_rate", "compound_idx",
                     "optimization_strategy", "retention_model", "simulation_interval", "fidelity_strategy", "retention_error_scale",
                     "retention_miss_probability", "retention_observation_sigma",
                     "column_length", "column_inner_diameter", "experiment_config", "advanced_peak_detection", "surrogate_model", "acquisition_function",
                     "budget", "seed_budget", "seed_strategy", "multi_fidelity",
                     "max_simulation_fidelity", "simulation_fidelity_decay", "low_fidelity_cost",
                     "max_consecutive_low_fidelity", "objective", "model_type",
                     "count_threshold", "resolution_threshold", "run_time_threshold"):
            setattr(self, attr, getattr(self.optimization_config, attr))

        self.compound_set_dir = compound_set_dir if compound_set_dir is not None else self.__compound_set_dir
        if compound_set_hash is None:
            cmpd_dir = [d for d in self.compound_set_dir.glob(f"{self.num_compound_sets}x{self.num_compounds}C_{int(self.max_run_time)}min_*")]
            if len(cmpd_dir) == 0:
                raise ValueError(f"No compound sets found for {self.num_compound_sets} sets of {self.num_compounds} compounds.")
            compound_set_hash = cmpd_dir[0].name.split("_")[-1]
        self.compound_set_hash = compound_set_hash

        root = self.optimization_config.working_dir or Path.cwd() / "results"
        self.optimization_config.working_dir = root if root.name == self.tag else root / self.tag

    def initialize_torch(self):
        self.optimization_config.initialize_torch()

    @property
    def tag(self) -> str:
        """
        Creates a unique tag for the current configuration based on its parameters.

        Returns:
            str: The unique tag string.
        """
        return f"OptimizationBenchmark_{self.num_compound_sets}x{self.num_compounds}C-{self.compound_set_hash}_Repeats-{self.iterations_per_compound_set}_{self.optimization_config.tag}"

    @property
    def compound_sets_folder(self) -> Path:
        tag = f"{self.num_compound_sets}x{self.num_compounds}C_{int(self.max_run_time)}min_{self.compound_set_hash}"
        return self.compound_set_dir / tag

    @property
    def logger(self) -> logging.Logger:
        return self.optimization_config.logger

    @property
    def working_dir(self) -> Path:
        return self.optimization_config.working_dir

    @property
    def single_fidelity(self) -> bool:
        return self.optimization_config.single_fidelity

    @classmethod
    def from_argparse(cls) -> "HPLCOptimizationBenchmarkConfig":

        parser = ArgumentParser(description="HPLC Benchmark Configuration")
        parser.add_argument("--num_compounds", type=int, default=None, help="Number of compounds in each set.")
        parser.add_argument("--compound_idx", type=int, default=None, help="Index of the compound to optimize for in single-peak resolution objective.")
        parser.add_argument("--num_compound_sets", type=int, default=None, help="Number of compound sets to optimize over.")
        parser.add_argument("--compound_set_hash", type=str, default=None, help="Hash identifier for the compound sets.")
        parser.add_argument("--compound_set_dir", type=str, default=None, help="Directory containing the compound sets.")
        parser.add_argument("--iterations_per_compound_set", type=int, default=None, help="Number of optimization iterations per compound set.")
        parser.add_argument("--num_segments", type=int, default=None, help="Number of segments in the multilinear gradient.")
        parser.add_argument("--min_segment_length", type=float, default=None, help="Minimum length of each gradient segment (minutes).")
        parser.add_argument("--segment_based", action="store_true", default=None, help="Optimize segment lengths instead of time points.")
        parser.add_argument("--flow_rate", type=float, default=None, help="Flow rate in mL/min.")
        parser.add_argument("--max_run_time", type=float, default=None, help="Maximum run time for the gradient (minutes).")
        parser.add_argument("--column_length", type=float, default=None, help="Column length in meters.")
        parser.add_argument("--column_inner_diameter", type=float, default=None, help="Nominal column inner diameter in meters.")
        parser.add_argument("--experiment_config", type=str, default=None, help="Path to the experiment configuration file.")
        parser.add_argument("--basic_peak_detection", action="store_true", default=None, help="Use single-trace peak detection algorithm.")
        parser.add_argument("--optimization_strategy", type=str, default=None, help="Optimization strategy.")
        parser.add_argument("--retention_model", type=str, default=None, help="Retention model.")
        parser.add_argument("--simulation_interval", type=float, default=None, help="Simulation interval.")
        parser.add_argument("--fidelity_strategy", type=str, default=None, help="Fidelity strategy.")
        parser.add_argument("--retention_error_scale", type=float, default=None, help="Retention error scale.")
        parser.add_argument("--retention_miss_probability", type=float, default=None, help="Probability of missing a peak that elutes within the run.")
        parser.add_argument("--retention_observation_sigma", type=float, default=None, help="Observation standard deviation for compoundwise retention fits, in minutes.")
        parser.add_argument("--surrogate_model", type=str, default=None, help="Type of surrogate model to use.")
        parser.add_argument("--acquisition_function", type=str, default=None, help="Acquisition function to use.")
        parser.add_argument("--budget", type=int, default=None, help="Total budget of experiments per compound set.")
        parser.add_argument("--seed_budget", type=int, default=None, help="Number of initial random seed experiments per compound set.")
        parser.add_argument("--multi_fidelity", action="store_true", default=None, help="Use multi-fidelity optimization.")
        parser.add_argument("--max_simulation_fidelity", type=float, default=None, help="Maximum simulation fidelity for multi-fidelity optimization.")
        parser.add_argument("--simulation_fidelity_decay", type=float, default=None, help="Decay rate of simulation fidelity for multi-fidelity optimization.")
        parser.add_argument("--low_fidelity_cost", type=float, default=None, help="Cost of low-fidelity experiments in multi-fidelity optimization.")
        parser.add_argument("--max_consecutive_low_fidelity", type=int, default=None, help="Maximum number of consecutive low-fidelity experiments.")
        parser.add_argument("--seed_strategy", type=str, default=None, help="Strategy for selecting seed experiments.")
        parser.add_argument("--objective", type=str, default=None, help="Type of objective function to use.")
        parser.add_argument("--model_type", choices=get_args(OptimizationModelType), default=None, help="Surrogate output representation.")
        parser.add_argument("--count_threshold", type=float, default=None, help="Threshold for missing peak count in the objective function.")
        parser.add_argument("--resolution_threshold", type=float, default=None, help="Threshold for resolution in the objective function.")
        parser.add_argument("--run_time_threshold", type=float, default=None, help="Threshold for run time in the objective function.")
        parser.add_argument("--working_dir", type=str, default=None, help="Parent directory to save results. Defaults to './results'.")
        parser.add_argument("--log_level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
        args = parser.parse_args()

        optimization_config = HPLCOptimizationConfig(
            num_compounds=args.num_compounds,
            compound_idx=args.compound_idx,
            num_segments=args.num_segments,
            min_segment_length=args.min_segment_length,
            segment_based=args.segment_based,
            max_run_time=args.max_run_time,
            flow_rate=args.flow_rate,
            column_length=args.column_length,
            column_inner_diameter=args.column_inner_diameter,
            experiment_config=Path(args.experiment_config) if args.experiment_config else None,
            advanced_peak_detection=not args.basic_peak_detection,
            optimization_strategy=args.optimization_strategy,
            retention_model=args.retention_model,
            simulation_interval=args.simulation_interval,
            fidelity_strategy=args.fidelity_strategy,
            retention_error_scale=args.retention_error_scale,
            retention_miss_probability=args.retention_miss_probability,
            retention_observation_sigma=args.retention_observation_sigma,
            surrogate_model=args.surrogate_model,
            acquisition_function=args.acquisition_function,
            budget=args.budget,
            seed_budget=args.seed_budget,
            seed_strategy=args.seed_strategy,
            multi_fidelity=args.multi_fidelity,
            max_simulation_fidelity=args.max_simulation_fidelity,
            simulation_fidelity_decay=args.simulation_fidelity_decay,
            low_fidelity_cost=args.low_fidelity_cost,
            max_consecutive_low_fidelity=args.max_consecutive_low_fidelity,
            objective=args.objective,
            model_type=args.model_type,
            count_threshold=args.count_threshold,
            resolution_threshold=args.resolution_threshold,
            run_time_threshold=args.run_time_threshold,
            working_dir=Path(args.working_dir) if args.working_dir else None,
            log_level=args.log_level,
        )

        return cls(
            optimization_config=optimization_config,
            num_compounds=args.num_compounds,
            num_compound_sets=args.num_compound_sets,
            compound_set_hash=args.compound_set_hash,
            compound_set_dir=Path(args.compound_set_dir) if args.compound_set_dir else None,
            iterations_per_compound_set=args.iterations_per_compound_set,
        )

    def to_dict(self) -> dict:
        return {
            "optimization_config": self.optimization_config.to_dict(),
            "num_compounds": self.num_compounds,
            "num_compound_sets": self.num_compound_sets,
            "compound_set_hash": self.compound_set_hash,
            "compound_set_dir": str(self.compound_set_dir),
            "iterations_per_compound_set": self.iterations_per_compound_set,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HPLCOptimizationBenchmarkConfig":
        optimization_config = HPLCOptimizationConfig.from_dict(data["optimization_config"])
        return cls(
            optimization_config=optimization_config,
            num_compounds=data.get("num_compounds"),
            num_compound_sets=data.get("num_compound_sets"),
            compound_set_hash=data.get("compound_set_hash"),
            compound_set_dir=Path(data["compound_set_dir"]) if data.get("compound_set_dir") else None,
            iterations_per_compound_set=data.get("iterations_per_compound_set"),
        )
