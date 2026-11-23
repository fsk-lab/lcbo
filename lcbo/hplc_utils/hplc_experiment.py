from dataclasses import dataclass
import numpy as np
from moses.datatypes import LCGradient

from .multilinear_gradient import LCGradientParameters


@dataclass
class HPLCExperiment:
    """
    Data class to represent the parameters and results of a single HPLC experiment.

    Fields:
        gradient (LCGradient): The gradient used in the experiment.
        num_compounds (int): The number of compounds in the mixture.
        retention_times (np.ndarray): Array of retention times for each compound (shape: (1, num_compounds).
        detection_fraction (float): The fraction of compounds detected.
        critical_resolution (float): The critical resolution between compounds.
        score (float): The overall score of the experiment based on defined objectives.
        high_fidelity (bool): Indicates if the experiment was conducted with high fidelity settings.
    """
    gradient: LCGradient
    num_compounds: int
    retention_times: np.ndarray  # shape: (num_compounds,)
    peak_resolutions: np.ndarray  # shape: (num_compounds, )
    fwhms: np.ndarray  # shape: (num_compounds,)
    peak_areas: np.ndarray  # shape: (num_compounds,)
    score: float = np.nan

    fidelity_level: float = 1.0
    segment_based: bool = False

    @property
    def detected(self) -> np.ndarray:
        """
        Returns a Boolean mask indicating which compounds were detected (i.e. have finite retention times, FWHMs, and
        peak areas) in the experiment.
        """
        retention_time_mask = np.isfinite(self.retention_times) & (self.retention_times > 0)
        fwhm_mask = np.isfinite(self.fwhms) & (self.fwhms > 0)
        peak_area_mask = np.isfinite(self.peak_areas) & (self.peak_areas > 0)
        return retention_time_mask & fwhm_mask & peak_area_mask

    @property
    def gradient_parameters(self) -> LCGradientParameters:
        return LCGradientParameters.from_gradient(self.gradient, segment_based=self.segment_based)

    @property
    def high_fidelity(self) -> bool:
        return self.fidelity_level >= 1.0 - 1E-9

    def single_peak_elution_resolution(self, peak_idx: int) -> tuple[float, float]:
        """
        Returns the elution (1: eluted, 0: non-eluted) and the resolution of the specified peak.
        If no finite pairwise resolution is available, return zero resolution while retaining the detection status.

        Args:
            peak_idx (int): The index of the peak to evaluate.

        Returns:
            float: 1.0 if the peak is detected (eluted), 0.0 otherwise.
            float: The resolution of the peak if detected and finite, 0.0 otherwise.
        """
        if self.detected[peak_idx]:
            resolution = self.peak_resolutions[peak_idx]
            return 1.0, resolution.item() if np.isfinite(resolution) else 0.0
        else:
            return 0.0, 0

    @property
    def num_peaks(self) -> int:
        return int(self.detected.sum())

    def num_resolved_peaks(self, resolution_threshold: float = 1.5) -> int:
        real_resolutions = self.peak_resolutions[self.detected & np.isfinite(self.peak_resolutions)]
        return int((real_resolutions >= resolution_threshold).sum())

    @property
    def critical_resolution(self) -> float:
        real_resolutions = self.peak_resolutions[self.detected & np.isfinite(self.peak_resolutions)]
        if len(real_resolutions) == 0:
            return 0.0
        return float(real_resolutions.min())

    def to_dict(self) -> dict:
        return {
            "gradient": self.gradient.to_dict(),
            "num_compounds": self.num_compounds,
            "retention_times": self.retention_times.tolist(),
            "peak_resolutions": self.peak_resolutions.tolist(),
            "fwhms": self.fwhms.tolist(),
            "peak_areas": self.peak_areas.tolist(),
            "score": self.score,
            "fidelity_level": float(self.fidelity_level),
            "segment_based": self.segment_based
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HPLCExperiment":

        # Just for backward compatibility
        if "fidelity_level" in data:
            fidelity_level = data["fidelity_level"]
        elif data.get("high_fidelity", True):
            fidelity_level = 1.0
        else:
            fidelity_level = 0.5

        return cls(
            gradient=LCGradient.from_dict(data["gradient"]),
            num_compounds=data["num_compounds"],
            retention_times=np.array(data["retention_times"]),
            peak_resolutions=np.array(data.get("peak_resolutions", [])),  # Temporary fix for backward compatibility
            fwhms=np.array(data["fwhms"]),
            peak_areas=np.array(data["peak_areas"]),
            score=data["score"],
            fidelity_level=fidelity_level,
            segment_based=data.get("segment_based", False)
        )
