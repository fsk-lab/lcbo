from typing import Optional, Literal
import torch

from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botier import AuxiliaryObjective, HierarchyScalarizationObjective

from ..utils.numpy_torch import convert_to_torch


class ObjectiveCalculator(MCMultiOutputObjective):

    def __init__(self, objectives: list[AuxiliaryObjective], normalized_objectives: bool = True):
        """
        Implementation of a naive objective calculator that can be used as a :class:`MCMultiOutputObjective`
        for botorch-type multi-objective acquisition functions. Calculates each objective from the model outputs and
        inputs using the provided list of :class:`AuxiliaryObjective` instances.

        Args:
            objectives (list[AuxiliaryObjective]): A list of AuxiliaryObjective instances.
            normalized_objectives (bool): Whether to normalize the objectives.
        """
        MCMultiOutputObjective.__init__(self)
        self.objectives = objectives
        self._norm = normalized_objectives

    def forward(self, samples: torch.Tensor, X: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Implementation of the forward pass.

        Args:
            samples (torch.Tensor): A :code:`... x m`-dim Tensors of samples (e.g. from a model posterior, in this case,
                                    the tensor shape is :code:`sample_shape x batch_shape x q x m`)
            X (torch.Tensor, optional): A :code:`... x d`-dim tensor of inputs. Relevant only if the objective depends
                                        on the inputs explicitly.

        Returns:
            Tensor: A :code:`... x n_obj`-dim Tensor of objective values.
        """
        if X is not None:
            if samples.dim() != X.dim():
                X = X.expand(*samples.size()[:-1], X.size(dim=-1))

        return torch.stack([obj(samples, X, normalize=self._norm) for obj in self.objectives], dim=-1)


def get_hplc_objective(
        hierarchical: bool = True,
        model_type: Literal["cumulative", "compoundwise"] = "cumulative",
        num_compounds: int = 10,
        compound_idx: Optional[int] = None,
        count_threshold: float = 0.5,
        resolution_objective: str = "critical",
        resolution_bounds: tuple[float, float] = (1.0, 5.0),
        resolution_threshold: float = 1.5,
        run_time_bounds: tuple[float, float] = (5.0, 30.0),
        run_time_threshold: float = 15.0,
        segment_based: bool = False,
) -> MCAcquisitionObjective | MCMultiOutputObjective:
    """
    Creates an optimization objective for HPLC gradient optimization, consisting of three sub-objectives:
        1. Detection Count, i.e. the number of detected compounds (to be maximized)
        2. Resolution Objective (specified by `resolution_objective`), i.e. either:
            - Critical Resolution, i.e. the minimum resolution between any two detected compounds (to be maximized)
            - Resolution Count, i.e. the number of peaks for which the resolution is above a certain threshold (to
              be maximized)
        3. Run Time, i.e. the total run time of the gradient (to be minimized)

    Assumes that model inputs come in the form of a multilinear gradient parameterization, i.e.
      - [t_1, t_2, ..., t_n, f_0, f_1, ..., f_n], so that t_n is the total run time (`segment_based` = False)
      - [dt_1, dt_2, ..., dt_n, f_0, f_1, ..., f_n], so that sum(dt_i) is the total run time (`segment_based` = True)

    Assumes that the model outputs come in the form of ...
        a. [detection_count, resolution_objective] if `model_type` is "cumulative"
        b. [t_r1, t_r2, ..., t_rN, log_fwhm1, log_fwhm2, ..., log_fwhmN] if `model_type` is "compoundwise"

    Args:
        hierarchical (bool): Whether to use hierarchical scalarization. If False, returns a simple ObjectiveCalculator
                             with the three objectives.
        model_type (Literal["cumulative", "compoundwise"]): Whether to use a cumulative "black-box" model or a
                                                            compoundwise model from which detection and resolution
                                                            objectives can be calculated.
        num_compounds (int): The total number of compounds in the compound set.
        compound_idx (int, optional): If `model_type` is "compoundwise", a single compound can be selected for
                                      target_specific optimization. If None, the detection count / resolution count is
                                      calculated across all compounds. Defaults to None.
        count_threshold (float): Satisfaction threshold for detected compound count.
        resolution_objective (str): The type of resolution objective to use. Either "critical" or "count".
        resolution_bounds (tuple): Lower and upper bounds for resolution.
        resolution_threshold (float): Satisfaction threshold for inter-peak resolution.
        run_time_bounds (tuple): Lower and upper bounds for run time.
        run_time_threshold (float): Satisfaction threshold for run time.
        segment_based (bool): Whether the gradient parameterization is segment-based. If True, the run time is assumed
                              to be the sum of all time segments. If False, the run time is assumed to be the last time
                              point of the entire gradient.

    Returns:
        MCAcquisitionObjective | MCMultiOutputObjective: An acquisition objective that can be used in botorch-type
                                                         acquisition functions.
    """
    if model_type not in ["cumulative", "compoundwise"]:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'cumulative' or 'compoundwise'.")

    if model_type == "cumulative":
        detection_objective = _get_count_objective_from_cumulative_model(
            threshold=count_threshold,
            model_output_index=0,
            num_compounds=num_compounds if compound_idx is None else 1
        )

    elif model_type == "compoundwise":
        detection_objective = _get_detection_count_objective_from_compoundwise_model(
            num_compounds=num_compounds,
            compound_idx=compound_idx,
            threshold=count_threshold,
            segment_based=segment_based
        )
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'cumulative' or 'compoundwise'.")

    if resolution_objective == "critical":
        if model_type == "cumulative":
            resolution_objective = _get_objective_from_cumulative_model(
                bounds=resolution_bounds,
                threshold=resolution_threshold,
                model_output_index=1
            )
        elif model_type == "compoundwise":
            resolution_objective = _get_critical_resolution_objective_from_compoundwise_model(
                compound_idx=compound_idx,
                bounds=resolution_bounds,
                threshold=resolution_threshold,
                segment_based=segment_based
            )
    elif resolution_objective == "count":
        if model_type == "cumulative":
            resolution_objective = _get_count_objective_from_cumulative_model(
                threshold=count_threshold,
                model_output_index=1,
                num_compounds=num_compounds if compound_idx is None else 1
            )
        elif model_type == "compoundwise":
            resolution_objective = _get_resolution_count_objective_from_compoundwise_model(
                compound_idx=compound_idx,
                num_compounds=num_compounds,
                threshold=count_threshold,
                resolution_threshold=resolution_threshold,
                segment_based=segment_based
            )
    else:
        raise ValueError(f"Invalid resolution_objective: {resolution_objective}. Must be 'critical' or 'count'.")

    run_time_objective = _get_run_time_objective(
        bounds=run_time_bounds,
        threshold=run_time_threshold,
        segment_based=segment_based
    )

    if hierarchical is True:
        return HierarchyScalarizationObjective(
            objectives=[detection_objective, resolution_objective, run_time_objective]
        )
    else:
        return ObjectiveCalculator(objectives=[detection_objective, resolution_objective, run_time_objective])


def _get_count_objective_from_cumulative_model(
        threshold: float,
        model_output_index: int,
        num_compounds: int
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for obtaining the detection / resolution count (as a minimization objective
    of undetected / unresolved compounds) from a cumulative model output.

    Args:
        threshold (float): Satisfaction threshold for detection / resolution count.
        model_output_index (int): Index of the model output corresponding to the detection / resolution count.
        num_compounds (int): Total number of compounds in the compound set.

    Returns:
        AuxiliaryObjective: An AuxiliaryObjective instance for the cumulative detection / resolution count.
    """
    return AuxiliaryObjective(
        maximize=False,
        lower_bound=0,
        upper_bound=num_compounds,
        abs_threshold=threshold,
        calculation=lambda y, x: num_compounds - y[..., model_output_index]
    )


def _get_objective_from_cumulative_model(
        bounds: tuple[float, float],
        threshold: float,
        model_output_index: int,
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for obtaining the objective directly from a cumulative model output.

    Args:
        bounds (tuple): Lower and upper bounds for the objective.
        threshold (float): Satisfaction threshold for the objective.
        model_output_index (int): Index of the model output corresponding to the objective.

    Returns:
        AuxiliaryObjective: An AuxiliaryObjective instance for the cumulative objective.
    """
    return AuxiliaryObjective(
        maximize=True,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        abs_threshold=threshold,
        output_index=model_output_index
    )


def _calculate_gradient_run_time(model_inputs, segment_based: bool):
    """
    Helper function to calculate the total run time of a gradient from model inputs (i.e. the gradient parameters),
    handling both segment-based and non-segment-based parameterizations.

    Args:
        model_inputs: The model inputs (gradient parameters) as a torch.Tensor or numpy.ndarray. Shape:
                      (..., 2 * n_segments + 1)
        segment_based (bool): Whether the gradient parameterization is segment-based.

    Returns:
        torch.Tensor: The total run time of each gradient. Shape: (..., 1)
    """
    gradient_params = convert_to_torch(model_inputs)
    if segment_based:
        return torch.sum(gradient_params[..., :gradient_params.shape[-1] // 2], dim=-1, keepdim=True)
    else:
        return gradient_params[..., gradient_params.shape[-1] // 2 - 1].unsqueeze(-1)


def _process_compoundwise_model_outputs(
        model_outputs,
        compound_idx: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to process compoundwise model outputs and extract retention times and peak FWHMs.

    Args:
        model_outputs: The model outputs as a torch.Tensor. Shape: (..., 2 * num_compounds)
        compound_idx (int, optional): If specified, only the retention time and peak FWHM for this compound index will
                                      be extracted.

    Returns:
        torch.Tensor: The retention times of the compounds. Shape: (..., num_compounds) or (..., 1) if compound_idx is
                      specified.
        torch.Tensor: The peak FWHMs of the compounds. Shape: (..., num_compounds) or (..., 1) if compound_idx is
                      specified.
    """
    num_compounds = model_outputs.shape[-1] // 2

    retention_times = convert_to_torch(model_outputs)[..., :num_compounds]
    fwhms = torch.exp(convert_to_torch(model_outputs)[..., num_compounds:].clamp(-20, 10)) # TODO: Adjust the clamp bounds

    if compound_idx is not None:
        retention_times = retention_times[..., compound_idx:compound_idx + 1]
        fwhms = fwhms[..., compound_idx:compound_idx + 1]

    return retention_times, fwhms


def _calculate_compoundwise_elution_probabilities(
        retention_times: torch.Tensor,
        peak_fwhms: torch.Tensor,
        gradient_run_times: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the elution probabilities for each compound based on retention times, peak FWHMs, and gradient run times.

    Args:
        retention_times (torch.Tensor): Retention times of the compounds. Shape: (..., num_compounds)
        peak_fwhms (torch.Tensor): Peak FWHMs of the compounds. Shape: (..., num_compounds)
        gradient_run_times (torch.Tensor): Total run time of each gradient. Shape: (..., 1)

    Returns:
        torch.Tensor: Elution probabilities for each compound. Shape: (..., num_compounds)
    """
    valid = torch.isfinite(retention_times) & torch.isfinite(peak_fwhms)
    retention_times = torch.where(valid, retention_times.clamp_min(0.0), 0.0)
    peak_fwhms = torch.where(valid, peak_fwhms.clamp_min(0.01), 0.0)
    peak_ends = retention_times + 3 * peak_fwhms / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0))))  # 3-sigma
    elution_probabilities = torch.sigmoid(10 * (gradient_run_times - peak_ends))
    return torch.where(valid, elution_probabilities, 0.0)


def _get_detection_count_objective_from_compoundwise_model(
        num_compounds: int,
        compound_idx: Optional[int] = None,
        threshold: float = 0.5,
        segment_based: bool = False,
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for calculating the detection count from a model output of individual retention
    times for each compound.

    Args:
        num_compounds (int): Total number of compounds in the compound set.
        compound_idx (int, optional): If specified, only the detection count for this compound index will be calculated.
        threshold (float): Satisfaction threshold for detection count.
        segment_based (bool): Whether the gradient parameterization is segment-based.

    Returns:
        AuxiliaryObjective: An AuxiliaryObjective instance for the detection count.
    """
    if compound_idx is not None:
        num_compounds = 1  # Only one compound is considered for detection count

    def calculate_detection_count(model_outputs: torch.Tensor, model_inputs: torch.Tensor) -> torch.Tensor:
        gradient_run_times = _calculate_gradient_run_time(model_inputs, segment_based)
        retention_times, peak_fwhms = _process_compoundwise_model_outputs(model_outputs, compound_idx)
        elution_probabilities = _calculate_compoundwise_elution_probabilities(retention_times, peak_fwhms, gradient_run_times)
        detected_compounds = torch.sum(elution_probabilities, dim=-1)
        return num_compounds - detected_compounds

    return AuxiliaryObjective(
        maximize=False,
        calculation=calculate_detection_count,
        lower_bound=0.0,
        upper_bound=num_compounds,
        abs_threshold=threshold,
    )


def _calculate_compoundwise_resolution(
        retention_times: torch.Tensor,
        peak_fwhms: torch.Tensor,
        gradient_run_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate per-compound resolutions (minimum of the two resolutions to adjacent peaks) from predicted retention times
    and FWHMs. The first and last detected peaks each use their only adjacent peak. A peak is detected when its
    retention time and FWHM are  finite and its 3-sigma peak end is within the gradient run time. Non-detected peaks
    have a resolution of 0; a single detected peak without neighbors has a resolution of infinity.

    Args:
        retention_times (torch.Tensor): Retention times of the compounds. Shape: (..., num_compounds)
        peak_fwhms (torch.Tensor): Peak FWHMs of the compounds. Shape: (..., num_compounds)
        gradient_run_times (torch.Tensor): Total run time of each gradient. Shape: (..., 1)

    Returns:
        torch.Tensor: Resolution for each compound in the original input order. Shape: (..., num_compounds)
        torch.Tensor: Boolean tensor indicating whether each compound has at least one neighbor. Shape: (..., num_compounds)
    """
    valid = torch.isfinite(retention_times) & torch.isfinite(peak_fwhms)
    retention_times = torch.where(valid, retention_times.clamp_min(0.0), 0.0)
    peak_fwhms = torch.where(valid, peak_fwhms.clamp_min(0.01), 0.0)
    peak_sigmas = peak_fwhms / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0))))  # FWHM to sigma
    detected = valid & (retention_times + 3 * peak_sigmas < gradient_run_times)

    sorting_indices = torch.where(detected, retention_times, torch.inf).argsort(dim=-1)
    sorted_detected = detected.gather(-1, sorting_indices)
    sorted_times = torch.where(detected, retention_times, 0.0).gather(-1, sorting_indices)
    sorted_sigmas = torch.where(detected, peak_sigmas, 1.0).gather(-1, sorting_indices)

    pair_resolutions = torch.diff(sorted_times, dim=-1) / (2 * (sorted_sigmas[..., :-1] + sorted_sigmas[..., 1:]))
    valid_pairs = sorted_detected[..., :-1] & sorted_detected[..., 1:]
    pair_resolutions = torch.where(valid_pairs, pair_resolutions, torch.inf)

    no_neighbor = torch.full_like(sorted_times[..., :1], torch.inf)
    backward_resolutions = torch.cat([no_neighbor, pair_resolutions], dim=-1)
    forward_resolutions = torch.cat([pair_resolutions, no_neighbor], dim=-1)
    sorted_resolutions = torch.minimum(backward_resolutions, forward_resolutions)
    sorted_resolutions = torch.where(sorted_detected, sorted_resolutions, 0.0)

    resolutions = torch.zeros_like(sorted_resolutions).scatter(-1, sorting_indices, sorted_resolutions)
    has_neighbor = detected & (detected.sum(dim=-1, keepdim=True) >= 2)

    return resolutions, has_neighbor


def _get_resolution_count_objective_from_compoundwise_model(
        num_compounds: int,
        compound_idx: Optional[int] = None,
        threshold: float = 0.5,
        resolution_threshold: float = 1.5,
        segment_based: bool = False
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for calculating the resolution count from a model output of individual retention
    times and peak FWHMs for each compound.

    Args:
        num_compounds (int): Total number of compounds in the compound set.
        compound_idx (int, optional): If specified, only the resolution count for this compound index will be calculated.
        threshold (float): Satisfaction threshold for resolution count.
        resolution_threshold (float): The resolution threshold for considering a peak as resolved.
        segment_based (bool): Whether the gradient parameterization is segment-based.
    """
    num_targets = 1 if compound_idx is not None else num_compounds

    def calculate_resolution_count(model_outputs: torch.Tensor, model_inputs: torch.Tensor) -> torch.Tensor:
        gradient_run_times = _calculate_gradient_run_time(model_inputs, segment_based)
        retention_times, peak_fwhms = _process_compoundwise_model_outputs(model_outputs)
        resolutions, has_neighbor = _calculate_compoundwise_resolution(retention_times, peak_fwhms, gradient_run_times)
        if compound_idx is not None:
            resolutions = resolutions[..., compound_idx:compound_idx + 1]
            has_neighbor = has_neighbor[..., compound_idx:compound_idx + 1]

        resolved = torch.sigmoid(10 * (resolutions - resolution_threshold))
        return num_targets - torch.where(has_neighbor, resolved, 0.0).sum(dim=-1)

    return AuxiliaryObjective(
        maximize=False,
        calculation=calculate_resolution_count,
        lower_bound=0.0,
        upper_bound=num_targets,
        abs_threshold=threshold
    )


def _get_critical_resolution_objective_from_compoundwise_model(
        compound_idx: Optional[int] = None,
        bounds: tuple[float, float] = (1.0, 5.0),
        threshold: float = 1.5,
        segment_based: bool = False
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for calculating the critical resolution from a model output of individual retention
    times and peak FWHMs for each compound.

    Args:
        compound_idx (int, optional): If specified, only the critical resolution for this compound index will be calculated.
        bounds (tuple): Lower and upper bounds for critical resolution.
        threshold (float): Satisfaction threshold for critical resolution.
        segment_based (bool): Whether the gradient parameterization is segment-based

    Returns:
        AuxiliaryObjective: An AuxiliaryObjective instance for the critical resolution.
    """
    def calculate_critical_resolution(model_outputs: torch.Tensor, model_inputs: torch.Tensor) -> torch.Tensor:
        gradient_run_times = _calculate_gradient_run_time(model_inputs, segment_based)
        retention_times, peak_fwhms = _process_compoundwise_model_outputs(model_outputs)
        resolutions, has_neighbor = _calculate_compoundwise_resolution(retention_times, peak_fwhms, gradient_run_times)
        if compound_idx is not None:
            resolutions = resolutions[..., compound_idx:compound_idx + 1]
            has_neighbor = has_neighbor[..., compound_idx:compound_idx + 1]

        minimum_resolution = resolutions.masked_fill(~has_neighbor, torch.inf).min(dim=-1).values
        return torch.where(has_neighbor.any(dim=-1), minimum_resolution, 0.0)


    return AuxiliaryObjective(
        maximize=True,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        abs_threshold=threshold,
        calculation=calculate_critical_resolution
    )


def _get_run_time_objective(
        bounds: tuple[float, float] = (5.0, 30.0),
        threshold: float = 15.0,
        segment_based: bool = False
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for calculating the run time from model inputs.

    Args:
        bounds (tuple): Lower and upper bounds for run time.
        threshold (float): Satisfaction threshold for run time.
        segment_based (bool): Whether the gradient parameterization is segment-based. If True, the run time is assumed
                              to be the sum of all time segments. If False, the run time is assumed to be the last time
                              point of the entire gradient.
    """
    return AuxiliaryObjective(
        maximize=False,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        abs_threshold=threshold,
        calculation=lambda y, x: _calculate_gradient_run_time(x, segment_based).squeeze(-1)
    )
