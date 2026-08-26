from typing import Optional
import numpy as np
import torch

from botorch.acquisition import MCAcquisitionObjective
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botier import AuxiliaryObjective, HierarchyScalarizationObjective
from botier.utils import logistic, smoothmin

from ..utils.numpy_torch import convert_to_torch, torch_nanmin


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

        return torch.stack(
            [obj(samples, X, normalize=self._norm) for obj in self.objectives],
            dim=-1
        )


def get_hplc_objective(
        hierarchical: bool = True,
        objective_type: str = "cumulative",
        num_compounds: int = 10,
        compound_idx: Optional[int] = None,
        count_bounds: tuple[float, float] = (0.0, 10.0),
        count_threshold: float = 0.5,
        resolution_objective: str = "critical",
        resolution_bounds: tuple[float, float] = (1.0, 5.0),
        resolution_threshold: float = 1.5,
        average_peak_width: float = 0.5,
        run_time_bounds: tuple[float, float] = (5.0, 30.0),
        run_time_threshold: float = 15.0,
        segment_based: bool = False,
) -> MCAcquisitionObjective | MCMultiOutputObjective:
    """
    Creates a HierarchyScalarizationObjective for HPLC optimization by hierarchically considering the following
    objectives:
        1. Detection Fraction, i.e. the fraction of detected compounds (to be maximized)
        2. Resolution Objective (specified by `resolution_objective`), i.e. either:
            - Critical Resolution, i.e. the minimum resolution between any two detected compounds (to be maximized)
            - Resolution Fraction, i.e. the fraction of peaks for which the resolution is above a certain threshold (to
              be maximized)
        3. Run Time, i.e. the total run time of the gradient (to be minimized)

    Assumes that model inputs come in the form of a multilinear gradient parameterization, i.e.
      - [t_1, t_2, ..., t_n, f_0, f_1, ..., f_n], so that t_n is the total run time (`segment_based` = False)
      - [dt_1, dt_2, ..., dt_n, f_0, f_1, ..., f_n], so that sum(dt_i) is the total run time (`segment_based` = True)

    Assumes that the model outputs come in the form of ...
        a. [detection_fraction, resolution_objective] if `objective_type` is "cumulative"
        b. [t_r1, t_r2, ..., t_rN] if `objective_type` is "compoundwise"

    Args:
        hierarchical (bool): Whether to use hierarchical scalarization. If False, returns a simple ObjectiveCalculator
                             with the three objectives.
        objective_type (str): The type of objective to use. Either "cumulative or "compoundwise"
        num_compounds (int): The total number of compounds in the compound set.
        compound_idx (int, optional): If `objective_type` is "compoundwise", a single compound can be selected for
                                      target_specific optimization. If None, the detection count / resolution count is
                                      calculated across all compounds. Defaults to None.
        count_bounds (tuple): Lower and upper bounds for detected compound count.
        count_threshold (float): Satisfaction threshold for detected compound count.
        resolution_objective (str): The type of resolution objective to use. Either "critical" or "count".
        resolution_bounds (tuple): Lower and upper bounds for resolution.
        resolution_threshold (float): Satisfaction threshold for inter-peak resolution.
        average_peak_width (float): The average peak width used to calculate resolutions between compounds. Only
                                    used for compoundwise objectives.
        run_time_bounds (tuple): Lower and upper bounds for run time.
        run_time_threshold (float): Satisfaction threshold for run time.
        segment_based (bool): Whether the gradient parameterization is segment-based. If True, the run time is assumed
                              to be the sum of all time segments. If False, the run time is assumed to be the last time
                              point of the entire gradient.

    Returns:
        HierarchyScalarizationObjective: The combined hierarchical objective.
    """
    if objective_type not in ["cumulative", "compoundwise"]:
        raise ValueError(f"Invalid objective_type: {objective_type}. Must be 'cumulative' or 'compoundwise'.")

    # 1. Define the detection fraction objective
    if objective_type == "cumulative":
        detection_objective = _get_cumulative_count(count_bounds, count_threshold, 0, num_compounds)
    elif objective_type == "compoundwise":
        detection_objective = _get_compoundwise_detection_count(
            num_compounds=num_compounds,
            compound_idx=compound_idx,
            bounds=count_bounds,
            threshold=count_threshold,
            segment_based=segment_based
        )
    else:
        raise ValueError(f"Invalid objective_type: {objective_type}. Must be 'cumulative' or 'compoundwise'.")

    # 2. Define the resolution objective
    if resolution_objective == "critical":
        if objective_type == "cumulative":
            resolution_objective = _get_cumulative_score(resolution_bounds, resolution_threshold, 1)
        elif objective_type == "compoundwise":
            resolution_objective = _get_compoundwise_critical_resolution_objective(
                compound_idx=compound_idx,
                bounds=resolution_bounds,
                threshold=resolution_threshold,
                average_peak_width=average_peak_width
            )
    elif resolution_objective == "count":
        if objective_type == "cumulative":
            resolution_objective = _get_cumulative_count(count_bounds, count_threshold, 1, num_compounds)
        elif objective_type == "compoundwise":
            resolution_objective = _get_compoundwise_resolution_count(
                compound_idx=compound_idx,
                num_compounds=num_compounds,
                bounds=count_bounds,
                threshold=count_threshold,
                resolution_threshold=resolution_threshold,
                average_peak_width=average_peak_width
            )
    else:
        raise ValueError(f"Invalid resolution_objective: {resolution_objective}. Must be 'critical' or 'fraction'.")

    run_time_objective = _get_run_time_objective(
        bounds=run_time_bounds,
        threshold=run_time_threshold,
        segment_based=segment_based
    )

    if hierarchical is True:
        return HierarchyScalarizationObjective(objectives=[detection_objective, resolution_objective, run_time_objective])
    else:
        return ObjectiveCalculator(objectives=[detection_objective, resolution_objective, run_time_objective])


def _get_cumulative_count(
        bounds: tuple[float, float],
        threshold: float,
        model_output_index: int,
        num_compounds: int
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for obtaining the cumulative detection / resolution count (as a minimization objective
    of undetected / unresolved compounds) from a model output.
    """
    return AuxiliaryObjective(
        maximize=False,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        abs_threshold=threshold,
        calculation=lambda y, x: num_compounds - y[..., model_output_index]
    )


def _get_cumulative_score(
        bounds: tuple[float, float],
        threshold: float,
        model_output_index: int,
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for obtaining the objective directly from a model output.
    """
    return AuxiliaryObjective(
        maximize=True,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        abs_threshold=threshold,
        output_index=model_output_index
    )


def _get_compoundwise_detection_count(
        num_compounds: int,
        compound_idx: Optional[int] = None,
        bounds: tuple[float, float] = (0.0, 10.0),
        threshold: float = 0.5,
        model_output_indices: slice = None,
        segment_based: bool = False,
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for calculating the detection count from a model output of individual retention
    times for each compound.

    Args:
        bounds (tuple): Lower and upper bounds for detection fraction.
        threshold (float): Satisfaction threshold for detection fraction.
        model_output_indices (slice, optional): Slice to interpret on specific model outputs as compound retention time
                                          predictions. Defaults to None, which uses all outputs.
        segment_based (bool): Whether the gradient parameterization is segment-based. If True, the run time is assumed
                              to be the sum of all time segments. If False, the run time is assumed to be the last time
                              point of the entire gradient.

    """
    if compound_idx is not None:
        num_compounds = 1
        bounds = (0.0, 1.0)

    if segment_based:
        def calculate_detection_count(model_outputs, model_inputs):
            gradient_params = convert_to_torch(model_inputs)
            gradient_run_times = torch.sum(gradient_params[..., :gradient_params.shape[-1] // 2], dim=-1, keepdim=True)
            retention_times = convert_to_torch(model_outputs)
            if model_output_indices is not None:
                retention_times = retention_times[..., model_output_indices]
            if compound_idx is not None:
                retention_times = retention_times[..., compound_idx:compound_idx+1]
            coefficients = logistic(-retention_times, -gradient_run_times, 1E1)
            detected_compounds = torch.sum(coefficients, dim=-1)
            return num_compounds - detected_compounds

    else:
        def calculate_detection_count(model_outputs, model_inputs):
            gradient_params = convert_to_torch(model_inputs)
            gradient_run_times = gradient_params[..., gradient_params.shape[-1] // 2 - 1].unsqueeze(-1)
            retention_times = convert_to_torch(model_outputs)
            if model_output_indices is not None:
                retention_times = retention_times[..., model_output_indices]
            if compound_idx is not None:
                retention_times = retention_times[..., compound_idx:compound_idx+1]
            coefficients = logistic(-retention_times, -gradient_run_times, 1E1)
            detected_compounds = torch.sum(coefficients, dim=-1)
            return num_compounds - detected_compounds

    return AuxiliaryObjective(
        maximize=False,
        calculation=calculate_detection_count,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        abs_threshold=threshold,
    )


def _get_compoundwise_critical_resolution_objective(
        compound_idx: Optional[int] = None,
        bounds: tuple[float, float] = (1.0, 5.0),
        threshold: float = 1.5,
        model_output_indices: slice = None,
        average_peak_width: float = 0.5
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for calculating the critical resolution from a model output of individual retention
    times for each compound.

    Args:
        compound_idx (int, optional): If specified, only the critical resolution for the selected compound is
                                      calculated. If None, the critical resolution across all compounds is calculated.
                                      Defaults to None.
        bounds (tuple): Lower and upper bounds for critical resolution.
        threshold (float): Satisfaction threshold for critical resolution.
        model_output_indices (slice, optional): Slice to interpret on specific model outputs as compound retention time 
                                          predictions. Defaults to None, which uses all outputs.
        average_peak_width (float): The average peak width used to calculate resolutions between compounds.
                                    Assumes resolution = (t_j - t_i) / average_peak_width.
    """
    def calculate_critical_resolution(model_outputs):

        retention_times = convert_to_torch(model_outputs)
        if model_output_indices is not None:
            retention_times = retention_times[..., model_output_indices]
        sorted_times, sorting_indices = torch.sort(retention_times, dim=-1)
        resolutions = (sorted_times[..., 1:] - sorted_times[..., :-1]) / average_peak_width

        if compound_idx is None:
            return torch_nanmin(resolutions, dim=-1).values
        else:
            forward_resolutions_sorted = torch.cat((resolutions, resolutions[..., -1:]), dim=-1)
            backward_resolutions_sorted = torch.cat((resolutions[..., :1], resolutions), dim=-1)

            n = sorting_indices.shape[-1]
            positions = torch.empty_like(sorting_indices)  # (..., num_compounds)
            positions.scatter_(dim=-1, index=sorting_indices, src=torch.arange(n).expand_as(sorting_indices))
            forward_resolutions = torch.gather(forward_resolutions_sorted, dim=-1, index=positions)
            backward_resolutions = torch.gather(backward_resolutions_sorted, dim=-1, index=positions)
            return smoothmin(
                forward_resolutions[..., compound_idx],
                backward_resolutions[..., compound_idx], 1E1
            )

    return AuxiliaryObjective(
        maximize=True,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        abs_threshold=threshold,
        calculation=lambda y, x: calculate_critical_resolution(y)
    )


def _get_compoundwise_resolution_count(
        num_compounds: int,
        compound_idx: Optional[int] = None,
        bounds: tuple[float, float] = (0.0, 10.0),
        threshold: float = 0.5,
        resolution_threshold: float = 1.5,
        model_output_indices: slice = None,
        average_peak_width: float = 0.5
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for calculating the resolution fraction from a model output of individual retention
    times for each compound.

    Args:
        bounds (tuple): Lower and upper bounds for resolution fraction.
        compound_idx (int, optional): If specified, only the resolution count for the selected compound is calculated.
                                      If None, the resolution count across all compounds is calculated. Defaults to
                                      None.
        threshold (float): Satisfaction threshold for resolution fraction.
        resolution_threshold (float): The resolution threshold above which peaks are considered resolved.
        model_output_indices (slice, optional): Slice to interpret on specific model outputs as compound retention time
                                          predictions. Defaults to None, which uses all outputs.
        average_peak_width (float): The average peak width used to calculate resolutions between compounds.
                                    Assumes resolution = (t_j - t_i) / average_peak_width.
    """
    if compound_idx is not None:
        num_compounds = 1
        bounds = (0.0, 1.0)

    def calculate_resolution_count(model_outputs):

        retention_times = convert_to_torch(model_outputs)
        if model_output_indices is not None:
            retention_times = retention_times[..., model_output_indices]
        sorted_times, sorting_indices = torch.sort(retention_times, dim=-1)
        resolutions = (sorted_times[..., 1:] - sorted_times[..., :-1]) / average_peak_width

        forward_resolutions_sorted = torch.cat((resolutions, resolutions[..., -1:]), dim=-1)
        backward_resolutions_sorted = torch.cat((resolutions[..., :1], resolutions), dim=-1)

        if compound_idx is None:
            resolutions = smoothmin(forward_resolutions_sorted, backward_resolutions_sorted, 1E1)
            num_resolved = torch.sum(logistic(resolutions, resolution_threshold, 1E1), dim=-1)
            return num_compounds - num_resolved
        else:
            n = sorting_indices.shape[-1]
            positions = torch.empty_like(sorting_indices)
            positions.scatter_(dim=-1, index=sorting_indices, src=torch.arange(n).expand_as(sorting_indices))
            forward_resolutions = torch.gather(forward_resolutions_sorted, dim=-1, index=positions)
            backward_resolutions = torch.gather(backward_resolutions_sorted, dim=-1, index=positions)
            resolutions = smoothmin(forward_resolutions[..., compound_idx], backward_resolutions[..., compound_idx], 1E1).unsqueeze(-1)
            coefficients = logistic(resolutions, resolution_threshold, 1E1)
            n_resolved = torch.sum(coefficients, dim=-1)
            return 1 - n_resolved

    return AuxiliaryObjective(
        maximize=False,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        abs_threshold=threshold,
        calculation=lambda y, x: calculate_resolution_count(y)
    )


def _get_run_time_objective(
        bounds: tuple[float, float] = (5.0, 30.0),
        threshold: float = 15.0,
        segment_based: bool = False
) -> AuxiliaryObjective:
    """
    Creates an AuxiliaryObjective for calculating the run time from model inputs.

    Assumes that model inputs come in the form of a multilinear gradient parameterization, i.e.
      - [t_1, t_2, ..., t_n, f_0, f_1, ..., f_n], so that t_n is the total run time (`segment_based` = False)
      - [dt_1, dt_2, ..., dt_n, f_0, f_1, ..., f_n], so that sum(dt_i) is the total run time (`segment_based` = True)

    Args:
        bounds (tuple): Lower and upper bounds for run time.
        threshold (float): Satisfaction threshold for run time.
        segment_based (bool): Whether the gradient parameterization is segment-based. If True, the run time is assumed
                              to be the sum of all time segments. If False, the run time is assumed to be the last time
                              point of the entire gradient.
    """
    def calculate_run_time(model_inputs):
        if segment_based is False:
            if isinstance(model_inputs, torch.Tensor):
                return model_inputs[..., model_inputs.shape[-1] // 2 - 1]
            elif isinstance(model_inputs, np.ndarray):
                return model_inputs[:, model_inputs.shape[1] // 2 - 1]
        else:
            if isinstance(model_inputs, torch.Tensor):
                time_segments = model_inputs[..., :model_inputs.shape[-1] // 2]
                return torch.sum(time_segments, dim=-1)
            elif isinstance(model_inputs, np.ndarray):
                time_segments = model_inputs[:, :model_inputs.shape[1] // 2]
                return np.sum(time_segments, axis=-1)

    return AuxiliaryObjective(
        maximize=False,
        lower_bound=bounds[0],
        upper_bound=bounds[1],
        abs_threshold=threshold,
        calculation=lambda y, x: calculate_run_time(x)
    )
