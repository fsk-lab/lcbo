from typing import Optional
import numpy as np
import torch

from botorch.models.model import Model, ModelList
from botorch.acquisition import MCAcquisitionFunction, MCAcquisitionObjective
from botorch.acquisition.monte_carlo import MCSampler
from botorch.acquisition.objective import PosteriorTransform
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.transforms import t_batch_mode_transform
from botorch.optim import optimize_acqf

from ..bo_utils.surrogate import FixedFidelitySurrogate
from .random import sample_from_polytope
from ..data.config import HPLCOptimizationConfig
from ..data.campaign import HPLCOptimizationCampaign


class StandardizedTargetGap(MCAcquisitionFunction):
    """
    Acquisition Function that implements the a standardized gap to a target value as the value of a candidate point.
    This is derived from the "Optimization as Estimation" approach (Wang, Zhou, Jegelka, AISTATS 2016), but uses an
    expert-provided target value instead of the estimation of the global optimum. The acquisition function value is
    calculated as:

    a(x) = (µ(x) - target) / σ(x) + epsilon

    where µ(x) and σ(x) are the posterior mean and standard deviation of the surrogate model at point x, respectively.

    Args:
        model (Model): The surrogate model (as in a typical BoTorch acquisition function).
        target_value (float): The target value to which the gap is calculated.
        sampler (MCSampler, optional): The MC sampler (as in a typical BoTorch Monte Carlo acquisition function).
        objective (MCAcquisitionObjective, optional): The acquisition objective (as in a typical BoTorch Monte Carlo
                                                      acquisition function).
        posterior_transform (PosteriorTransform, optional): The posterior transform (as in a typical BoTorch Monte Carlo
                                                            acquisition function).
        X_pending (torch.Tensor, optional): The pending points (as in a typical BoTorch acquisition function).
        epsilon (float, optional): A small constant added to the denominator for numerical stability. Defaults to 1e-6.
    """

    def __init__(
            self,
            model: Model,
            target_value: float,
            sampler: Optional[MCSampler] = None,
            objective: Optional[MCAcquisitionObjective] = None,
            posterior_transform: Optional[PosteriorTransform] = None,
            X_pending: Optional[torch.Tensor] = None,
            epsilon: float = 1e-6
    ):

        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self.target_value = float(target_value)
        self.epsilon = epsilon

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass to calculate the acquisition function value at the given input points X.

        Args:
            X (torch.Tensor): A :code:`batch_size x d`-dim tensor of input points where the acquisition function should
                              be evaluated.

        Returns:
            torch.Tensor: A :code:`batch_size`-dim tensor of acquisition function values at the input points.
        """
        samples, obj = self._get_samples_and_objectives(X)
        obj = obj.squeeze(-1)  # shape: `batch_size`
        mean = obj.mean(dim=0)
        std = obj.std(dim=0).clamp_min(self.epsilon)
        return (mean - self.target_value) / std


def optimize_acquisition_function(
        surrogate: Model,
        campaign: HPLCOptimizationCampaign,
        config: HPLCOptimizationConfig,
) -> tuple[bool, torch.Tensor]:
    """
    Optimize the acquisition function to recommend the next point to evaluate.

    Args:
        surrogate (Model): The trained surrogate model.
        campaign (HPLCOptimizationCampaign): The current campaign containing past experiments.
        config (HPLCOptimizationConfig): The optimization configuration containing acquisition function settings.

    Returns:
        bool: Indicates if high-fidelity point is recommended (True) or low-fidelity (False).
        torch.Tensor: The recommended next point to evaluate.
    """
    if config.multi_fidelity:
        surrogate = FixedFidelitySurrogate(
            base_model=surrogate,
            fixed_fidelity=1.0,
            fidelity_index=config.num_optimization_variables
        )

    # Resolve the acquisition function
    if config.acquisition_function.lower() == "random":
        return True, sample_from_polytope(config.bounds, config.constraints, n_samples=1)
    elif config.acquisition_function.lower() == "ei":
        best_f = float(np.max(campaign.scores))
        acqf = qExpectedImprovement(model=surrogate, best_f=best_f, objective=config.optimization_objective)
    elif config.acquisition_function.lower() == "logei":
        best_f = float(np.max(campaign.scores))
        acqf = qLogExpectedImprovement(model=surrogate, best_f=best_f, objective=config.optimization_objective)
    elif config.acquisition_function.lower() == "ucb":
        acqf = qUpperConfidenceBound(model=surrogate, beta=1.0, objective=config.optimization_objective)
    elif config.acquisition_function.lower() == "ucbexp":
        acqf = qUpperConfidenceBound(model=surrogate, beta=5.0, objective=config.optimization_objective)
    elif config.acquisition_function.lower() == "stgap":
        acqf = StandardizedTargetGap(
            model=surrogate,
            target_value=len(config.optimization_objective.objectives),  # 3.0 for BoTier with 3 objectives
            objective=config.optimization_objective
        )
    elif config.acquisition_function.lower() == "ehvi":

        if config.multi_fidelity:
            raise NotImplementedError("Multi-fidelity EHVI is not implemented yet.")

        with torch.no_grad():
            x_prev = torch.tensor(campaign.get_gradient_parameters(fidelity_threshold=1.0)[0]).unsqueeze(0)
            y_pred = surrogate.posterior(x_prev).mean
            obj_values = config.optimization_objective(y_pred, x_prev).squeeze(0)

        ref_point = torch.zeros(obj_values.shape[-1]).to(obj_values)  # always non-negative maximization objectives
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=obj_values)

        acqf = qExpectedHypervolumeImprovement(
            model=surrogate,
            objective=config.optimization_objective,
            partitioning=partitioning,
            ref_point=ref_point
        )
    else:
        raise NotImplementedError(f"Acquisition function {config.acquisition_function} is not implemented.")

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=config.bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 5, "init_batch_limit": 1024},
        inequality_constraints=config.constraints if config.constraints else None
    )

    if config.single_fidelity:
        return True, candidates
    else:
        var_red_hf = determine_variance_reduction(
            surrogate.base_model,
            candidates,
            config,
            fidelity_level=1.0
        )

        var_red_lf = determine_variance_reduction(
            surrogate.base_model,
            candidates,
            config,
            fidelity_level=campaign.expected_fidelity_level
        )
        config.logger.debug(f"Variance reduction at candidate point: LF={var_red_lf:.2e}, HF={var_red_hf:.2e}")

        if var_red_hf <= 0:
            return True, candidates
        elif var_red_lf / config.low_fidelity_cost <= var_red_hf:
            return True, candidates

        return False, candidates


def determine_variance_reduction(
        surrogate_model: ModelList,
        candidate_point: torch.Tensor,
        config: HPLCOptimizationConfig,
        fidelity_level: float,
        num_references: int = 256,
        trust_region_quantile: float = 0.9,
) -> float:
    """
    Determines the expected variance reduction in the final optimization objective by evaluating the candidate point
    at the specified fidelity level.
        1. Samples reference points from the input space using a Sobol sequence. Keeps only the top quantile points
           based on predicted objective values (from the posterior mean).
        2. For each model output in the ModelList surrogate, computes the covariance reduction at the candidate point
           for each reference point.
        3. Computes the gradients of the optimization objective at the reference points with respect to the model
           outputs.
        4. Estimates the variance reduction through a linear approximation of the objective around the reference points:
             ΔVar[obj] ≈ Σ (∂obj/∂y_i)^2 * ΔVar[y_i]

    Args:
        surrogate_model (Model): The trained surrogate model.
        candidate_point (torch.Tensor): The candidate point to evaluate (1 x n_features).
        config (HPLCOptimizationConfig): The optimization configuration.
        fidelity_level (float): The fidelity level to evaluate (0.0 to 1.0).
        num_references (int, optional): Number of reference points to use. Defaults to 256.
        trust_region_quantile (float, optional): Quantile for trust region selection. Defaults to 0.9.

    Returns:
        float: The expected variance reduction at the candidate point.
    """
    # Step 1: Generate reference points and filter based on objective values
    ref_points = sample_from_polytope(config.bounds, config.constraints, num_references, seed=torch.initial_seed())  # (n, d)
    with torch.no_grad():
        x_ = torch.cat([ref_points, torch.ones(ref_points.shape[0], 1)], dim=-1)  # (n, d+1)
        posterior_means = surrogate_model.posterior(x_).mean  # (n, m)
    obj_values = config.optimization_objective(posterior_means, ref_points)  # (n,)
    mask = obj_values >= obj_values.quantile(trust_region_quantile)
    ref_points = ref_points[mask, ...]  # (r, d)
    posterior_means = posterior_means[mask, ...]  # (r, m)

    # Step 2: Compute covariance reduction for each model output in the ModelList
    z_ref = torch.cat([ref_points, torch.ones(ref_points.shape[0], 1)], dim=-1)  # (r, d+1)
    z_cand = torch.cat([candidate_point, torch.tensor([[fidelity_level]])], dim=-1)  # (1, d+1)
    z = torch.cat([z_ref, z_cand], dim=0)  # (r+1, d+1)

    var_red = torch.empty(z_ref.shape[0], posterior_means.shape[1])  # (r, m)

    def leaf_models(model: Model) -> list[Model]:  # needed for output ordering
        if hasattr(model, "models"):
            return [leaf for child in model.models for leaf in leaf_models(child)]
        return [model]

    output_idx = 0
    for model in leaf_models(surrogate_model):
        with torch.no_grad():
            posterior = model.posterior(z)
            cov = posterior.mvn.covariance_matrix
            if model.num_outputs == 1 or getattr(posterior.mvn, "_interleaved", True):
                cov = cov.reshape(z.shape[0], model.num_outputs, z.shape[0], model.num_outputs)
            else:
                cov = cov.reshape(model.num_outputs, z.shape[0], model.num_outputs, z.shape[0]).permute(1, 0, 3, 2)
            observation_variances = model.posterior(z_cand, observation_noise=True).variance.reshape(-1)

        for i in range(model.num_outputs):
            cov_i = cov[:, i, :, i]
            var_red[:, output_idx] = (cov_i[:-1, -1] ** 2 / observation_variances[i].clamp_min(1E-12)).detach()
            output_idx += 1

    # Step 3: Compute gradients of the objective at the reference points
    objective_gradients = torch.empty_like(posterior_means)  # (r, m)
    for i in range(posterior_means.shape[0]):
        y = posterior_means[i, :].clone().detach().requires_grad_(True).unsqueeze(0)  # (1, m)
        obj = config.optimization_objective(y, ref_points[i, :].detach().unsqueeze(0))  # (1,)
        grad, = torch.autograd.grad(obj, y, retain_graph=False, create_graph=False)  # (1, m)
        objective_gradients[i, :] = grad.squeeze(0).detach()

    # Step 4: Estimate variance reduction as
    return torch.mean(torch.sum(var_red * objective_gradients**2, dim=-1)).item()
