import math
from typing import Optional
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskVariationalGP
from botorch.models.kernels.linear_truncated_fidelity import LinearTruncatedFidelityKernel
from botorch.models.transforms import Normalize, Standardize
from botorch.optim.fit import fit_gpytorch_mll_scipy
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import GammaPrior, NormalPrior
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

from .laplace import LastLayerLaplaceModel


def prepare_retention_observations(
        y: torch.Tensor,
        t: torch.Tensor,
        miss_probability: float,
        observation_sigma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Standardize]:
    """
    Standardizes measured retention times, run limits, and observation noise on the same scale.

    Args:
        y (torch.Tensor): Measured retention times, including NaN for ambiguous non-detections. Shape: (n_samples,).
        t (torch.Tensor): Upper run time limits for censored regression. Shape: (n_samples,).
        miss_probability (float): Probability of missing a peak that eluted within the run.
        observation_sigma (float): Retention-time observation standard deviation, in minutes.

    Returns:
        torch.Tensor: Standardized targets (shape: (n_samples,)).
        torch.Tensor: Boolean mask identifying censored observations (shape: (n_samples,)).
        torch.Tensor: Miss probabilities for each observation (shape: (n_samples,)).
        torch.Tensor: Standardized observation noise (shape: (n_samples,)).
        Standardize: Outcome transform for unstandardizing predictions.
    """
    y, t = y.reshape(-1), t.reshape(-1)
    detected = torch.isfinite(y)
    targets = torch.where(detected, y, t)
    miss_probabilities = torch.where(torch.isnan(y), torch.full_like(y, miss_probability), 0.0)

    observed_times = y[detected]
    center = observed_times.mean() if len(observed_times) else t.mean()
    scale = observed_times.std() if len(observed_times) > 1 else t.median() / 4
    scale = scale.clamp_min(observation_sigma)

    outcome_transform = Standardize(m=1).to(y)
    outcome_transform.means.fill_(center), outcome_transform.stdvs.fill_(scale)
    outcome_transform._stdvs_sq.fill_(scale.square())
    outcome_transform._is_trained.fill_(True)
    outcome_transform.eval()

    standardized_targets = (targets - center) / scale
    censored_observations = ~detected
    standardized_observation_noise = observation_sigma / scale

    return standardized_targets, censored_observations, miss_probabilities, standardized_observation_noise, outcome_transform


def _log_non_detection(z: torch.Tensor, miss_probability: torch.Tensor) -> torch.Tensor:
    """
    Evaluate log(q + (1 - q) Phi(z)), including the known-late case q = 0.

    Args:
        z (torch.Tensor): Standardized retention times (shape: (n_samples,)).
        miss_probability (torch.Tensor): Probability of missing a peak that eluted within the run (shape: (n_samples,)).

    Returns:
        torch.Tensor: Log likelihood for each non-detection (shape: (n_samples,)).
    """
    log_survival = torch.special.log_ndtr(z)
    # Use an interior value in the unused branch to avoid taking log(0) in autograd.
    ambiguous = miss_probability > 0
    q = torch.where(ambiguous, miss_probability, 0.5)
    log_mixture = torch.logaddexp(q.log(), torch.log1p(-q) + log_survival)
    return torch.where(ambiguous, log_mixture, log_survival)


class CensoredGaussianLikelihood(GaussianLikelihood):
    """
    A Gaussian likelihood with special treatment for `nan` observations.

    The log likelihood for a "real" observation is the usual Gaussian log likelihood. The log likelihood for a `nan`
    observation is the log probability of a non-detection, which is a mixture of the Gaussian survival function and a
    constant probability of measurement error. The mixture is integrated by Gauss-Hermite quadrature.

    Args:
        censored (torch.Tensor): Boolean mask identifying observations that should be treated as censored (shape:
                                 (n_samples,)).
        miss_probability (torch.Tensor): Probability of measurement error (incorrect censoring).
        sigma (torch.Tensor): Standard deviation of a finite-value observation.
    """

    def __init__(self, censored: torch.Tensor, miss_probability: torch.Tensor, sigma: torch.Tensor) -> None:
        super().__init__(noise_constraint=Positive())
        self.to(sigma)
        self.noise = sigma.square()
        self.raw_noise.requires_grad_(False)
        self.register_buffer("censored", censored)
        self.register_buffer("miss_probability", miss_probability)
        self.quadrature = GaussHermiteQuadrature1D(num_locs=32).to(sigma)

    def expected_log_prob(self, target: torch.Tensor, input: MultivariateNormal, *args, **kwargs) -> torch.Tensor:
        """
        Integrate detected observations analytically and censored observations by quadrature.

        Args:
            target (torch.Tensor): The observed target values (shape: (n_samples,)).
            input (MultivariateNormal): The predicted distribution over latent function values (shape: (n_samples,)).
            *args: Additional positional arguments for the base method.
            **kwargs: Additional keyword arguments for the base method.

        Returns:
            torch.Tensor: The expected log likelihood for each observation (shape: (n_samples,)).
        """
        detected = super().expected_log_prob(target, input, *args, **kwargs)
        if not self.censored.any():
            return detected

        sigma = self.noise.sqrt()
        censored = self.quadrature(lambda latent: _log_non_detection((latent - target) / sigma, self.miss_probability), input)
        return torch.where(self.censored, censored, detected)


def train_censored_gp_model(
        x: torch.Tensor,
        y: torch.Tensor,
        bounds: torch.Tensor,
        f: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        miss_probability: float = 0.05,
        observation_sigma: float = 0.25
) -> SingleTaskVariationalGP:
    """
    Fits a censored variational Gaussian Process with inducing points at the distinct training inputs.

    Args:
        x (torch.Tensor): The training inputs (gradient parameters). Shape: (n_samples, n_features).
        y (torch.Tensor): Retention times for one compound. NaN denotes an ambiguous non-detection and +inf denotes
                          known late elution.
        bounds (torch.Tensor): The bounds for the input features. Shape: (2, n_features).
        f (Optional[torch.Tensor]): The training fidelities for multi-fidelity models. Shape: (n_samples, 1).
        t (Optional[torch.Tensor]): The upper bounds for censored regression. Shape: (n_samples,).
        miss_probability (float): Probability of missing a peak that eluted within the run.
        observation_sigma (float): Retention-time observation standard deviation, in minutes.

    Returns:
        SingleTaskVariationalGP: The fitted censored Gaussian Process surrogate model.
    """
    if t is None:
        raise ValueError("Run times are required for censored retention fitting.")

    targets, censored, miss_probabilities, sigma, outcome_transform = prepare_retention_observations(
        y,
        t,
        miss_probability,
        observation_sigma
    )

    covar_module = None
    if f is not None:
        x = torch.cat([x, f], dim=-1)
        bounds = torch.cat([bounds, bounds.new_tensor([[0.0], [1.0]])], dim=-1)
        kernel = LinearTruncatedFidelityKernel(
            fidelity_dims=[x.shape[-1] - 1],
            dimension=x.shape[-1],
            nu=2.5,
            power_prior=GammaPrior(3.0, 3.0)
        )
        covar_module = ScaleKernel(kernel, outputscale_prior=GammaPrior(2.0, 0.15))

    input_transform = Normalize(d=x.shape[-1], bounds=bounds)
    likelihood = CensoredGaussianLikelihood(censored, miss_probabilities, sigma)
    model = SingleTaskVariationalGP(
        x,
        likelihood=likelihood,
        covar_module=covar_module,
        mean_module=ConstantMean(constant_prior=NormalPrior(0.0, 1.0)),
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        inducing_points=input_transform(x).unique(dim=0),
        learn_inducing_points=False
    )
    model.model.train_targets = targets
    mll = VariationalELBO(likelihood, model.model, num_data=x.shape[0])
    fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_scipy, optimizer_kwargs={"options": {"maxiter": 150}})
    # Freeze fitted parameters so acquisition gradients can reuse the GP's prediction caches.
    return model.requires_grad_(False).eval()


def retention_log_prob(
        prediction: torch.Tensor,
        target: torch.Tensor,
        censored: torch.Tensor,
        miss_probability: torch.Tensor,
        sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluates the log likelihood for measured retention times and non-detections.

    Args:
        prediction (torch.Tensor): Predicted retention times.
        target (torch.Tensor): Measured times or run limits, on the same scale as the predictions.
        censored (torch.Tensor): Boolean mask identifying observations represented by a run limit.
        miss_probability (torch.Tensor): Probability of missing a peak that eluted within the run.
        sigma (torch.Tensor): Standard deviation of a retention-time observation.

    Returns:
        torch.Tensor: Log likelihood for each observation.
    """
    z = (prediction - target) / sigma
    log_detected = -0.5 * (z.square() + math.log(2 * math.pi)) - sigma.log()
    return torch.where(censored, _log_non_detection(z, miss_probability), log_detected)


def train_censored_nn_model(
        x: torch.Tensor,
        y: torch.Tensor,
        bounds: torch.Tensor,
        f: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        *,
        fcnn: torch.nn.Sequential,
        miss_probability: float = 0.05,
        observation_sigma: float = 0.25
) -> LastLayerLaplaceModel:
    """
    Fits a neural network to retention observations and applies last-layer Laplace inference.

    Args:
        x (torch.Tensor): The training inputs (gradient parameters). Shape: (n_samples, n_features).
        y (torch.Tensor): Retention times for one compound, including missing and known late peaks.
        bounds (torch.Tensor): The bounds for the input features. Shape: (2, n_features).
        f (Optional[torch.Tensor]): The training fidelities for multi-fidelity models. Shape: (n_samples, 1).
        t (Optional[torch.Tensor]): The upper bounds for censored regression. Shape: (n_samples,).
        fcnn (torch.nn.Sequential): The neural network to fit.
        miss_probability (float): Probability of missing a peak that eluted within the run.
        observation_sigma (float): Retention-time observation standard deviation, in minutes.

    Returns:
        LastLayerLaplaceModel: The fitted censored Laplace surrogate model.
    """
    if f is not None:
        raise NotImplementedError("Multi-fidelity neural network surrogate models are not implemented yet.")
    if t is None:
        raise ValueError("Run times are required for censored retention fitting.")
    targets, censored, miss_probabilities, sigma, outcome_transform = prepare_retention_observations(y, t, miss_probability, observation_sigma)
    input_transform = Normalize(d=x.shape[-1], bounds=bounds)

    return LastLayerLaplaceModel(
        fcnn=fcnn,
        x=x,
        log_prob=lambda pred: retention_log_prob(pred, targets, censored, miss_probabilities, sigma),
        sigma=sigma,
        input_transform=input_transform,
        outcome_transform=outcome_transform
    )
