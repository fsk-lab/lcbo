from typing import Optional, Literal
import numpy as np
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelList, SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.models.model import Model
from botorch.models.transforms import Normalize, Standardize
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..data.config import HPLCOptimizationConfig
from .laplace import LastLayerLaplaceModel
from .censoring import train_censored_gp_model, train_censored_nn_model


class FixedFidelitySurrogate(Model):
    """
    Exposes a multi-fidelity surrogate at a fixed fidelity for acquisition optimization. This is more efficient than
    botorch's `FixedFidelityAcquisitionFunction`.

    Args:
        base_model (Model): The multi-fidelity surrogate model to wrap.
        fixed_fidelity (float | torch.Tensor): The fidelity value to fix for acquisition optimization.
        fidelity_index (int): The index of the fidelity dimension in the input space.
    """

    def __init__(self, base_model: Model, fixed_fidelity: float | torch.Tensor, fidelity_index: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("fixed_fidelity", torch.as_tensor(fixed_fidelity))
        self.fidelity_index = fidelity_index

    @property
    def num_outputs(self) -> int:
        """
        Returns the number of outputs of the wrapped surrogate.
        """
        return self.base_model.num_outputs

    def posterior(self, X: torch.Tensor, **kwargs) -> Posterior:
        """
        Evaluates the posterior of the wrapped surrogate at the fixed fidelity for the given input points.

        Args:
            X (torch.Tensor): The input points at which to evaluate the posterior. Shape: (batch_size, n_features).
            **kwargs: Additional keyword arguments to pass to the base model's posterior method.
        """
        fidelity = self.fixed_fidelity.to(X).expand(*X.shape[:-1], 1)
        X_with_fid = torch.cat([X[..., :self.fidelity_index], fidelity, X[..., self.fidelity_index:]], dim=-1)
        return self.base_model.posterior(X_with_fid, **kwargs)


class DefaultValueModel(Model):
    """
    A surrogate model that returns a fixed default value for all inputs. This is used when no finite observations are
    available for a particular output, but a default value is desired.

    Args:
        default_value (float): The default value to return for all inputs.
    """

    num_outputs = 1
    batch_shape = torch.Size()

    def __init__(self, default_value: float) -> None:
        super().__init__()
        self.register_buffer("default_value", torch.as_tensor(default_value))

    def posterior(self, X: torch.Tensor, **kwargs) -> GPyTorchPosterior:
        """
        Returns a posterior with a fixed mean and a small covariance for the given input points.
        """
        mean = X.new_full(X.shape[:-1], self.default_value)
        covariance = 1E-8 * torch.eye(X.shape[-2], dtype=X.dtype, device=X.device).expand(*mean.shape[:-1], -1, -1)
        return GPyTorchPosterior(MultivariateNormal(mean, covariance))


def train_surrogate_model(
        config: HPLCOptimizationConfig,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_fidelities: Optional[np.ndarray] = None,
        retention_run_times: Optional[np.ndarray] = None,
        default_value: Optional[float] = None,
) -> Model:
    """
    Fit a surrogate to the campaign inputs and targets.

    Args:
        config (HPLCOptimizationConfig): The configuration for the HPLC optimization.
        train_x (np.ndarray): The training inputs (gradient parameters). Shape: (n_samples, n_features).
        train_y (np.ndarray): The training targets (objective values). Shape: (n_samples, n_outputs).
        train_fidelities (Optional[np.ndarray]): The training fidelities for multi-fidelity models. Shape:
                                                 (n_samples, 1). Required if config.multi_fidelity is True.
        retention_run_times (Optional[np.ndarray]): The run times for censored regression of retention times. Shape:
                                                    (n_samples,) or (n_samples, 1).
        default_value (Optional[float]): The default value to use for outputs with no finite observations. If None,
                                         the model will raise an error for outputs with no finite observations.
    """
    x = torch.tensor(train_x, dtype=torch.get_default_dtype())
    y = torch.tensor(train_y, dtype=torch.get_default_dtype())
    bounds = config.bounds
    f = None
    if config.multi_fidelity:
        if train_fidelities is None:
            raise ValueError("Training fidelities are required for multi-fidelity models.")
        f = torch.tensor(train_fidelities, dtype=torch.get_default_dtype()).reshape(-1, 1)
    t = x.new_tensor(retention_run_times).reshape(-1) if retention_run_times is not None else None

    retention_settings = {"miss_probability": config.retention_miss_probability, "observation_sigma": config.retention_observation_sigma}
    if config.surrogate_model == "GP":
        return train_gp_surrogate(x, y, bounds, f, t, default_value=default_value, **retention_settings)
    if config.surrogate_model in ("NN", "DNN"):
        return train_nn_surrogate(x, y, bounds, f, t, architecture=config.surrogate_model, default_value=default_value, **retention_settings)
    raise ValueError(f"Unsupported surrogate model type: {config.surrogate_model}")


def train_gp_surrogate(
        x: torch.Tensor,
        y: torch.Tensor,
        bounds: torch.Tensor,
        f: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        default_value: Optional[float] = None,
        miss_probability: float = 0.05,
        observation_sigma: float = 0.25,
) -> Model:
    """
    Fits a Gaussian Process surrogate model to the provided training data.

    Args:
        x (torch.Tensor): The training inputs (gradient parameters). Shape: (n_samples, n_features).
        y (torch.Tensor): The training targets (objective values). Shape: (n_samples, n_outputs).
        bounds (torch.Tensor): The bounds for the input features. Shape: (2, n_features).
        f (Optional[torch.Tensor]): The training fidelities for multi-fidelity models. Shape: (n_samples, 1).
        t (Optional[torch.Tensor]): The upper bounds for censored regression. Shape: (n_samples,).
        default_value (Optional[float]): The default value to use for outputs with no finite observations. If None,
                                         the model will raise an error for outputs with no finite observations.
        miss_probability (float): Probability of missing a peak that eluted within the run.
        observation_sigma (float): Retention-time observation standard deviation, in minutes.
    """
    models = []
    for output_idx in range(y.shape[-1]):
        values = y[:, output_idx]
        mask = ~torch.isneginf(values) if t is not None else torch.isfinite(values)
        if not mask.any():
            if default_value is not None:
                models.append(DefaultValueModel(default_value))
                continue
            raise ValueError(f"No finite observations for model output {output_idx}.")
        x_, y_ = x[mask], y[mask, output_idx, None]
        f_ = f[mask] if f is not None else None

        if t is not None:
            t_ = t[mask]
            model = train_censored_gp_model(x_, y_, bounds, f_, t_, miss_probability, observation_sigma)
        else:
            model = train_gp_model(x_, y_, bounds, f_)
        models.append(model)

    return ModelList(*models)


def train_gp_model(
        x: torch.Tensor,
        y: torch.Tensor,
        bounds: torch.Tensor,
        f: Optional[torch.Tensor] = None
) -> Model:
    """
    Fits a Gaussian Process surrogate model to the provided training data using BoTorch's SingleTaskGP or
    SingleTaskMultiFidelityGP.

    Args:
        x (torch.Tensor): The training inputs (gradient parameters). Shape: (n_samples, n_features).
        y (torch.Tensor): The training targets (objective values). Shape: (n_samples, n_outputs).
        bounds (torch.Tensor): The bounds for the input features. Shape: (2, n_features).
        f (Optional[torch.Tensor]): The training fidelities for multi-fidelity models. Shape: (n_samples, 1).

    Returns:
        Model: The fitted Gaussian Process surrogate model.
    """
    if f is not None:
        x = torch.cat([x, f], dim=-1)
        bounds = torch.cat([bounds, bounds.new_tensor([[0.], [1.]])], dim=-1)
        input_transform = Normalize(d=x.shape[-1], bounds=bounds)
        outcome_transform = Standardize(m=y.shape[-1])

        model = SingleTaskMultiFidelityGP(
            x,
            y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            data_fidelities=[x.shape[-1] - 1]
        )

    else:
        input_transform = Normalize(d=x.shape[-1], bounds=bounds)
        outcome_transform = Standardize(m=y.shape[-1])
        model = SingleTaskGP(x, y, input_transform=input_transform, outcome_transform=outcome_transform)

    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def train_nn_surrogate(
        x: torch.Tensor,
        y: torch.Tensor,
        bounds: torch.Tensor,
        f: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        architecture: Literal["NN", "DNN"] = "NN",
        default_value: Optional[float] = None,
        miss_probability: float = 0.05,
        observation_sigma: float = 0.25,
) -> Model:
    """
    Fits a last-layer Laplace surrogate for each output. Gaussian regression uses finite targets, and censored
    regression also uses non-detections and known late peaks.

    Args:
        x (torch.Tensor): The training inputs (gradient parameters). Shape: (n_samples, n_features).
        y (torch.Tensor): The training targets (objective values). Shape: (n_samples, n_outputs).
        bounds (torch.Tensor): The bounds for the input features. Shape: (2, n_features).
        f (Optional[torch.Tensor]): The training fidelities for multi-fidelity models. Shape: (n_samples, 1).
        t (Optional[torch.Tensor]): The upper bounds for censored regression. Shape: (n_samples,).
        architecture (Literal["NN", "DNN"]): The neural network architecture (NN or DNN).
        default_value (Optional[float]): The default value to use for outputs with no finite observations. If None,
                                         the model will raise an error for outputs with no finite observations.
        miss_probability (float): Probability of missing a peak that eluted within the run.
        observation_sigma (float): Retention-time observation standard deviation, in minutes.

    Returns:
        Model: The fitted Laplace neural-network surrogate model.
    """
    if f is not None:
        raise NotImplementedError("Multi-fidelity neural network surrogate models are not implemented yet.")

    models = []
    for output_idx in range(y.shape[-1]):
        values = y[:, output_idx]
        mask = ~torch.isneginf(values) if t is not None else torch.isfinite(values)
        if not mask.any():
            if default_value is not None:
                models.append(DefaultValueModel(default_value))
                continue
            raise ValueError(f"No finite observations for model output {output_idx}.")
        x_, y_ = x[mask], y[mask, output_idx, None]

        if architecture == "NN":
            layer_sizes = [50]
        elif architecture == "DNN":
            layer_sizes = [25, 25]
        else:
            raise ValueError(f"Unsupported surrogate model type: {architecture}")

        layers = []
        n = x.shape[-1]
        for width in layer_sizes:
            layers.extend([torch.nn.Linear(n, width), torch.nn.ReLU()])
            n = width
        layers.append(torch.nn.Linear(n, 1))
        network = torch.nn.Sequential(*layers)

        if t is not None:
            model = train_censored_nn_model(
                x=x_,
                y=y_,
                bounds=bounds,
                t=t[mask],
                fcnn=network,
                miss_probability=miss_probability,
                observation_sigma=observation_sigma
            )
        else:
            model = train_nn_model(
                x=x_,
                y=y_,
                bounds=bounds,
                fcnn=network,
                observation_sigma=observation_sigma
            )
        models.append(model)

    return ModelList(*models)


def train_nn_model(
        x: torch.Tensor,
        y: torch.Tensor,
        bounds: torch.Tensor,
        fcnn: torch.nn.Sequential,
        observation_sigma: float = 0.25
) -> Model:
    """
    Fits a neural network with Gaussian observation noise and a last-layer Laplace approximation.

    Args:
        x (torch.Tensor): The training inputs (gradient parameters). Shape: (n_samples, n_features).
        y (torch.Tensor): The training targets (objective values). Shape: (n_samples, n_outputs).
        bounds (torch.Tensor): The bounds for the input features. Shape: (2, n_features).
        fcnn (torch.nn.Sequential): The neural network to fit. The final layer must have one output.
        observation_sigma (float): Fixed Gaussian observation standard deviation on the standardized target scale.

    Returns:
        Model: The fitted last-layer Laplace model, or a list of independent models for multiple outputs.
    """
    input_transform = Normalize(d=x.shape[-1], bounds=bounds)
    outcome_transform = Standardize(m=1).to(y)
    targets, _ = outcome_transform(y)
    outcome_transform.eval()
    sigma = y.new_tensor(observation_sigma)
    likelihood = torch.distributions.Normal(targets.squeeze(-1), sigma)

    return LastLayerLaplaceModel(fcnn, x, likelihood.log_prob, sigma, input_transform, outcome_transform)
