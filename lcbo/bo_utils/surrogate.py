from typing import Optional
import numpy as np
import torch

from botorch.models.model import Model
from botorch.models import SingleTaskGP, SingleTaskMultiFidelityGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from laplace_bayesopt.botorch import LaplaceBoTorch

from ..data.config import HPLCOptimizationConfig


class FixedFidelitySurrogate(Model):
    """
    A wrapper class for surrogate models that enforces a fixed fidelity level during posterior predictions. This is
    useful in multi-fidelity optimization scenarios where predictions need to be made at a specific fidelity level, and
    is numerically more stable than using FixedFeatureAcquisitionFunction from BoTorch.
    """
    def __init__(
            self,
            base_model: Model,
            fixed_fidelity: float | torch.Tensor,
            fidelity_index: int
    ):

        super().__init__()
        self.base_model = base_model
        if isinstance(fixed_fidelity, float):
            fixed_fidelity = torch.tensor(fixed_fidelity, dtype=torch.get_default_dtype())
        self.fixed_fidelity = fixed_fidelity
        self.fidelity_index = fidelity_index

    def posterior(self, X: torch.Tensor, **kwargs):
        """
        Generate posterior predictions at the fixed fidelity level.

        Args:
            X (torch.Tensor): Input data for which to generate posterior predictions.
            **kwargs: Additional keyword arguments passed to the base model's posterior method.

        Returns:
            Posterior: The posterior predictions from the base model at the fixed fidelity level.
        """
        fixed_fidelity_column = self.fixed_fidelity.expand(*X.shape[:-1], 1)
        X_fixed = torch.cat(
            [
                X[..., :self.fidelity_index],
                fixed_fidelity_column,
                X[..., self.fidelity_index:]
            ],
            dim=-1
        )
        return self.base_model.posterior(X_fixed, **kwargs)


def train_surrogate_model(
        config: HPLCOptimizationConfig,
        train_x: np.ndarray,
        train_y: np.ndarray,
        train_fidelities: Optional[np.ndarray] = None
) -> Model:
    """
    Train a surrogate model based on the provided training data and optimization configuration.

    Args:
        config (HPLCOptimizationConfig): The optimization configuration containing model settings.
        train_x (np.ndarray): The input features for training the surrogate model.
        train_y (np.ndarray): The target values for training the surrogate model.
        train_fidelities (Optional[np.ndarray], optional): Fidelity levels for multi-fidelity models. Defaults to None.

    Returns:
        Model: The trained surrogate model.
    """
    train_x = torch.tensor(train_x, dtype=torch.get_default_dtype())
    train_y = torch.tensor(train_y, dtype=torch.get_default_dtype())

    if config.single_fidelity:
        train_inputs = train_x
        bounds = config.bounds
    else:
        train_fidelities = torch.tensor(train_fidelities, dtype=torch.get_default_dtype())
        train_inputs = torch.cat([train_x, train_fidelities], dim=-1)
        bounds = torch.cat([config.bounds, torch.tensor([[0.], [1.]])], dim=-1)

    if config.surrogate_model == "GP":
        model = train_gp_surrogate(config, train_inputs, train_y, bounds)
    elif config.surrogate_model in ("NN", "DNN", "WDNN", "NNLT"):
        model = train_nn_surrogate(config, train_inputs, train_y, bounds)
    else:
        raise ValueError(f"Unsupported surrogate model type: {config.surrogate_model}")

    return model


def train_gp_surrogate(
        config: HPLCOptimizationConfig,
        x: torch.Tensor,
        y: torch.Tensor,
        bounds: torch.Tensor
) -> Model:
    """
    Trains a botorch.model.SingleTaskGP model on the input data.

    Args:
        config: HPLCOptimizationConfig: The optimization configuration.
        x: Input data (n_points x n_features).
        y: Output data (n_points or n_points x n_tasks).
        bounds: Bounds for the input data (2 x n_features).
    """
    if config.single_fidelity:
        model = SingleTaskGP(
            x,
            y,
            input_transform=Normalize(d=x.shape[1], bounds=bounds),
            outcome_transform=Standardize(m=y.shape[1])
        )

    else:
        model = SingleTaskMultiFidelityGP(
            x,
            y,
            input_transform=Normalize(d=x.shape[1], bounds=bounds),
            outcome_transform=Standardize(m=y.shape[1]),
            data_fidelities=[x.shape[-1] - 1]
        )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model


def train_nn_surrogate(
        config: HPLCOptimizationConfig,
        x: torch.Tensor,
        y: torch.Tensor,
        bounds: torch.Tensor
) -> Model:
    """
    Trains a laplace_bayesopt.LaplaceBoTorch model on the input data.

    Args:
        config: HPLCOptimizationConfig: The optimization configuration.
        x: Input data (n_points x n_features).
        y: Output data (n_points x n_tasks).
        bounds: Bounds for the input data (2 x n_features).
    """
    if config.multi_fidelity:
        raise NotImplementedError("Multi-fidelity neural network surrogate models are not implemented yet.")

    if config.surrogate_model == "NN":
        layer_sizes = [50]
    elif config.surrogate_model == "DNN":
        layer_sizes = [25, 25]
    else:
        raise ValueError(f"Unsupported surrogate model type: {config.surrogate_model}")

    def nn_generator():
        layers = [torch.nn.Linear(x.shape[1], layer_sizes[0]), torch.nn.ReLU()]
        for i in range(1, len(layer_sizes)):
            layers.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(layer_sizes[-1], y.shape[1]))

        return torch.nn.Sequential(*layers)

    model = LaplaceBoTorch(
        get_net=nn_generator,
        train_X=x,
        train_Y=y,
        input_transform=Normalize(d=x.shape[1], bounds=bounds),
        outcome_transform=Standardize(m=y.shape[1])
    )

    return model
