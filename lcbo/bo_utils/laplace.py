import math
from typing import Callable, Optional
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskVariationalGP
from botorch.models.kernels.linear_truncated_fidelity import LinearTruncatedFidelityKernel
from botorch.models.model import Model
from botorch.models.transforms import Normalize, Standardize
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import GammaPrior, NormalPrior
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D


class LastLayerLaplaceModel(Model):
    """
    Fits a neural network at the posterior mode and applies a Laplace approximation to its final layer.

    Args:
        fcnn (torch.nn.Sequential): The neural network to fit. The final layer must have one output.
        x (torch.Tensor): The training inputs. Shape: (n_samples, n_features).
        log_prob (Callable[[torch.Tensor], torch.Tensor]): The observation log likelihood evaluated at the predicted
                                                          standardized targets.
        sigma (torch.Tensor): The observation standard deviation on the standardized target scale.
        input_transform (Normalize): The transform used to normalize the inputs.
        outcome_transform (Standardize): The fitted transform used to restore the original target scale.
    """

    num_outputs = 1
    batch_shape = torch.Size()

    def __init__(
            self,
            fcnn: torch.nn.Sequential,
            x: torch.Tensor,
            log_prob: Callable[[torch.Tensor], torch.Tensor],
            sigma: torch.Tensor,
            input_transform: Normalize,
            outcome_transform: Standardize
    ) -> None:
        super().__init__()
        self.net = fcnn.to(x)
        self.input_transform = input_transform
        self.outcome_transform = outcome_transform
        self.register_buffer("sigma", sigma)

        x = self.input_transform(x)
        self._fit_network(x, log_prob)
        self.net.requires_grad_(False)
        self.register_buffer("precision_cholesky", self._fit_precision(x, log_prob))

    def _fit_network(self, x: torch.Tensor, log_prob: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Fits the network weights at the posterior mode under a unit Gaussian weight prior.

        Args:
            x (torch.Tensor): The normalized training inputs. Shape: (n_samples, n_features).
            log_prob (Callable[[torch.Tensor], torch.Tensor]): The observation log likelihood at the predicted targets.
        """
        optimizer = torch.optim.LBFGS(self.net.parameters(), max_iter=150, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            """
            Evaluates the negative log posterior and updates its gradients.

            Returns:
                torch.Tensor: The negative log posterior at the current weights.
            """
            optimizer.zero_grad()
            predictions = self.net(x).squeeze(-1)
            penalty = sum(parameter.square().sum() for parameter in self.net.parameters()) / 2
            loss = -log_prob(predictions).sum() + penalty
            loss.backward()
            return loss

        optimizer.step(closure)

    def _fit_precision(self, x: torch.Tensor, log_prob: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Computes the precision of the last-layer weights using the observation likelihood curvature.

        Args:
            x (torch.Tensor): The normalized training inputs. Shape: (n_samples, n_features).
            log_prob (Callable[[torch.Tensor], torch.Tensor]): The observation log likelihood at the predicted targets.

        Returns:
            torch.Tensor: The lower Cholesky factor of the last-layer precision matrix.
        """
        predictions = self.net(x).squeeze(-1).detach().requires_grad_(True)
        loss = -log_prob(predictions).sum()
        gradient, = torch.autograd.grad(loss, predictions, create_graph=True)
        curvature, = torch.autograd.grad(gradient.sum(), predictions)

        features = self._last_layer_features(x)
        prior_precision = torch.eye(features.shape[-1], dtype=x.dtype, device=x.device)
        # The censoring mixture can produce negative curvature. Keep its positive part for Laplace.
        weighted_features = curvature.clamp_min(0).unsqueeze(-1) * features
        precision = prior_precision + features.mT @ weighted_features
        return torch.linalg.cholesky(precision)

    def _last_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the hidden activations and appends the bias feature for the final layer.

        Args:
            x (torch.Tensor): The normalized inputs. Shape: (..., n_points, n_features).

        Returns:
            torch.Tensor: The features for the final layer, including the bias.
        """
        hidden = self.net[:-1](x)
        return torch.cat([hidden, torch.ones_like(hidden[..., :1])], dim=-1)

    def posterior(
            self,
            X: torch.Tensor,
            observation_noise: bool = False,
            posterior_transform: Optional[Callable[[GPyTorchPosterior], Posterior]] = None,
            **kwargs,
    ) -> Posterior:
        """
        Evaluates the joint posterior at the given input points on the original target scale.

        Args:
            X (torch.Tensor): The input points at which to evaluate the posterior. Shape: (..., n_points, n_features).
            observation_noise (bool): Whether to include Gaussian observation noise in the posterior.
            posterior_transform (Optional[Callable[[GPyTorchPosterior], Posterior]]): A transform to apply to the
                                                                                  posterior.
            **kwargs: Additional keyword arguments from BoTorch.

        Returns:
            Posterior: The joint posterior over the given input points.
        """
        X = self.input_transform(X)
        features = self._last_layer_features(X)
        projected = torch.linalg.solve_triangular(self.precision_cholesky, features.mT, upper=False)

        diagonal = self.sigma.square() if observation_noise else 1E-8
        covariance = projected.mT @ projected + diagonal * torch.eye(X.shape[-2], dtype=X.dtype, device=X.device)
        distribution = MultivariateNormal(self.net(X).squeeze(-1), covariance)
        posterior = self.outcome_transform.untransform_posterior(GPyTorchPosterior(distribution))
        return posterior_transform(posterior) if posterior_transform is not None else posterior
