from typing import Optional
import torch
from botorch.utils.sampling import HitAndRunPolytopeSampler
from botorch.utils.sampling import normalize_linear_constraints, sparse_to_dense_constraints


def sample_from_polytope(
        bounds: torch.Tensor,
        inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]],
        n_samples: int,
        thinning: int = 32,
        seed: Optional[int] = None
) -> torch.Tensor:
    """
    Samples from a polytope defined by bounds and inequality constraints. This gives more representative samples than
    randomly sampling from the bounds.

    Uses a HitAndRunPolytopeSampler, as implemented in botorch.utils.sampling.

    Args:
        bounds: Bounds for the input data (2 x n_features).
        inequality_constraints: A list of inequality constraints in the form of (indices, coefficients, value).
        n_samples: The number of samples to draw.
        thinning: The thinning factor to apply to the samples.
        seed: The random seed to use.

    Returns:
        A tensor of samples from the polytope (n_samples x n_features).
    """
    # normalize bounds
    normalized_bounds = torch.stack([torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])])

    # normalize constraints and convert to dense form
    if inequality_constraints:
        constraints = normalize_linear_constraints(bounds, inequality_constraints)
        a, b = sparse_to_dense_constraints(constraints=constraints, d=bounds.shape[-1])
        dense_constraints = -a, -b
    else:
        dense_constraints = None

    # instantiate sampler and draw samples
    sampler = HitAndRunPolytopeSampler(
        inequality_constraints=dense_constraints,
        bounds=normalized_bounds,
        n_burnin=1024
    )
    samples = sampler.draw(n=n_samples*thinning, seed=seed)[::thinning]

    return bounds[0] + samples * (bounds[1] - bounds[0])
