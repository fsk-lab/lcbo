from collections import namedtuple
import numpy as np
import torch


def convert_to_torch(input: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert a NumPy array to a PyTorch tensor if it is not already a tensor.

    Args:
        input (np.ndarray | torch.Tensor): The input data to convert.

    Returns:
        torch.Tensor: The converted PyTorch tensor.
    """
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif isinstance(input, torch.Tensor):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


def convert_to_numpy(input: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array if it is not already an array.

    Args:
        input (np.ndarray | torch.Tensor): The input data to convert.

    Returns:
        np.ndarray: The converted NumPy array.
    """
    if isinstance(input, torch.Tensor):
        return input.numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


def torch_nanmin(tensor: torch.Tensor, dim: int = 0) -> namedtuple:
    """
    Compute the minimum of a tensor while ignoring NaN values.

    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension along which to compute the minimum. Default is 0.

    Returns:
        namedtuple: A named tuple (values, indices) where 'values' contains the minimum values and 'indices' contains
                    the indices of the minimum values along the specified dimension.
    """
    tensor_safe = torch.where(torch.isnan(tensor), torch.tensor(float('inf'), device=tensor.device), tensor)
    return torch.min(tensor_safe, dim=dim)


def replace_nonfinite(
        input: np.ndarray | torch.Tensor,
        nan_value: float | np.ndarray | torch.Tensor = 0.0,
        inf_value: float | np.ndarray | torch.Tensor = 1.0
) -> np.ndarray | torch.Tensor:
    """
    Replace non-finite values (NaN and Inf) in the input with specified finite values.

    Args:
        input (np.ndarray | torch.Tensor): The input data.
        nan_value (float): The value(s) to replace NaNs with. Default is 0.0.
        inf_value (float): The value to replace Infs with. Default is 1.0.

    Returns:
        np.ndarray | torch.Tensor: The input data with non-finite values replaced.
    """
    if isinstance(input, np.ndarray):
        output = input.copy()

        if isinstance(nan_value, float):
            nan_value = np.full_like(output, nan_value)
        elif isinstance(nan_value, np.ndarray):
            nan_value = np.repeat(nan_value.flatten()[:, np.newaxis], output.shape[1], axis=1)
        else:
            raise TypeError("nan_value must be a float or a NumPy array.")

        if isinstance(inf_value, float):
            inf_value = np.full_like(output, inf_value)
        elif isinstance(inf_value, np.ndarray):
            inf_value = np.repeat(inf_value.flatten()[:, np.newaxis], output.shape[1], axis=1)
        else:
            raise TypeError("inf_value must be a float or a NumPy array.")

        output[np.isnan(output)] = nan_value[np.isnan(output)]
        output[np.isposinf(output)] = inf_value[np.isposinf(output)]
        output[np.isneginf(output)] = -inf_value[np.isneginf(output)]

    elif isinstance(input, torch.Tensor):
        output = input.clone()

        if isinstance(nan_value, float):
            nan_value = torch.full_like(output, nan_value)
        elif isinstance(nan_value, torch.Tensor):
            nan_value = nan_value.unsqueeze(-1).expand_as(output)
        else:
            raise TypeError("nan_value must be a float or a PyTorch tensor.")

        if isinstance(inf_value, float):
            inf_value = torch.full_like(output, inf_value)
        elif isinstance(inf_value, torch.Tensor):
            inf_value = inf_value.unsqueeze(-1).expand_as(output)
        else:
            raise TypeError("inf_value must be a float or a PyTorch tensor.")

        output[torch.isnan(output)] = nan_value[torch.isnan(output)]
        output[torch.isposinf(output)] = inf_value[torch.isposinf(output)]
        output[torch.isneginf(output)] = -inf_value[torch.isneginf(output)]

    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")

    return output
