from uuid import uuid4

import numpy as np
from scipy.optimize import LinearConstraint
import torch
from moses.datatypes import LCGradient


class LCGradientParameters(object):
    """
    A class to handle the conversion between MoSeS LCGradient objects and their parameter representations.

    The parameter representation is a numpy array that can take two possible forms:
        - `segment-based`: [dt_1, dt_2, ..., dt_n, f_0, f_1, ..., f_n], where dt_i are the durations of each segment
          and f_i are the fractions of organic eluent at these time points.
        - `time-based`: [t_1, t_2, ..., t_n, f_0, f_1, ..., f_n], where t_i are the absolute end times of each segment
          (excluding t_0 = 0) and f_i are the fractions of organic eluent at these time points.

    Thus, for a gradient with n linear segments, the parameter array has a shape of `2n + 1`.

    Args:
        parameters (np.ndarray): A numpy array of parameter values (shape: `2n + 1`).
        segment_based (bool): If True, the parameters are interpreted as segment durations. If False, they are
                              interpreted as absolute end times.


    """
    def __init__(self, parameters: np.ndarray, segment_based: bool = False):
        self.parameters = parameters
        self.segment_based = segment_based

    @property
    def num_segments(self) -> int:
        """
        Returns the number of linear segments in the gradient.

        Returns:
            int: The number of segments.
        """
        return (len(self.parameters) - 1) // 2

    def to_gradient(
            self,
            flow_rate: float,
            eluents: tuple[str, str] = ("A", "B"),
            equilibration_time: float = 0.0,
            tag: str = None
    ) -> LCGradient:
        """
        Convert the parameter representation to an LCGradient object.

        Args:
            flow_rate (float): The flow rate of the HPLC method.
            eluents (tuple[str, str]): A tuple containing the names of the aqueous and organic eluents.
            equilibration_time (float): The equilibration time before the gradient starts.
            tag (str, optional): An optional name tag for the gradient object.

        Returns:
            LCGradient: The gradient object constructed from the parameter values.
        """
        if self.segment_based:
            time_steps = np.cumsum(self.parameters[:self.num_segments]).round(decimals=4).tolist()
        else:
            time_steps = self.parameters[:self.num_segments].round(decimals=4).tolist()

        fractions = self.parameters[self.num_segments:].tolist()

        return LCGradient(
            name=tag or str(uuid4()),
            time_steps=[0.0] + time_steps,
            proportions_b=fractions,
            flow_rate=flow_rate,
            eluents=eluents,
            equilibration_time=equilibration_time
        )

    @classmethod
    def from_gradient(cls, gradient: LCGradient, segment_based: bool = False) -> "LCGradientParameters":
        """
        Create an LCGradientParameters object from an LCGradient object.

        Args:
            gradient (LCGradient): The gradient object containing parameter values.
            segment_based (bool): If True, the parameters are interpreted as segment durations. If False, they are
                                  interpreted as absolute end times.

        Returns:
            LCGradientParameters: An instance of LCGradientParameters containing the parameter representation.
        """
        gradient_table = gradient.gradient_table
        time_steps = gradient_table["time"].tolist()
        fractions = gradient_table[gradient.eluent_b].tolist()

        if segment_based:
            durations = np.diff(time_steps).round(decimals=4).tolist()
            parameters = np.array(durations + fractions)
        else:
            parameters = np.array(time_steps[1:] + fractions)

        return cls(parameters=parameters, segment_based=segment_based)


class LCGradientConstraints(object):
    """
    A class to handle the bounds and constraints for optimizing multilinear HPLC gradients. The gradients can be
    represented in either a segment-based or time-based format.
      - `segment-based`: [dt_1, dt_2, ..., dt_n, f_0, f_1, ..., f_n], where dt_i are the durations of each segment
         and f_i are the fractions of organic eluent at these time points. In this format, each dt_i and their sum
         must be less than or equal to the maximum run time. In addition, f_i <= f_(i+1) must hold.
      - `time-based`: [t_1, t_2, ..., t_n, f_0, f_1, ..., f_n], where t_i are the absolute end times of each segment
         (excluding t_0 = 0) and f_i are the fractions of organic eluent at these time points. In this format, each t_i
         must be less than or equal to the maximum run time, and t_i < t_(i+1). In addition, f_i <= f_(i+1) must hold.


    Args:
        num_segments (int): The number of linear segments in the gradient.
        min_segment_length (float): The minimum allowed length for each segment.
        max_run_time (float): The maximum allowed run time for the gradient.
        segment_based (bool): If True, the parameters are interpreted as segment durations. If False, they are
                              interpreted as absolute end times.
    """

    def __init__(
            self,
            num_segments: int,
            min_segment_length,
            max_run_time: float,
            segment_based: bool = False
    ):

        if num_segments * min_segment_length > max_run_time:
            raise ValueError("The product of num_segments and min_segment_length cannot exceed max_run_time.")

        self.num_segments = num_segments
        self.min_segment_length = min_segment_length
        self.max_run_time = max_run_time
        self.segment_based = segment_based

        self.bounds = self._get_bounds()
        self.constraints = self._get_constraints()

    def _get_bounds(self) -> torch.Tensor:
        """
        Defines the bounds for a `num_segments`-sized multilinear HPLC gradient. Returns bounds in the format
        required by BoTorch's optimization routines.
        """
        bounds = [
            [self.min_segment_length] * self.num_segments + [0.0] * (self.num_segments + 1),  # lower bounds
            [self.max_run_time] * self.num_segments + [1.0] * (self.num_segments + 1)  # upper bounds
        ]
        return torch.tensor(bounds, dtype=torch.get_default_dtype())

    def _get_constraints(self) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """
        Defines the constraints for a `num_segments`-sized multilinear HPLC gradient. Returns constraints in the format
        required by BoTorch's optimization routines.
        """
        time_constraints, fraction_constraints = [], []

        # Setup Time Constraints
        if not self.segment_based:
            for idx in range(self.num_segments - 1):
                time_constraints.append(
                    (
                        torch.tensor([idx, idx + 1]),
                        torch.tensor([-1., 1.], dtype=torch.get_default_dtype()),
                        self.min_segment_length
                    )
                )  # t_i + min_segment_length < t_(i+1)
        else:
            time_constraints.append(
                (
                    torch.arange(self.num_segments),
                    -1. * torch.ones(self.num_segments, dtype=torch.get_default_dtype()),
                    -self.max_run_time
                )
            )  # sum(dt_i) <= max_run_time

        # Setup Fraction Constraints
        for idx in range(self.num_segments - 1):
            fraction_constraints.append(
                (
                    torch.tensor([self.num_segments + idx, self.num_segments + idx + 1]),
                    torch.tensor([-1., 1.], dtype=torch.get_default_dtype()), 0.0)
            )  # f_i <= f_(i+1)
        fraction_constraints.append(
            (
                torch.tensor([2 * self.num_segments - 1, 2 * self.num_segments]),
                torch.tensor([-1., 1.], dtype=torch.get_default_dtype()), 0.0)
        )  # f_(n-1) <= f_n

        return time_constraints + fraction_constraints

    @property
    def bounds_scipy(self) -> np.ndarray:
        """
        Returns the bounds in a format compatible with SciPy's optimization routines.
        """
        return self.bounds.T.numpy()

    @property
    def constraints_scipy(self) -> LinearConstraint:
        """
        Returns the constraints in a format compatible with SciPy's optimization routines.
        """
        matrix, rhs = [], []
        for indices, coefficients, value in self.constraints:
            row = np.zeros(self.bounds.shape[1])
            row[indices.numpy()] = coefficients.numpy()
            matrix.append(row)
            rhs.append(value)
        return LinearConstraint(np.array(matrix), np.array(rhs), np.inf)
