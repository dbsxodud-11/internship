import math
from typing import List, Tuple, Optional

import torch
from torch import Tensor
import pandas as pd

from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.transforms import unnormalize


class Ackley(SyntheticTestFunction):
    r"""Ackley test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
            exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        self._bounds = [(-32.768, 32.768) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)
        self.a = 40
        self.b = 0.05
        self.c = math.pi
        self.d = torch.FloatTensor([0.7, 0.6, 0.2, 0.8, 0.4]).unsqueeze(0) * 65.592 - 32.768

    def evaluate_true(self, X: Tensor) -> Tensor:
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.norm((X - self.d), dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * (X - self.d)), dim=-1)))
        return part1 + part2 + a + math.e
    

class GroundTruth:
    def __init__(self, dim=5):
        self.function = Ackley(dim=dim, negate=True).to(dtype=torch.float32)
        

    def __call__(self, x):
        return self.function(unnormalize(x, self.function.bounds)).unsqueeze(-1) + 100.0


if __name__ == "__main__":
    x = torch.rand(size=(10000, 5))
    func = GroundTruth()
    y = func(x)
    
    x = x[y.flatten() <= 90]
    y = y[y.flatten() <= 90]
    
    df = pd.DataFrame(torch.cat([x, y], dim=-1), columns=["A", "B", "C", "D", "E", "score"])
    df.to_csv("data/log.csv", index=False)
