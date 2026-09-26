"""Cross-sectional distributions for the initial metric S(i).

Register a new distribution with ``@metrics.register("name")``. ``initial``
must return a one-dimensional float array of length ``n`` and must draw only
from the given Generator.
"""

from __future__ import annotations

import numpy as np

from ab_aa_lab.registry import metrics


class Metric:
    name = "metric"

    def initial(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@metrics.register("normal")
class Normal(Metric):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.mu = float(mu)
        self.sigma = float(sigma)

    def initial(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mu, self.sigma, size=n)


@metrics.register("lognormal")
class LogNormal(Metric):
    def __init__(self, mu: float = 0.0, sigma: float = 0.5) -> None:
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.mu = float(mu)
        self.sigma = float(sigma)

    def initial(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma, size=n)


@metrics.register("uniform")
class Uniform(Metric):
    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        if high < low:
            raise ValueError("high must be >= low")
        self.low = float(low)
        self.high = float(high)

    def initial(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=n)
