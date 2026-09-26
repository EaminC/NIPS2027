"""Temporal models for S(i).

A process maps the current population vector to the next one. ``static`` is
the iid-over-time special case. Register a new process with
``@processes.register("name")``.
"""

from __future__ import annotations

import numpy as np

from ab_aa_lab.registry import processes


class Process:
    name = "process"

    def step(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@processes.register("static")
class Static(Process):
    def step(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        del rng
        return values


@processes.register("ar1")
class AR1(Process):
    """x <- mu + phi * (x - mu) + eps.

    ``phi = 1`` is a random walk. ``|phi| < 1`` mean-reverts, which is the
    mechanism that makes a frozen group's AB gap decay after launch.
    """

    def __init__(self, phi: float = 0.9, sigma: float = 0.1, mu: float = 0.0) -> None:
        if not -1 <= phi <= 1:
            raise ValueError("phi must be in [-1, 1]")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.phi = float(phi)
        self.sigma = float(sigma)
        self.mu = float(mu)

    def step(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(0.0, self.sigma, size=values.shape)
        return self.mu + self.phi * (values - self.mu) + noise


@processes.register("random_walk")
class RandomWalk(Process):
    def __init__(self, sigma: float = 0.1, drift: float = 0.0) -> None:
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.sigma = float(sigma)
        self.drift = float(drift)

    def step(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return values + self.drift + rng.normal(0.0, self.sigma, size=values.shape)
