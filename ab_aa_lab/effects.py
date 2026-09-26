"""Measurement-time interventions.

Effects do not write back into the latent metric, and strategies do not see
them. They answer a separate question: if the launched treatment bucket also
carried a real parameter change, how would the published AB and AA move.

The default ``null`` effect is the pure selection experiment.
"""

from __future__ import annotations

import numpy as np

from ab_aa_lab.registry import effects
from ab_aa_lab.types import Claim


class Effect:
    name = "effect"

    def apply(
        self,
        values: np.ndarray,
        assignment: np.ndarray,
        claim: Claim,
    ) -> np.ndarray:
        raise NotImplementedError


@effects.register("null")
class NullEffect(Effect):
    def apply(
        self,
        values: np.ndarray,
        assignment: np.ndarray,
        claim: Claim,
    ) -> np.ndarray:
        del assignment, claim
        return values


@effects.register("constant_lift")
class ConstantLift(Effect):
    """Add ``delta`` to every user in the claimed treatment bucket."""

    def __init__(self, delta: float = 0.0) -> None:
        self.delta = float(delta)

    def apply(
        self,
        values: np.ndarray,
        assignment: np.ndarray,
        claim: Claim,
    ) -> np.ndarray:
        out = np.array(values, dtype=float, copy=True)
        out[assignment == claim.treatment_bucket] += self.delta
        return out
