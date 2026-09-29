"""三种过程：静止、每轮重抽、AR(1)。"""

import numpy as np

from abhack.utils.distributions import Distribution


def advance(
    scores: np.ndarray,
    dist: Distribution,
    kind: str,
    rng: np.random.Generator,
    phi: float = 0.0,
    shock: float = 0.0,
) -> np.ndarray:
    """从当前分数走出一轮。static 原样返回。"""
    scores = np.asarray(scores, dtype=float)
    if kind == "static":
        return scores
    if kind == "redraw":
        return np.asarray(dist.draw(rng, scores.size), dtype=float)
    if kind == "ar1":
        if dist.discrete:
            raise ValueError("ar1 does not apply to a discrete score; use static or redraw")
        if not -1.0 <= phi <= 1.0:
            raise ValueError("phi must be between -1 and 1")
        if shock < 0:
            raise ValueError("shock must be >= 0")
        mu = dist.theory_mean()
        noise = rng.normal(0.0, shock, size=scores.size)
        return mu + phi * (scores - mu) + noise
    raise ValueError("process must be static, redraw, or ar1")
