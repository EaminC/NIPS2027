"""按指定分布给用户打分。分布定义在 abhack.utils.distributions。"""

import numpy as np

from abhack.utils.distributions import Distribution


def draw_metric(n: int, dist: Distribution, seed: int = 0) -> np.ndarray:
    """抽 n 个分数。"""
    return dist.sample(n, seed)


def check_metric(
    values: np.ndarray,
    dist: Distribution,
    seed: int = 0,
    replay: bool = True,
    alpha: float = 0.001,
) -> dict:
    """逆检验：用种子重放，并检验样本是否服从这个分布。"""
    values = np.asarray(values)
    if values.size == 0:
        return {
            "ok": False,
            "errors": ["no scores"],
            "replay_ok": False,
            "stat": float("nan"),
            "pvalue": float("nan"),
            "mean": float("nan"),
            "theory_mean": dist.theory_mean(),
        }
    errors: list[str] = []
    replay_ok = True
    if replay:
        redrawn = dist.sample(values.size, seed)
        if dist.discrete:
            replay_ok = bool(np.array_equal(redrawn, values.astype(np.int64)))
        else:
            rounded = np.array(
                [float(format(float(v), ".17g")) for v in redrawn], dtype=np.float64
            )
            replay_ok = bool(np.array_equal(rounded, values))
        if not replay_ok:
            errors.append("replay does not match the file")
    stat, pvalue = dist.goodness_of_fit(values)
    if pvalue < alpha:
        errors.append(f"fit p={pvalue:.4g}, threshold {alpha:g}")
    return {
        "ok": not errors,
        "errors": errors,
        "replay_ok": replay_ok,
        "stat": stat,
        "pvalue": pvalue,
        "mean": float(np.mean(values)),
        "theory_mean": dist.theory_mean(),
    }
