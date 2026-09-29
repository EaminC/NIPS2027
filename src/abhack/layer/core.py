"""分桶检验。哈希算法在 abhack.utils.hash。"""

import numpy as np

from abhack.utils.hash import hash_buckets as assign_buckets
from abhack.utils.hash import layer_key

__all__ = ["assign_buckets", "check_assignment", "layer_key"]


def check_assignment(
    user_ids: np.ndarray,
    buckets: np.ndarray,
    n_buckets: int,
    layer: str,
) -> list[str]:
    """对照哈希重算一遍。返回错误列表，空列表表示通过。"""
    ids = np.asarray(user_ids, dtype=np.int64)
    got = np.asarray(buckets, dtype=np.int64)
    if ids.shape != got.shape or ids.ndim != 1:
        return ["user_id and bucket have different lengths"]
    expected = assign_buckets(ids, n_buckets, layer)
    errors = []
    n_bad = int(np.count_nonzero(expected != got))
    if n_bad:
        errors.append(f"{n_bad} users have a bucket that does not match a fresh hash")
    if got.size and (np.any(got < 0) or np.any(got >= n_buckets)):
        errors.append(f"bucket ids must be in 0 .. {n_buckets - 1}")
    return errors
