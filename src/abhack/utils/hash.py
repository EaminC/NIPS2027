"""层内固定哈希。

同一个 user_id 和层名永远进同一个桶。不用 Python 内置 hash，那个值每个进程不一样。
"""

import numpy as np

_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_MASK64 = (1 << 64) - 1
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def layer_key(layer: str) -> np.uint64:
    """把层名收成固定的 64 位整数。乘法按 2^64 回绕。"""
    if layer == "":
        raise ValueError("layer name must not be empty")
    digest = _FNV_OFFSET
    for byte in layer.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _MASK64
    return np.uint64(digest)


def hash_buckets(user_ids: np.ndarray, n_buckets: int, layer: str) -> np.ndarray:
    """返回与 user_ids 等长的桶号，范围是 0 .. n_buckets-1。"""
    if n_buckets < 2:
        raise ValueError("bucket count must be at least 2")
    ids = np.asarray(user_ids)
    if ids.ndim != 1:
        raise ValueError("user_id must be a 1-d array")
    if ids.size and np.any(ids < 0):
        raise ValueError("user_id must be non-negative")
    # Stafford mix。只做一次乘法移位时，相邻 user_id 会大量留在同一个桶。
    mixed = ids.astype(np.uint64) ^ layer_key(layer)
    mixed = mixed + _GOLDEN
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return (mixed % np.uint64(n_buckets)).astype(np.int64)
