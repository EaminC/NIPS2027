"""生成 user_id。"""

from typing import Iterator

import numpy as np


def generate_users(M: int) -> np.ndarray:
    """返回 0..M-1 的 user_id 数组。"""
    if M < 0:
        raise ValueError("M must be >= 0")
    return np.arange(M, dtype=np.int64)


def iter_chunks(M: int, chunk_size: int = 1_000_000) -> Iterator[np.ndarray]:
    """按块生成 user_id，避免 M 很大时爆内存。"""
    if M < 0:
        raise ValueError("M must be >= 0")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    for start in range(0, M, chunk_size):
        yield np.arange(start, min(start + chunk_size, M), dtype=np.int64)
