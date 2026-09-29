"""用户 CSV 读写。文件只有一列 user_id。"""

import os
import warnings
from typing import Iterator

import numpy as np

from .core import iter_chunks


def write_csv(users, path: str, chunk_size: int = 1_000_000) -> None:
    """把用户写入 CSV。

    users 可以是人数 M，也可以是一列 user_id。
    传入 M 时按块生成，不必先在内存里放下全部 id。
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if isinstance(users, (int, np.integer)):
        chunks: Iterator[np.ndarray] = iter_chunks(int(users), chunk_size)
    else:
        ids = np.asarray(users, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("user_id must be a 1-d array")
        chunks = (
            ids[start : start + chunk_size] for start in range(0, ids.size, chunk_size)
        )
    _ensure_parent(path)
    with open(path, "w", newline="") as f:
        f.write("user_id\n")
        for ids in chunks:
            if ids.size:
                np.savetxt(f, ids, fmt="%d")


def iter_csv(path: str, chunk_size: int = 1_000_000) -> Iterator[np.ndarray]:
    """按块读 user_id。"""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    with open(path, "r", newline="") as f:
        header = f.readline().strip()
        if header != "user_id":
            raise ValueError(f"header must be user_id, got {header!r}")
        while True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                chunk = np.loadtxt(f, dtype=np.int64, max_rows=chunk_size)
            if np.size(chunk) == 0:
                break
            chunk = np.atleast_1d(chunk)
            yield chunk
            if chunk.size < chunk_size:
                break


def read_csv(path: str) -> np.ndarray:
    """读入全部 user_id。"""
    parts = list(iter_csv(path))
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(parts)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
