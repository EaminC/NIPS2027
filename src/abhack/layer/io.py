"""分桶 CSV 读写。两列：user_id,bucket。"""

import os
import warnings
from typing import Iterator

import numpy as np

from abhack.users.io import iter_csv

from .core import assign_buckets, check_assignment


def write_assignment(
    users_path: str,
    path: str,
    layer: str,
    n_buckets: int,
    chunk_size: int = 1_000_000,
) -> int:
    """按用户文件的行顺序分桶并写入。返回人数。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        f.write("user_id,bucket\n")
        for ids in iter_csv(users_path, chunk_size):
            buckets = assign_buckets(ids, n_buckets, layer)
            np.savetxt(f, np.column_stack([ids, buckets]), fmt="%d,%d")
            n += int(ids.size)
    return n


def iter_assignment(
    path: str, chunk_size: int = 1_000_000
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """按块读 user_id 和 bucket。"""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    with open(path, "r", newline="") as f:
        header = f.readline().strip()
        if header != "user_id,bucket":
            raise ValueError(f"header must be user_id,bucket, got {header!r}")
        while True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                chunk = np.loadtxt(
                    f, delimiter=",", dtype=np.int64, max_rows=chunk_size, ndmin=2
                )
            if chunk.size == 0:
                break
            yield chunk[:, 0], chunk[:, 1]
            if chunk.shape[0] < chunk_size:
                break


def count_assignment(
    path: str, n_buckets: int, chunk_size: int = 1_000_000
) -> tuple[np.ndarray, int]:
    """统计每个桶的人数。越界桶号不计入条形图。"""
    counts = np.zeros(n_buckets, dtype=np.int64)
    total = 0
    for ids, buckets in iter_assignment(path, chunk_size):
        total += int(ids.size)
        if not buckets.size:
            continue
        valid = (buckets >= 0) & (buckets < n_buckets)
        if np.any(valid):
            counts += np.bincount(buckets[valid], minlength=n_buckets)
    return counts, total


def check_assignment_file(
    users_path: str,
    assignment_path: str,
    layer: str,
    n_buckets: int,
    chunk_size: int = 1_000_000,
) -> tuple[list[str], np.ndarray, int]:
    """逐块对照用户文件和分桶文件。返回错误、各桶计数、人数。"""
    counts = np.zeros(n_buckets, dtype=np.int64)
    errors: list[str] = []
    n = 0
    users = iter_csv(users_path, chunk_size)
    assigned = iter_assignment(assignment_path, chunk_size)
    while True:
        ids = next(users, None)
        pair = next(assigned, None)
        if ids is None and pair is None:
            break
        if ids is None or pair is None:
            errors.append("user file and bucket file have different lengths")
            break
        got_ids, buckets = pair
        if ids.shape != got_ids.shape or not np.array_equal(ids, got_ids):
            errors.append("user_id does not match the user file")
            break
        errors.extend(check_assignment(ids, buckets, n_buckets, layer))
        if buckets.size and np.all((buckets >= 0) & (buckets < n_buckets)):
            counts += np.bincount(buckets, minlength=n_buckets)
        n += int(ids.size)
        if errors:
            break
    return errors, counts, n
