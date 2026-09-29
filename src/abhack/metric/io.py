"""分数 CSV 读写。两列：user_id 和调用者指定的分数名。"""

import os
import warnings

import numpy as np

from abhack.users.io import iter_csv
from abhack.utils.distributions import Distribution

from .core import check_metric


def write_metric(
    users_path: str,
    path: str,
    name: str,
    dist: Distribution,
    seed: int = 0,
    chunk_size: int = 1_000_000,
) -> int:
    """按用户文件的行顺序打分。随机数沿行顺序消耗。返回人数。"""
    if name == "" or name == "user_id" or any(char in name for char in ",\n\r"):
        raise ValueError("score name must be non-empty, must not be user_id, and must not contain a comma or newline")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = 0
    with open(path, "w", newline="") as f:
        f.write(f"user_id,{name}\n")
        for ids in iter_csv(users_path, chunk_size):
            values = dist.draw(rng, int(ids.size))
            if not dist.discrete:
                lines = [
                    f"{int(uid)},{format(float(value), '.17g')}"
                    for uid, value in zip(ids.tolist(), values.tolist())
                ]
            else:
                lines = [
                    f"{int(uid)},{int(value)}"
                    for uid, value in zip(ids.tolist(), values.tolist())
                ]
            if lines:
                f.write("\n".join(lines) + "\n")
            n += int(ids.size)
    return n


def read_metric(path: str) -> tuple[np.ndarray, np.ndarray, str]:
    """返回 user_id、分数、分数名。"""
    with open(path, "r", newline="") as f:
        header = f.readline().strip().split(",")
        if len(header) != 2 or header[0] != "user_id" or header[1] == "":
            raise ValueError(f"header must be user_id,<score name>, got {header}")
        name = header[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            raw = np.loadtxt(f, delimiter=",", dtype=str)
    if raw.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), name
    raw = np.atleast_2d(raw)
    user_ids = raw[:, 0].astype(np.int64)
    values = raw[:, 1].astype(np.float64)
    return user_ids, values, name


def check_metric_file(
    path: str,
    dist: Distribution,
    seed: int = 0,
    replay: bool = True,
    alpha: float = 0.001,
    name: str | None = None,
) -> tuple[dict, str]:
    """读分数 CSV 并做逆检验。"""
    _ids, values, found = read_metric(path)
    if name is not None and name != found:
        result = {
            "ok": False,
            "errors": [f"score name mismatch: file has {found}, argument is {name}"],
            "replay_ok": False,
            "stat": float("nan"),
            "pvalue": float("nan"),
            "mean": float("nan"),
            "theory_mean": float("nan"),
        }
        return result, found
    return check_metric(values, dist, seed, replay=replay, alpha=alpha), found


def write_score_table(
    path: str,
    user_ids: np.ndarray,
    columns: list[tuple[str, np.ndarray, bool]],
) -> None:
    """把多个分数并成一张表：user_id 加上每一列分数。第三项为 True 表示该列是 0/1。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    names = [name for name, _values, _discrete in columns]
    with open(path, "w", newline="") as handle:
        handle.write("user_id," + ",".join(names) + "\n")
        for index, user_id in enumerate(user_ids.tolist()):
            cells = [str(int(user_id))]
            for _name, values, discrete in columns:
                value = values[index]
                cells.append(str(int(value)) if discrete else format(float(value), ".17g"))
            handle.write(",".join(cells) + "\n")
