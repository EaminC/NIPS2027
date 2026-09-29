"""哈希和分布。调用时直接用这里的函数，不要在实验里重写。"""

from .distributions import (
    NAMES,
    Distribution,
    bernoulli,
    exponential,
    make,
    normal,
    uniform,
)
from .hash import hash_buckets, layer_key

__all__ = [
    "NAMES",
    "Distribution",
    "bernoulli",
    "exponential",
    "hash_buckets",
    "layer_key",
    "make",
    "normal",
    "uniform",
]
