"""
layer -- 把用户哈希进某一层的 k 个桶。桶号只由 user_id 和层名决定。

哈希在 abhack.utils.hash.hash_buckets。这里只负责读写和检验。

SDK:
    from abhack.utils.hash import hash_buckets
    buckets = hash_buckets(ids, n_buckets=2, layer="layer0")

CLI:
    python -m abhack.layer -i exp/users.csv -o exp/layer0.csv --layer layer0 -k 2
    python -m abhack.layer --check -i exp/users.csv -b exp/layer0.csv --layer layer0 -k 2
"""

from .core import assign_buckets, check_assignment, layer_key
from .io import check_assignment_file, iter_assignment, write_assignment

__all__ = [
    "assign_buckets",
    "check_assignment",
    "check_assignment_file",
    "iter_assignment",
    "layer_key",
    "write_assignment",
]
