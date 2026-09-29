"""
users -- 生成 M 个用户并写入 CSV（只有 user_id）。

SDK:
    from abhack.users import generate_users, write_csv, read_csv

    ids = generate_users(1_000_000)
    write_csv(ids, "exp/users.csv")
    write_csv(1_000_000, "exp/users.csv")  # 按块写，不先占满内存

CLI:
    python -m abhack.users -M 1000000 -o exp/users.csv
"""

from .core import generate_users, iter_chunks
from .io import iter_csv, read_csv, write_csv

__all__ = [
    "generate_users",
    "iter_chunks",
    "iter_csv",
    "read_csv",
    "write_csv",
]
