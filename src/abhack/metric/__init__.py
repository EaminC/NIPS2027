"""
metric -- 给每个用户打一个自定义名字的分数，分数独立同分布于指定分布。

SDK:
    from abhack.utils.distributions import normal
    from abhack.metric import draw_metric

    values = draw_metric(1000, normal(mu=0.0, sigma=1.0), seed=0)

CLI:
    python -m abhack.metric -i exp/out/users.csv -o exp/out/ctr.csv --name ctr \\
        --dist normal --mu 0 --sigma 1 --seed 0
    python -m abhack.metric --check -i exp/out/ctr.csv --name ctr \\
        --dist normal --mu 0 --sigma 1 --seed 0
"""

from .core import check_metric, draw_metric
from .io import check_metric_file, read_metric, write_metric, write_score_table

__all__ = [
    "check_metric",
    "check_metric_file",
    "draw_metric",
    "read_metric",
    "write_metric",
    "write_score_table",
]
