"""Allocation rules used by the study.

System: sticky_buckets / unhackable_buckets
Agent: honest_arm / keep_extremes (+ label_best_worst)
"""

from .core import (
    honest_arm,
    keep_extremes,
    label_best_worst,
    sticky_buckets,
    traffic_mask,
    unhackable_buckets,
)

__all__ = [
    "honest_arm",
    "keep_extremes",
    "label_best_worst",
    "sticky_buckets",
    "traffic_mask",
    "unhackable_buckets",
]
