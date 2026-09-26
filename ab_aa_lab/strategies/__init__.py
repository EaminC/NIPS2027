"""Importing this package registers the builtin strategies."""

from ab_aa_lab.strategies.extremes import ExtremesKeeper
from ab_aa_lab.strategies.honest import Honest
from ab_aa_lab.strategies.oracle import Oracle

__all__ = ["ExtremesKeeper", "Honest", "Oracle"]
