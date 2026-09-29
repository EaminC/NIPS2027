"""System property vs experimenter behavior.

System:
  hackable   -- layer hash is fixed once and never changes
  unhackable -- every entry into the layer draws a fresh random split

Agent:
  honest -- no peeking; A/B is random each round
  hacker -- keep the best and worst groups, reshuffle the middle

Theory upper / lower are not experiments. On each system they are
best-group mean minus worst-group mean, and the reverse.
"""

from .core import AGENTS, CELLS, RoundRow, SYSTEMS, simulate

__all__ = ["AGENTS", "CELLS", "RoundRow", "SYSTEMS", "simulate"]
