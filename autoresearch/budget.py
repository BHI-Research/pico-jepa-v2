"""Budget control: wallclock, iteration count, plateau detection.

The user gives a wallclock cap and an iteration cap. The loop terminates
when either is reached, or when the search plateaus (no improvement for K
consecutive iterations across all phases). All three are checked between
iterations — never mid-experiment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Budget:
    max_wallclock_s: Optional[float] = None
    max_iters: Optional[int] = None
    plateau_patience: int = 5  # consecutive iters with no improvement allowed

    started_at: float = field(default_factory=time.time)
    iters_done: int = 0
    consecutive_no_improvement: int = 0

    def remaining_wallclock_s(self) -> Optional[float]:
        if self.max_wallclock_s is None:
            return None
        return self.max_wallclock_s - (time.time() - self.started_at)

    def can_continue(self) -> bool:
        if self.max_wallclock_s is not None and (time.time() - self.started_at) >= self.max_wallclock_s:
            return False
        if self.max_iters is not None and self.iters_done >= self.max_iters:
            return False
        if self.consecutive_no_improvement >= self.plateau_patience:
            return False
        return True

    def record_iteration(self, improved: bool) -> None:
        self.iters_done += 1
        if improved:
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1

    def status(self) -> dict:
        return {
            "iters_done": self.iters_done,
            "elapsed_s": time.time() - self.started_at,
            "remaining_s": self.remaining_wallclock_s(),
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "plateau_patience": self.plateau_patience,
        }


def parse_wallclock(spec: str) -> float:
    """Parse '6h', '30m', '90s', '1.5h'. Returns seconds."""
    spec = spec.strip().lower()
    if spec.endswith("h"):
        return float(spec[:-1]) * 3600
    if spec.endswith("m"):
        return float(spec[:-1]) * 60
    if spec.endswith("s"):
        return float(spec[:-1])
    return float(spec)  # bare number = seconds
