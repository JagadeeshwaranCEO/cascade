"""
cascade.signals.speed
~~~~~~~~~~~~~~~~~~~~~~
Real-time token generation speed monitor.

Speed drops on a fixed AMD GPU indicate KV cache growth, memory
bandwidth saturation, or other runtime pressure. It's a leading
indicator — speed drops *before* the model starts producing bad output.
"""
from __future__ import annotations
import time
from collections import deque


class SpeedMonitor:
    """
    Tracks tokens/sec in a rolling window and exposes it as a
    normalized signal in [0, 1] where 1.0 = full slowdown.

    Calibrate `baseline_tps` by profiling the target AMD GPU model.
    """

    def __init__(self, window: int = 20, baseline_tps: float = 30.0):
        self.window = window
        self.baseline_tps = baseline_tps
        self._timestamps: deque[float] = deque(maxlen=window)

    def record_token(self):
        """Call every time a new token is generated."""
        self._timestamps.append(time.monotonic())

    @property
    def current_tps(self) -> float:
        """Tokens per second in the recent window."""
        if len(self._timestamps) < 2:
            return self.baseline_tps
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else self.baseline_tps

    @property
    def normalized_speed(self) -> float:
        """Fraction of baseline [0, 1]. 1.0 = at or above baseline."""
        return min(self.current_tps / self.baseline_tps, 1.0)

    @property
    def speed_signal(self) -> float:
        """Inverted speed for fusion: 1.0 = very slow (bad), 0.0 = fast (good)."""
        return 1.0 - self.normalized_speed
