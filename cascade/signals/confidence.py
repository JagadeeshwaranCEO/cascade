"""
cascade.signals.confidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Token-level log-probability confidence signals.

Unlike entropy (which measures the *distribution*), confidence
measures the *actual probability* of the tokens the model generated.
Low confidence = model assigned low probability to its own output.
"""
from __future__ import annotations
import math
from typing import Sequence


def logprob_confidence(logprob: float) -> float:
    """Convert a single log-prob → confidence [0, 1]. Clamps to avoid underflow."""
    return math.exp(max(logprob, -20.0))


def avg_confidence(logprobs: Sequence[float]) -> float:
    """Mean confidence over a token sequence."""
    if not logprobs:
        return 0.0
    return sum(logprob_confidence(lp) for lp in logprobs) / len(logprobs)


def min_confidence(logprobs: Sequence[float]) -> float:
    """Minimum confidence — the weakest link. Single low token = alarm."""
    if not logprobs:
        return 0.0
    return min(logprob_confidence(lp) for lp in logprobs)


def confidence_decay_rate(history: list[float], window: int = 8) -> float:
    """
    Rate of confidence change over `window` recent steps.
    Positive = confidence rising. Negative = falling (bad).
    """
    if len(history) < 2:
        return 0.0
    recent = history[-window:]
    return (recent[-1] - recent[0]) / max(len(recent), 1)
