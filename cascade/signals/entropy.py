"""
cascade.signals.entropy
~~~~~~~~~~~~~~~~~~~~~~~~
Shannon entropy of the next-token probability distribution.

High entropy = model is genuinely uncertain what token comes next.
This is the strongest single signal for detecting model confusion —
it measures the *spread* of the prediction, not just the top token.
"""
from __future__ import annotations
import math
from typing import Sequence


def token_entropy(logprobs: Sequence[float]) -> float:
    """
    Shannon entropy of a next-token distribution in nats.

    Args:
        logprobs: top-k log-probabilities from the model's output head
    Returns:
        entropy H = -sum(p * log(p)) in nats
    """
    if not logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log(p + 1e-12) for p in probs if p > 0)


def rolling_entropy(history: list[float], window: int = 10) -> float:
    """Mean entropy over the last `window` steps."""
    if not history:
        return 0.0
    recent = history[-window:]
    return sum(recent) / len(recent)


def entropy_trend(history: list[float], window: int = 5) -> float:
    """
    Slope of entropy over the last `window` steps (linear regression).
    Positive slope = entropy rising = model getting more confused.
    """
    if len(history) < 2:
        return 0.0
    recent = history[-window:]
    n = len(recent)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(recent) / n
    numerator   = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0.0
