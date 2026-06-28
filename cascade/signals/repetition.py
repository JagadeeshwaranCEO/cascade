"""
cascade.signals.repetition
~~~~~~~~~~~~~~~~~~~~~~~~~~~
N-gram repetition detection during streaming generation.

Repetition is a strong signal for model failure — it means the model
is stuck in a loop and local inference has degraded to degenerate output.
Detecting it early allows Cascade to abort and escalate before the
user sees nonsense.
"""
from __future__ import annotations
from collections import Counter
from typing import Sequence


def ngram_repetition_score(tokens: Sequence[str], n: int = 3) -> float:
    """
    Distinct-n repetition score [0, 1].
    0.0 = fully diverse, no repetition.
    1.0 = every n-gram is a repeat.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    distinct_ratio = len(set(ngrams)) / len(ngrams)
    return 1.0 - distinct_ratio


def token_id_repetition(token_ids: Sequence[int], window: int = 20) -> float:
    """
    Fast sliding-window check using token IDs.
    Returns the frequency of the most common token in the window.
    > 0.3 → strong repetition signal.
    """
    if not token_ids:
        return 0.0
    recent = list(token_ids)[-window:]
    counts = Counter(recent)
    top_freq = counts.most_common(1)[0][1] / len(recent)
    return top_freq if top_freq > 0.2 else 0.0
