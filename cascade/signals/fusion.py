"""
cascade.signals.fusion
~~~~~~~~~~~~~~~~~~~~~~~
Multi-signal fusion — the brain of the RuntimeController's decision loop.

Instead of routing on a single confidence threshold, Cascade computes
a weighted combination of 5 independent runtime signals. Each signal
captures a different failure mode:

    entropy      — model is confused about the next token distribution
    confidence   — the generated tokens themselves have low probability
    repetition   — the model is stuck in a loop
    speed        — the AMD GPU is under memory/compute pressure
    classifier   — the query was pre-classified as hard

The fused score is in [0, 1] where:
    0.0  =  model is confident, fast, non-repetitive → stay local
    1.0  =  model is struggling in every dimension  → escalate

The EscalationPredictor then projects this score forward N tokens
to predict escalation *before* it happens.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..core.inference_process import SignalFrame

# ---------------------------------------------------------------------------
# Weight configuration — must sum to 1.0
# ---------------------------------------------------------------------------

SIGNAL_WEIGHTS: dict[str, float] = {
    "entropy":    0.35,   # Primary: token distribution uncertainty
    "confidence": 0.25,   # Direct token-level probability
    "repetition": 0.15,   # Degenerate generation detection
    "speed":      0.10,   # KV-cache / memory pressure proxy
    "classifier": 0.15,   # Query-level pre-classification hardness
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Raw signal container
# ---------------------------------------------------------------------------

@dataclass
class RawSignals:
    """
    Raw values from each signal channel, collected at one decoding step.
    Units vary per signal — fusion normalizes them.
    """
    entropy:    float   # [0, ∞]  Shannon entropy of next-token distribution
    confidence: float   # [0, 1]  Mean confidence of recent tokens (higher = better)
    repetition: float   # [0, 1]  N-gram repetition rate (higher = worse)
    speed:      float   # [0, 1]  Normalized tokens/sec, inverted (higher = slower)
    classifier: float   # [0, 1]  Pre-classification query hardness score


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_entropy(entropy: float, midpoint: float = 2.0, steepness: float = 2.0) -> float:
    """
    Map raw Shannon entropy [0, ∞] → [0, 1] using a sigmoid.
    entropy ≈ 0    → ~0.12  (very confident next token)
    entropy ≈ 2.0  → ~0.50  (uncertain — typical language model)
    entropy ≈ 4.0  → ~0.88  (very confused)
    """
    return 1.0 / (1.0 + math.exp(-steepness * (entropy - midpoint)))


def _invert_confidence(confidence: float) -> float:
    """High confidence is good → invert so high value means stress."""
    return 1.0 - max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# Core fusion function
# ---------------------------------------------------------------------------

def fuse_signals(signals: RawSignals, step: int) -> SignalFrame:
    """
    Combine raw signals into a single fused quality score.

    Returns a SignalFrame ready to append to InferenceProcess.signal_history.

    Score interpretation:
        < 0.30   safe — LOCAL_FAST is handling this well
        0.30–0.50  watch — consider upgrading precision
        0.50–0.65  warn — upgrade to LOCAL_RECOVER
        > 0.65   escalate — EscalationPredictor should fire
    """
    entropy_norm    = _normalize_entropy(signals.entropy)
    confidence_inv  = _invert_confidence(signals.confidence)
    repetition_norm = max(0.0, min(1.0, signals.repetition))
    speed_norm      = max(0.0, min(1.0, signals.speed))
    classifier_norm = max(0.0, min(1.0, signals.classifier))

    fused = (
        SIGNAL_WEIGHTS["entropy"]    * entropy_norm    +
        SIGNAL_WEIGHTS["confidence"] * confidence_inv  +
        SIGNAL_WEIGHTS["repetition"] * repetition_norm +
        SIGNAL_WEIGHTS["speed"]      * speed_norm      +
        SIGNAL_WEIGHTS["classifier"] * classifier_norm
    )

    return SignalFrame(
        step=step,
        timestamp=time.time(),
        token_entropy=round(signals.entropy, 4),
        avg_logprob=round(signals.confidence, 4),
        repetition_score=round(repetition_norm, 4),
        generation_speed=round(speed_norm, 4),
        fused_score=round(fused, 4),
    )


def fused_score_history(frames: list[SignalFrame]) -> list[float]:
    """Extract the fused score series for trend analysis."""
    return [f.fused_score for f in frames]


def score_zone(fused: float) -> str:
    """Human-readable zone label for dashboard coloring."""
    if fused < 0.30:
        return "GREEN"
    if fused < 0.50:
        return "YELLOW"
    if fused < 0.65:
        return "ORANGE"
    return "RED"
