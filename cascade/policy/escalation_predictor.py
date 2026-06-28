"""
cascade.policy.escalation_predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predictive (not reactive) escalation controller.

Most routing systems react when a signal falls below a threshold.
Cascade predicts when the threshold *will* be crossed — and starts
the remote model early so cloud latency overlaps with local decoding.

At the moment of prediction, Cascade launches Fireworks AI in parallel.
The local model continues running. Whichever produces a quality result
first wins. No GPU sits idle.

Implementation:
    1. Maintain a rolling window of fused signal scores
    2. Fit a linear trend (np.polyfit, deg=1) over the window
    3. Project the trend forward `lookahead` tokens
    4. Compute P(score > threshold in next N tokens)
    5. Smooth with sigmoid to avoid binary jumping
    6. Return probability to the RuntimeController
"""
from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

from ..core.inference_process import SignalFrame

log = logging.getLogger(__name__)

ESCALATION_THRESHOLD = 0.65   # Fused score above which we consider escalation needed
PREDICTION_LOOKAHEAD = 20     # Predict N tokens ahead
TRIGGER_PROBABILITY  = 0.70   # Start remote when P(escalation) >= this


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class EscalationPredictor:
    """
    Projects the fused signal trajectory forward and predicts
    whether the model will need to escalate within `lookahead` tokens.

    The key insight: if the slope of the signal trend suggests crossing
    the threshold in 20 tokens, we can start the remote model NOW —
    at zero extra latency cost, since the local model keeps running.
    """

    def __init__(
        self,
        threshold:    float = ESCALATION_THRESHOLD,
        lookahead:    int   = PREDICTION_LOOKAHEAD,
        min_history:  int   = 5,
        trigger_prob: float = TRIGGER_PROBABILITY,
    ):
        self.threshold    = threshold
        self.lookahead    = lookahead
        self.min_history  = min_history
        self.trigger_prob = trigger_prob

    def predict(self, frames: list[SignalFrame]) -> float:
        """
        Returns P(escalation within next `lookahead` tokens) in [0, 1].

        Returns 0.0 if there is insufficient signal history.
        Uses linear extrapolation with sigmoid smoothing.
        """
        if len(frames) < self.min_history:
            return 0.0

        scores = np.array([f.fused_score for f in frames], dtype=float)
        steps  = np.arange(len(scores), dtype=float)

        # Linear trend via least-squares
        slope, intercept = np.polyfit(steps, scores, deg=1)

        # Project forward
        current_step = len(scores)
        projected = [
            slope * (current_step + i) + intercept
            for i in range(1, self.lookahead + 1)
        ]

        # Fraction of lookahead window predicted above threshold
        over_threshold = sum(1 for s in projected if s >= self.threshold)
        raw_prob = over_threshold / self.lookahead

        # Sigmoid smoothing: sharpens around 50% to avoid gradual creep
        smoothed = _sigmoid(8.0 * (raw_prob - 0.5))

        log.debug(
            "Escalation prediction: slope=%.4f raw_prob=%.2f smoothed=%.2f",
            slope, raw_prob, smoothed,
        )
        return round(smoothed, 3)

    def should_start_remote(self, frames: list[SignalFrame]) -> bool:
        """
        True when predicted escalation probability exceeds `trigger_prob`.

        This is the moment Cascade launches Fireworks AI in parallel
        without waiting for local inference to fail first.
        """
        prob = self.predict(frames)
        if prob >= self.trigger_prob:
            log.info(
                "Remote trigger fired: P(escalation)=%.0f%%  (threshold=%.0f%%)",
                prob * 100,
                self.trigger_prob * 100,
            )
            return True
        return False

    def summary(self, frames: list[SignalFrame]) -> dict:
        """Dashboard-ready snapshot of the current prediction state."""
        prob = self.predict(frames)
        return {
            "escalation_probability":  prob,
            "should_start_remote":     prob >= self.trigger_prob,
            "current_fused_score":     frames[-1].fused_score if frames else 0.0,
            "threshold":               self.threshold,
            "trigger_probability":     self.trigger_prob,
            "history_length":          len(frames),
            "zone":                    _zone(prob),
        }


def _zone(prob: float) -> str:
    if prob < 0.30:  return "safe"
    if prob < 0.60:  return "watch"
    if prob < 0.80:  return "warning"
    return "escalating"
