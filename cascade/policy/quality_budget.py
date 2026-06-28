"""
cascade.policy.quality_budget
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Quality budget accounting — tracks abstract "spend" units and USD cost.

Every InferenceProcess starts with a quality_budget (default 100).
Each state transition deducts units. If the budget is exhausted, the
state machine blocks further upgrades.

This gives a simple, intuitive product story:
  "Simple request: spend 20. Hard request: spend 200."
instead of the harder-to-explain confidence threshold framing.
"""
from __future__ import annotations
import logging
from ..core.inference_process import InferenceProcess, InferenceState, QUALITY_SPEND

log = logging.getLogger(__name__)

# Approximate Fireworks AI cost per 1K output tokens (update from docs on kickoff day)
TOKEN_COST_USD_PER_1K: dict[InferenceState, float] = {
    InferenceState.LOCAL_FAST:    0.0,
    InferenceState.LOCAL_VERIFY:  0.0,
    InferenceState.LOCAL_RECOVER: 0.0,
    InferenceState.REMOTE_ESCAPE: 0.0009,
}


class QualityBudget:
    def __init__(self, process: InferenceProcess):
        self.process = process
        self._log: list[tuple[InferenceState, int]] = []

    def spend(self, state: InferenceState) -> bool:
        cost = QUALITY_SPEND.get(state, 0)
        if cost > self.process.quality_budget:
            log.warning("[%s] Budget overflow: need %d, have %d",
                        self.process.process_id, cost, self.process.quality_budget)
            return False
        self.process.quality_budget -= cost
        self._log.append((state, cost))
        return True

    def can_afford(self, state: InferenceState) -> bool:
        return QUALITY_SPEND.get(state, 0) <= self.process.quality_budget

    def add_token_cost(self, state: InferenceState, n_tokens: int):
        rate = TOKEN_COST_USD_PER_1K.get(state, 0.0)
        self.process.cost_actual_usd += (n_tokens / 1000.0) * rate

    def summary(self) -> dict:
        return {
            "budget_remaining": self.process.quality_budget,
            "spend_log": [(s.value, c) for s, c in self._log],
            "cost_usd": round(self.process.cost_actual_usd, 6),
        }
