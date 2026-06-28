"""
cascade.policy.online_learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Domain-aware routing policy that evolves with each request.

Tracks: P(local succeeds | domain).
Uses this to recommend quality budgets and pre-set classifier scores
so the router gets smarter over the course of the eval run.
"""
from __future__ import annotations
import json
import logging
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

DOMAIN_LABELS = [
    "math", "coding", "legal", "medical",
    "creative", "factual", "reasoning", "general",
]


class OnlineLearner:
    """
    Lightweight online learner that adjusts routing policy per domain
    without retraining any model weights.

    After each completed InferenceProcess, call `record()`.
    Before each new process, call `recommended_budget()` and
    `classifier_prior()` to seed the signal fusion layer.
    """

    def __init__(self, state_path: Path | None = None):
        # domain → list of (local_won, quality_score)
        self._history: dict[str, list[tuple[bool, float]]] = defaultdict(list)
        self._state_path = state_path
        if state_path and state_path.exists():
            self._load(state_path)

    def record(self, domain: str, local_won: bool, quality: float):
        self._history[domain].append((local_won, quality))
        if self._state_path:
            self._save(self._state_path)

    def local_success_rate(self, domain: str) -> float:
        records = self._history.get(domain, [])
        if not records:
            return 0.5
        wins = sum(1 for won, q in records if won and q >= 0.75)
        return wins / len(records)

    def recommended_budget(self, domain: str) -> int:
        rate = self.local_success_rate(domain)
        if rate >= 0.80:  return 60    # Mostly local
        if rate >= 0.50:  return 100   # Mixed
        return 160                      # Mostly remote

    def classifier_prior(self, domain: str) -> float:
        """Hardness prior for signal fusion [0, 1]. Higher = harder."""
        return 1.0 - self.local_success_rate(domain)

    def domain_summary(self) -> dict:
        return {
            domain: {
                "local_success_rate":  round(self.local_success_rate(domain), 3),
                "n_samples":           len(records),
                "recommended_budget":  self.recommended_budget(domain),
            }
            for domain, records in self._history.items()
        }

    def _save(self, path: Path):
        data = {d: list(records) for d, records in self._history.items()}
        path.write_text(json.dumps(data))

    def _load(self, path: Path):
        data = json.loads(path.read_text())
        for domain, records in data.items():
            self._history[domain] = [tuple(r) for r in records]
        log.info("OnlineLearner: loaded %d domains from %s",
                 len(self._history), path)
