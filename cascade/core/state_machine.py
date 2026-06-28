"""
cascade.core.state_machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Formal state machine controlling an InferenceProcess lifecycle.

This is the enforcement layer — every state transition is validated
against the budget, latency deadline, and legal transition graph
before being allowed. The RuntimeController proposes transitions;
the StateMachine approves or blocks them.

Transition graph:
    INITIALIZING → LOCAL_FAST
    LOCAL_FAST   → LOCAL_VERIFY | REMOTE_ESCAPE | FINISHED
    LOCAL_VERIFY → LOCAL_RECOVER | REMOTE_ESCAPE | FINISHED
    LOCAL_RECOVER→ REMOTE_ESCAPE | FINISHED
    REMOTE_ESCAPE→ FINISHED
    FINISHED     → (terminal)
"""
from __future__ import annotations

import logging
from typing import Optional

from .inference_process import InferenceProcess, InferenceState, QUALITY_SPEND

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legal transitions
# ---------------------------------------------------------------------------

TRANSITIONS: dict[InferenceState, list[InferenceState]] = {
    InferenceState.INITIALIZING:   [InferenceState.LOCAL_FAST],
    InferenceState.LOCAL_FAST:     [InferenceState.LOCAL_VERIFY,
                                    InferenceState.REMOTE_ESCAPE,
                                    InferenceState.FINISHED],
    InferenceState.LOCAL_VERIFY:   [InferenceState.LOCAL_RECOVER,
                                    InferenceState.REMOTE_ESCAPE,
                                    InferenceState.FINISHED],
    InferenceState.LOCAL_RECOVER:  [InferenceState.REMOTE_ESCAPE,
                                    InferenceState.FINISHED],
    InferenceState.REMOTE_ESCAPE:  [InferenceState.FINISHED],
    InferenceState.FINISHED:       [],
}


class StateMachineError(Exception):
    pass


class InferenceStateMachine:
    """
    Guards every state transition for an InferenceProcess.

    Usage:
        sm = InferenceStateMachine(process)
        if sm.transition(InferenceState.LOCAL_VERIFY):
            # transition succeeded, process.state is now LOCAL_VERIFY
        ok, reason = sm.can_transition(InferenceState.REMOTE_ESCAPE)
        # check before committing
    """

    def __init__(self, process: InferenceProcess):
        self.process = process

    # ── Inspection ───────────────────────────────────────────────────────────

    def can_transition(self, target: InferenceState) -> tuple[bool, str]:
        """Return (allowed, reason_string) without modifying state."""
        current = self.process.state

        # 1. Legal path
        allowed = TRANSITIONS.get(current, [])
        if target not in allowed:
            return False, f"No path {current.value} → {target.value}"

        # 2. Quality budget
        required = QUALITY_SPEND.get(target, 0)
        if required > self.process.quality_budget:
            return False, (
                f"Quality budget {self.process.quality_budget} "
                f"< required {required} for {target.value}"
            )

        # 3. Latency deadline
        if self.process.elapsed_ms >= self.process.latency_budget_ms:
            return False, (
                f"Latency budget exhausted "
                f"({self.process.elapsed_ms:.0f}ms >= {self.process.latency_budget_ms}ms)"
            )

        return True, "ok"

    # ── Mutation ─────────────────────────────────────────────────────────────

    def transition(self, target: InferenceState) -> bool:
        """
        Attempt a transition. Returns True on success, False if blocked.
        On success, process.state is updated immediately.
        """
        ok, reason = self.can_transition(target)
        if not ok:
            log.warning(
                "[%s] BLOCKED %s → %s: %s",
                self.process.process_id,
                self.process.state.value,
                target.value,
                reason,
            )
            return False

        log.info(
            "[%s] %s → %s  (budget_remaining=%d, elapsed=%.0fms)",
            self.process.process_id,
            self.process.state.value,
            target.value,
            self.process.budget_remaining,
            self.process.elapsed_ms,
        )
        self.process.state = target
        return True

    def force_finish(self, reason: str = "forced"):
        """
        Unconditional jump to FINISHED — used on timeout, error, or cancellation.
        Bypasses budget/path checks intentionally.
        """
        log.warning("[%s] force_finish: %s", self.process.process_id, reason)
        self.process.state = InferenceState.FINISHED

    # ── Convenience predicates ────────────────────────────────────────────────

    @property
    def is_finished(self) -> bool:
        return self.process.state == InferenceState.FINISHED

    @property
    def is_local(self) -> bool:
        return self.process.state in (
            InferenceState.LOCAL_FAST,
            InferenceState.LOCAL_VERIFY,
            InferenceState.LOCAL_RECOVER,
        )

    @property
    def is_remote(self) -> bool:
        return self.process.state == InferenceState.REMOTE_ESCAPE

    def next_local_upgrade(self) -> Optional[InferenceState]:
        """Return the next local state to upgrade to, if any."""
        upgrades = {
            InferenceState.LOCAL_FAST:    InferenceState.LOCAL_VERIFY,
            InferenceState.LOCAL_VERIFY:  InferenceState.LOCAL_RECOVER,
            InferenceState.LOCAL_RECOVER: None,
        }
        return upgrades.get(self.process.state)
