"""
cascade.core.runtime_controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The RuntimeController is Cascade's 'OS kernel'.

It orchestrates an InferenceProcess through its full lifecycle:
  1. Allocates resource budgets
  2. Starts local AMD GPU inference (speculative, immediate)
  3. Observes 5 runtime signals per token via the signal layer
  4. Queries the EscalationPredictor every N tokens
  5. Manages state transitions via the StateMachine
  6. Launches Fireworks AI in parallel when escalation is predicted
  7. Races both models via SpeculativeParallelExecutor
  8. Merges the result and logs token waste + cost

Design principle: closed-loop control, not one-shot routing.
The controller never stops observing. Every token is a new data point.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

from .inference_process import InferenceProcess, InferenceState, QUALITY_SPEND
from .state_machine import InferenceStateMachine
from ..signals.entropy import token_entropy, entropy_trend
from ..signals.confidence import avg_confidence
from ..signals.repetition import ngram_repetition_score
from ..signals.speed import SpeedMonitor
from ..signals.fusion import RawSignals, fuse_signals, score_zone
from ..policy.quality_budget import QualityBudget
from ..policy.escalation_predictor import EscalationPredictor
from ..policy.online_learning import OnlineLearner
from ..execution.remote.fireworks import FireworksClient
from ..execution.remote.parallel_merge import SpeculativeParallelExecutor, RaceResult

log = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
SIGNAL_WINDOW      = 10    # tokens between signal reads
ESCALATION_TRIGGER = 0.70  # P(escalation) to launch remote
UPGRADE_Q4_SCORE   = 0.42  # fused score to upgrade LOCAL_FAST → LOCAL_VERIFY
UPGRADE_Q8_SCORE   = 0.55  # fused score to upgrade LOCAL_VERIFY → LOCAL_RECOVER
EARLY_EXIT_SCORE   = 0.18  # fused score below which we stop early (very confident)
MIN_TOKENS_EARLY   = 20    # don't early-exit before this many tokens


class RuntimeController:
    """
    Closed-loop inference controller.

    Designed to be instantiated per-server (not per-request).
    The local engine is shared across requests; the process-level
    objects (StateMachine, QualityBudget, etc.) are created fresh
    for each InferenceProcess.
    """

    def __init__(
        self,
        local_engine=None,
        fireworks_client: Optional[FireworksClient] = None,
        learner: Optional[OnlineLearner] = None,
        broadcast_fn: Optional[Callable[[dict], Awaitable[None]]] = None,
        escalation_trigger: float = ESCALATION_TRIGGER,
        quality_threshold:  float = 0.75,
    ):
        self.local_engine       = local_engine
        self.fireworks_client   = fireworks_client
        self.learner            = learner or OnlineLearner()
        self.broadcast_fn       = broadcast_fn
        self.escalation_trigger = escalation_trigger
        self.quality_threshold  = quality_threshold

        self.predictor = EscalationPredictor(trigger_prob=escalation_trigger)
        self.executor  = SpeculativeParallelExecutor(
            quality_threshold=quality_threshold,
        )

    # ── Public entrypoint ─────────────────────────────────────────────────────

    async def run(self, process: InferenceProcess) -> InferenceProcess:
        """
        Execute an InferenceProcess to completion.

        This is the main decoding loop. Every token is observed.
        State transitions happen inside the loop based on live signals.
        Returns the process with all outcome fields populated.
        """
        t_start = time.monotonic()

        sm      = InferenceStateMachine(process)
        budget  = QualityBudget(process)
        speed   = SpeedMonitor(
            baseline_tps=self._baseline_tps(InferenceState.LOCAL_FAST),
        )

        # Per-request signal accumulators
        generated_tokens: list[str]  = []
        logprob_history:  list[float] = []
        entropy_history:  list[float] = []
        token_ids:        list[int]   = []

        # ── INIT → LOCAL_FAST ─────────────────────────────────────────────────
        if not (sm.transition(InferenceState.LOCAL_FAST)
                and budget.spend(InferenceState.LOCAL_FAST)):
            sm.force_finish("budget exhausted at init")
            process.latency_ms = (time.monotonic() - t_start) * 1_000
            return process

        await self._push(process, "state_change")

        # ── DECODING LOOP ─────────────────────────────────────────────────────
        step = 0
        async for token_data in self._local_stream(
            process.prompt, state=process.state
        ):
            token      = token_data.get("token", "")
            logprob    = token_data.get("logprob", -1.5)
            top_lps    = token_data.get("top_logprobs", [])
            token_id   = token_data.get("token_id", 0)

            generated_tokens.append(token)
            logprob_history.append(logprob)
            token_ids.append(token_id)
            speed.record_token()
            process.tokens_local += 1
            budget.add_token_cost(process.state, 1)
            step += 1

            # Collect entropy
            step_entropy = (
                token_entropy(top_lps) if top_lps else abs(logprob)
            )
            entropy_history.append(step_entropy)

            # ── SIGNAL FUSION (every SIGNAL_WINDOW tokens) ───────────────────
            if step % SIGNAL_WINDOW == 0 or step == 1:
                raw = RawSignals(
                    entropy=step_entropy + 0.5 * entropy_trend(entropy_history),
                    confidence=avg_confidence(logprob_history[-16:]),
                    repetition=ngram_repetition_score(
                        generated_tokens[-40:], n=3
                    ),
                    speed=speed.speed_signal,
                    classifier=self._classifier_prior(process),
                )
                frame = fuse_signals(raw, step=step)
                process.signal_history.append(frame)

                fused  = frame.fused_score
                zone   = score_zone(fused)
                e_prob = self.predictor.predict(process.signal_history)

                await self._push(process, "signal", {
                    "signal":      frame.to_dict(),
                    "zone":        zone,
                    "escalation_prob": e_prob,
                    "step":        step,
                    "token":       token,
                })

                log.debug(
                    "[%s] step=%d fused=%.3f zone=%s P(esc)=%.0f%%",
                    process.process_id, step, fused, zone, e_prob * 100,
                )

                # ── STATE MACHINE EVALUATION ─────────────────────────────────

                # Early exit on high confidence
                if (fused < EARLY_EXIT_SCORE
                        and step >= MIN_TOKENS_EARLY
                        and process.state == InferenceState.LOCAL_FAST):
                    log.info("[%s] Early exit at step %d (fused=%.3f)",
                             process.process_id, step, fused)
                    break

                # Upgrade: LOCAL_FAST → LOCAL_VERIFY
                if (process.state == InferenceState.LOCAL_FAST
                        and fused >= UPGRADE_Q4_SCORE
                        and sm.can_transition(InferenceState.LOCAL_VERIFY)[0]):
                    sm.transition(InferenceState.LOCAL_VERIFY)
                    budget.spend(InferenceState.LOCAL_VERIFY)
                    speed.baseline_tps = self._baseline_tps(InferenceState.LOCAL_VERIFY)
                    await self._push(process, "state_change")

                # Upgrade: LOCAL_VERIFY → LOCAL_RECOVER
                if (process.state == InferenceState.LOCAL_VERIFY
                        and fused >= UPGRADE_Q8_SCORE
                        and sm.can_transition(InferenceState.LOCAL_RECOVER)[0]):
                    sm.transition(InferenceState.LOCAL_RECOVER)
                    budget.spend(InferenceState.LOCAL_RECOVER)
                    speed.baseline_tps = self._baseline_tps(InferenceState.LOCAL_RECOVER)
                    await self._push(process, "state_change")

                # ── SPECULATIVE PARALLEL EXECUTION TRIGGER ───────────────────
                if (e_prob >= self.escalation_trigger
                        and process.state in (InferenceState.LOCAL_VERIFY,
                                              InferenceState.LOCAL_RECOVER)
                        and self.fireworks_client is not None
                        and sm.can_transition(InferenceState.REMOTE_ESCAPE)[0]):

                    log.info(
                        "[%s] Launching parallel race — P(esc)=%.0f%% "
                        "at step %d (fused=%.3f)",
                        process.process_id, e_prob * 100, step, fused,
                    )
                    sm.transition(InferenceState.REMOTE_ESCAPE)
                    await self._push(process, "remote_triggered", {
                        "escalation_prob": e_prob,
                        "local_tokens_so_far": process.tokens_local,
                    })

                    draft = "".join(generated_tokens)
                    race  = await self._parallel_race(process, draft, t_start)
                    return self._finalize_race(process, race, t_start, budget)

        # ── LOCAL COMPLETED WITHOUT ESCALATION ───────────────────────────────
        process.response    = "".join(generated_tokens)
        process.route_taken = f"local:{process.state.value}"
        process.quality_score = self._score_response(process.response)
        sm.transition(InferenceState.FINISHED)
        process.latency_ms = (time.monotonic() - t_start) * 1_000

        self.learner.record(
            domain=self._detect_domain(process.prompt),
            local_won=True,
            quality=process.quality_score,
        )
        await self._push(process, "finished")
        log.info(
            "[%s] DONE local route=%s latency=%.0fms cost=$%.6f waste=%.1f%%",
            process.process_id, process.route_taken,
            process.latency_ms, process.cost_actual_usd, process.waste_pct * 100,
        )
        return process

    # ── Parallel race ─────────────────────────────────────────────────────────

    async def _parallel_race(
        self,
        process: InferenceProcess,
        draft:   str,
        t_start: float,
    ) -> RaceResult:
        """
        Launch local (continue from draft) and remote (Fireworks AI)
        simultaneously. Neither idles. Best response wins.
        """
        tokens_before = process.tokens_local

        async def local_continue() -> dict:
            tokens: list[str] = []
            t0 = time.monotonic()
            async for tok in self._local_stream(
                process.prompt, state=InferenceState.LOCAL_RECOVER
            ):
                tokens.append(tok.get("token", ""))
                process.tokens_local += 1
            return {
                "text":              draft + "".join(tokens),
                "completion_tokens": len(tokens),
                "latency_ms":        (time.monotonic() - t0) * 1_000,
            }

        async def remote_generate() -> dict:
            async with self.fireworks_client as client:
                return await client.generate(
                    prompt=process.prompt,
                    draft_prefix=draft if len(draft.strip()) > 40 else None,
                )

        return await self.executor.race(
            local_coro=local_continue,
            remote_coro=remote_generate,
            local_tokens_preflight=tokens_before,
            quality_scorer=self._score_response,
        )

    def _finalize_race(
        self,
        process: InferenceProcess,
        race:    RaceResult,
        t_start: float,
        budget:  QualityBudget,
    ) -> InferenceProcess:
        process.response       = race.text
        process.route_taken    = race.winner
        process.tokens_wasted  = race.wasted_tokens
        process.tokens_reused  = race.reused_tokens
        process.tokens_remote  = race.remote_tokens
        process.quality_score  = self._score_response(race.text)
        process.latency_ms     = (time.monotonic() - t_start) * 1_000
        budget.add_token_cost(InferenceState.REMOTE_ESCAPE, race.remote_tokens)

        InferenceStateMachine(process).transition(InferenceState.FINISHED)

        self.learner.record(
            domain=self._detect_domain(process.prompt),
            local_won=(race.winner == "local"),
            quality=process.quality_score,
        )
        asyncio.create_task(self._push(process, "finished", race.to_dict()))
        log.info(
            "[%s] DONE race winner=%s latency=%.0fms waste=%d tokens "
            "remote=%d tokens cost=$%.6f",
            process.process_id, race.winner,
            process.latency_ms, race.wasted_tokens,
            race.remote_tokens, process.cost_actual_usd,
        )
        return process

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _local_stream(self, prompt: str, state: InferenceState):
        """Dispatch to real engine or dev stub."""
        from ..execution.local.quant_selector import quant_for_state
        quant = quant_for_state(state)
        if self.local_engine and getattr(self.local_engine, "_ready", False):
            async for tok in self.local_engine.stream(prompt, quant=quant):
                yield tok
        else:
            from ..execution.local.engine import LocalInferenceEngine
            stub = LocalInferenceEngine()
            async for tok in stub._stub_stream(prompt, quant):
                yield tok

    async def _push(
        self, process: InferenceProcess, event: str, extra: dict = None
    ):
        if not self.broadcast_fn:
            return
        payload = {"event": event, "process": process.to_dict()}
        if extra:
            payload.update(extra)
        try:
            await self.broadcast_fn(payload)
        except Exception as exc:
            log.debug("Broadcast failed: %s", exc)

    def _classifier_prior(self, process: InferenceProcess) -> float:
        domain = self._detect_domain(process.prompt)
        return self.learner.classifier_prior(domain)

    @staticmethod
    def _detect_domain(prompt: str) -> str:
        p = prompt.lower()
        if any(k in p for k in ("law", "contract", "legal", "liability")):
            return "legal"
        if any(k in p for k in ("diagnos", "symptom", "patient", "medical")):
            return "medical"
        if any(k in p for k in ("def ", "class ", "function", "algorithm",
                                  "implement", "code", "python", "c++")):
            return "coding"
        if any(k in p for k in ("prove", "theorem", "integral", "equation",
                                  "calculus", "matrix")):
            return "math"
        if any(k in p for k in ("poem", "story", "creative", "haiku", "write")):
            return "creative"
        return "general"

    @staticmethod
    def _score_response(text: str) -> float:
        """
        Lightweight heuristic quality scorer.
        Replace with LLM-as-judge on kickoff day when eval tasks are revealed.
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        words = text.split()
        # Penalize very short responses, reward reasonable length
        length_score = min(len(words) / 80.0, 1.0)
        # Penalize heavy repetition
        unique_ratio = len(set(words)) / max(len(words), 1)
        return round(0.5 * length_score + 0.5 * unique_ratio, 3)

    @staticmethod
    def _baseline_tps(state: InferenceState) -> float:
        from ..execution.local.quant_selector import quant_for_state, baseline_tps
        return baseline_tps(quant_for_state(state))
