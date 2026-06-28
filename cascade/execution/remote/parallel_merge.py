"""
cascade.execution.remote.parallel_merge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Speculative parallel execution — the hardware efficiency core of Cascade.

Traditional escalation:
    local fails → stop → wait for remote → response
    (AMD GPU sits idle while waiting for Fireworks AI)

Cascade speculative parallel:
    local continues → remote starts simultaneously → both race → best wins
    (AMD GPU never idles, cloud latency overlaps local decoding)

This is the key contribution cited in: "no idle GPU."
The merge engine evaluates quality from whichever coroutine finishes first
and cancels the other cleanly via asyncio.Task.cancel().
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class RaceResult:
    """
    Outcome of a speculative parallel race.
    Fully documents which path won and why, for dashboard and cost logging.
    """
    winner:          str    # "local" | "remote" | "local_fallback"
    text:            str    # Winning response
    local_tokens:    int    # Tokens AMD GPU generated (may include wasted)
    remote_tokens:   int    # Tokens Fireworks AI generated (0 if local wins)
    wasted_tokens:   int    # Local tokens discarded when remote wins
    reused_tokens:   int    # Local draft tokens remote leveraged (saves cost)
    local_latency_ms:  float
    remote_latency_ms: float
    total_latency_ms:  float

    def to_dict(self) -> dict:
        return {
            "winner":            self.winner,
            "local_tokens":      self.local_tokens,
            "remote_tokens":     self.remote_tokens,
            "wasted_tokens":     self.wasted_tokens,
            "reused_tokens":     self.reused_tokens,
            "waste_pct":         round(self.wasted_tokens / max(self.local_tokens, 1) * 100, 1),
            "local_latency_ms":  round(self.local_latency_ms, 1),
            "remote_latency_ms": round(self.remote_latency_ms, 1),
            "total_latency_ms":  round(self.total_latency_ms, 1),
        }


class SpeculativeParallelExecutor:
    """
    Races a local AMD GPU coroutine against a Fireworks AI remote coroutine.

    Both are launched simultaneously with asyncio.create_task().
    The first to finish above `quality_threshold` wins.
    The loser is cancelled (asyncio.Task.cancel) with no resource leak.
    """

    def __init__(
        self,
        quality_threshold: float = 0.75,
        remote_timeout_ms: float = 10_000.0,
    ):
        self.quality_threshold = quality_threshold
        self.remote_timeout_ms = remote_timeout_ms

    async def race(
        self,
        local_coro:            Callable[[], Awaitable[dict]],
        remote_coro:           Callable[[], Awaitable[dict]],
        local_tokens_preflight: int = 0,
        quality_scorer:        Optional[Callable[[str], float]] = None,
    ) -> RaceResult:
        """
        Launch both coroutines concurrently. Evaluate and merge results.

        Args:
            local_coro:            async callable returning {text, completion_tokens, latency_ms}
            remote_coro:           async callable returning {text, completion_tokens, latency_ms}
            local_tokens_preflight: tokens already generated before this race started
            quality_scorer:        optional fn(text) → [0, 1] quality score
        """
        t_race_start = time.monotonic()

        local_task  = asyncio.create_task(local_coro(),  name="local")
        remote_task = asyncio.create_task(remote_coro(), name="remote")

        local_result:  Optional[dict] = None
        remote_result: Optional[dict] = None
        pending = {local_task, remote_task}

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=self.remote_timeout_ms / 1_000.0,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    log.warning("Race timeout after %.0fms", self.remote_timeout_ms)
                    break

                for task in done:
                    try:
                        result = task.result()
                    except Exception as exc:
                        log.error("Task %s failed: %s", task.get_name(), exc)
                        continue

                    if task.get_name() == "local":
                        local_result = result
                        log.info("Local finished (%.0fms)", result.get("latency_ms", 0))
                    else:
                        remote_result = result
                        log.info("Remote finished (%.0fms)", result.get("latency_ms", 0))

                # Early exit: take first quality result
                for candidate in (local_result, remote_result):
                    if candidate is None:
                        continue
                    score = quality_scorer(candidate["text"]) if quality_scorer else 1.0
                    if score >= self.quality_threshold:
                        log.info("Quality gate passed (%.2f >= %.2f) — cancelling other task",
                                 score, self.quality_threshold)
                        break
                else:
                    continue  # Neither passed yet, keep waiting
                break          # We have a winner

        finally:
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        total_ms = (time.monotonic() - t_race_start) * 1_000
        return self._pick_winner(
            local_result, remote_result,
            local_tokens_preflight, total_ms, quality_scorer,
        )

    # ── Winner selection ─────────────────────────────────────────────────────

    def _pick_winner(
        self,
        local:  Optional[dict],
        remote: Optional[dict],
        local_tokens_preflight: int,
        total_ms: float,
        quality_scorer: Optional[Callable[[str], float]],
    ) -> RaceResult:
        def score(r: dict) -> float:
            return quality_scorer(r["text"]) if quality_scorer else 0.9

        local_score  = score(local)  if local  else 0.0
        remote_score = score(remote) if remote else 0.0

        # Prefer local — it's free (zero API cost)
        if local and local_score >= self.quality_threshold:
            return RaceResult(
                winner="local",
                text=local["text"],
                local_tokens=local.get("completion_tokens", 0) + local_tokens_preflight,
                remote_tokens=remote.get("completion_tokens", 0) if remote else 0,
                wasted_tokens=0,
                reused_tokens=0,
                local_latency_ms=local.get("latency_ms", 0.0),
                remote_latency_ms=remote.get("latency_ms", 0.0) if remote else 0.0,
                total_latency_ms=total_ms,
            )

        # Remote wins
        if remote:
            return RaceResult(
                winner="remote",
                text=remote["text"],
                local_tokens=local_tokens_preflight,
                remote_tokens=remote.get("completion_tokens", 0),
                wasted_tokens=local_tokens_preflight,       # pre-flight tokens wasted
                reused_tokens=0,
                local_latency_ms=local.get("latency_ms", 0.0) if local else 0.0,
                remote_latency_ms=remote.get("latency_ms", total_ms),
                total_latency_ms=total_ms,
            )

        # Nothing passed quality — fall back to local anyway
        return RaceResult(
            winner="local_fallback",
            text=local["text"] if local else "",
            local_tokens=local_tokens_preflight,
            remote_tokens=0,
            wasted_tokens=0,
            reused_tokens=0,
            local_latency_ms=total_ms,
            remote_latency_ms=0.0,
            total_latency_ms=total_ms,
        )
