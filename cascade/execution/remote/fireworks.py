"""
cascade.execution.remote.fireworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Async Fireworks AI API client.

Designed for speculative parallel execution — the client can be
started immediately when the EscalationPredictor fires, then cancelled
cleanly if the local model wins the race first.

Key feature: draft_prefix injection.
When the local model has already generated N tokens before escalating,
those tokens are sent as a partial assistant message. The remote model
continues from there instead of regenerating from scratch — saving
remote tokens and reducing latency further.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator, Optional

import httpx

log = logging.getLogger(__name__)

FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
DEFAULT_TIMEOUT = 30.0


class FireworksClient:
    """
    Async Fireworks AI client.

    Usage (context manager — cleans up connections automatically):
        async with FireworksClient() as client:
            result = await client.generate(prompt)

    Or in parallel with asyncio.create_task():
        task = asyncio.create_task(FireworksClient().generate(prompt))
        # Can be cancelled cleanly if local wins
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        model:    Optional[str] = None,
        timeout:  float = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY", "")
        self.model   = model   or os.environ.get(
            "FIREWORKS_MODEL",
            # Updated to revealed model on kickoff day — placeholder here
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
        )
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=FIREWORKS_BASE,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()

    # ── Single-shot generation ───────────────────────────────────────────────

    async def generate(
        self,
        prompt:       str,
        max_tokens:   int   = 512,
        temperature:  float = 0.7,
        draft_prefix: Optional[str] = None,
    ) -> dict:
        """
        Non-streaming generation.

        If `draft_prefix` is provided (local draft from before escalation),
        it is injected as a partial assistant message so the remote model
        continues from there — reducing completion tokens and latency.
        """
        messages = [{"role": "user", "content": prompt}]
        if draft_prefix and len(draft_prefix.strip()) > 30:
            # Inject local draft: remote continues instead of regenerating
            messages.append({"role": "assistant", "content": draft_prefix})

        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      False,
        }

        t0 = time.monotonic()
        if not self._client:
            async with httpx.AsyncClient(
                base_url=FIREWORKS_BASE,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                timeout=self.timeout,
            ) as tmp:
                resp = await tmp.post("/chat/completions", json=payload)
        else:
            resp = await self._client.post("/chat/completions", json=payload)

        resp.raise_for_status()
        data      = resp.json()
        latency   = (time.monotonic() - t0) * 1_000
        usage     = data.get("usage", {})
        content   = data["choices"][0]["message"]["content"]

        log.info(
            "Fireworks: %d prompt + %d completion tokens in %.0fms",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            latency,
        )

        return {
            "text":               content,
            "prompt_tokens":      usage.get("prompt_tokens", 0),
            "completion_tokens":  usage.get("completion_tokens", 0),
            "latency_ms":         round(latency, 1),
            "model":              self.model,
            "draft_injected":     draft_prefix is not None,
        }

    # ── Streaming generation ─────────────────────────────────────────────────

    async def generate_stream(
        self,
        prompt:      str,
        max_tokens:  int   = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        SSE streaming — yields text chunks as they arrive.
        Used for real-time dashboard confidence heatmap updates.
        """
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      True,
        }
        client = self._client
        if not client:
            raise RuntimeError("Use async with FireworksClient() as client:")

        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    text  = delta.get("content", "")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError):
                    continue
