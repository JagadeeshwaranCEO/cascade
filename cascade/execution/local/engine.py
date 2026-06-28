"""
cascade.execution.local.engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AMD ROCm local inference engine via vLLM.

Wraps vLLM's async engine and exposes a streaming token generator
that also yields per-token log-probabilities for signal extraction.

On AMD Developer Cloud (MI300X / RX 7900):
    - Uses ROCm-compiled vLLM
    - Model loaded once at startup
    - Supports hot-swapping quantization via separate model variants
      (Q2_K, Q4_K_M, Q8_0 loaded as separate engine instances or
       via GGUF if llama.cpp ROCm is used as the backend)
"""
from __future__ import annotations
import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

log = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "/app/models")


class LocalInferenceEngine:
    """
    AMD ROCm local model runner.
    Streams tokens with log-probabilities for the signal layer.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.model_path = model_path or os.environ.get("LOCAL_MODEL_PATH", "")
        self.device     = device
        self._engines: dict[str, object] = {}  # quant → vLLM AsyncLLMEngine
        self._ready     = False

    async def load(self, quants: list[str] = ("q4_k_m",)):
        """
        Pre-load model variants at startup.
        Call once before serving requests.
        """
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
            for quant in quants:
                model_path = f"{self.model_path}.{quant}.gguf"
                if not os.path.exists(model_path):
                    log.warning("Model not found: %s — using stub", model_path)
                    continue
                args = AsyncEngineArgs(
                    model=model_path,
                    dtype="auto",
                    device=self.device,
                    max_model_len=4096,
                    enable_prefix_caching=True,
                )
                self._engines[quant] = AsyncLLMEngine.from_engine_args(args)
                log.info("Loaded %s on %s", quant, self.device)
            self._ready = bool(self._engines)
        except ImportError:
            log.warning("vLLM not available — using stub engine")
            self._ready = False

    async def stream(
        self,
        prompt: str,
        quant:  str = "q4_k_m",
        max_tokens: int = 512,
    ) -> AsyncGenerator[dict, None]:
        """
        Async token stream with log-probabilities.

        Yields dicts:
            {token: str, logprob: float, top_logprobs: list[float]}
        """
        engine = self._engines.get(quant)
        if engine is None:
            async for chunk in self._stub_stream(prompt, quant):
                yield chunk
            return

        # Real vLLM path
        from vllm import SamplingParams
        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            logprobs=5,      # Top-5 log-probs for entropy computation
        )
        request_id = f"req-{id(prompt)}"
        async for output in engine.generate(prompt, sampling, request_id):
            for completion in output.outputs:
                if not completion.token_ids:
                    continue
                token_id  = completion.token_ids[-1]
                logprob   = list(completion.logprobs[-1].values())[0].logprob if completion.logprobs else -1.0
                top_lps   = [v.logprob for v in completion.logprobs[-1].values()] if completion.logprobs else []
                yield {
                    "token":       completion.text,
                    "logprob":     logprob,
                    "top_logprobs": top_lps,
                }

    async def _stub_stream(
        self, prompt: str, quant: str
    ) -> AsyncGenerator[dict, None]:
        """Development stub — simulates a real AMD model stream."""
        import math, random
        words = (
            "The quick brown fox jumps over the lazy dog and "
            "illustrates streaming inference with simulated logprobs "
        ).split()
        # Simulate degrading confidence mid-stream (to test escalation)
        for i, word in enumerate(words[:50]):
            quality_drop = max(0.0, (i - 15) / 35.0)
            logprob = random.uniform(-0.5, -0.05) - quality_drop * 3
            top_lps = [logprob + random.uniform(-1.5, 0.3) for _ in range(5)]
            yield {"token": word + " ", "logprob": logprob, "top_logprobs": top_lps}
            await asyncio.sleep(1.0 / 30.0)  # Simulate ~30 tps
