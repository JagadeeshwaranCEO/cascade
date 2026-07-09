"""
cascade.execution.local.quant_selector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Maps InferenceState → quantization config for the local AMD model.

Ported from TinyLLM-ARM-Pro benchmark data.
Q2_K / Q4_K_M / Q8_0 tradeoffs are AMD-profiled on kickoff day
and calibrated here.
"""
from __future__ import annotations
from cascade.core.inference_process import InferenceState

QUANT_MAP: dict[InferenceState, str] = {
    InferenceState.LOCAL_FAST:    "q2_k",   # 2-bit: ~6× speedup, ~68% RAM reduction
    InferenceState.LOCAL_VERIFY:  "q4_k_m", # 4-bit: balanced speed/quality
    InferenceState.LOCAL_RECOVER: "q8_0",   # 8-bit: highest local quality
}

QUANT_BASELINE_TPS: dict[str, float] = {
    "q2_k":   60.0,  # Fastest — calibrate on AMD GPU at kickoff
    "q4_k_m": 35.0,
    "q8_0":   18.0,
}

def quant_for_state(state: InferenceState) -> str:
    return QUANT_MAP.get(state, "q4_k_m")

def baseline_tps(quant: str) -> float:
    return QUANT_BASELINE_TPS.get(quant, 30.0)
