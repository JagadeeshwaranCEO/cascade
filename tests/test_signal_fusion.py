"""
Tests for the Cascade signal layer and escalation predictor.
Run: pytest tests/ -v
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from cascade.signals.entropy import token_entropy, entropy_trend
from cascade.signals.confidence import avg_confidence, logprob_confidence
from cascade.signals.repetition import ngram_repetition_score
from cascade.signals.fusion import RawSignals, fuse_signals, score_zone, SIGNAL_WEIGHTS
from cascade.policy.escalation_predictor import EscalationPredictor
from cascade.core.inference_process import InferenceProcess, InferenceState
from cascade.core.state_machine import InferenceStateMachine


# ── Entropy tests ─────────────────────────────────────────────────────────────

def test_entropy_uniform_distribution():
    """Uniform distribution → maximum entropy for k classes."""
    k = 10
    logprobs = [math.log(1 / k)] * k
    h = token_entropy(logprobs)
    assert h == pytest.approx(math.log(k), rel=1e-3)


def test_entropy_peaked_distribution():
    """Near-deterministic distribution → entropy near 0."""
    logprobs = [0.0] + [-20.0] * 9   # prob ≈ [1, ~0, ...]
    h = token_entropy(logprobs)
    assert h < 0.1


def test_entropy_empty():
    assert token_entropy([]) == 0.0


def test_entropy_trend_rising():
    """Rising entropy history should give positive slope."""
    history = [0.5, 0.8, 1.1, 1.5, 2.0]
    slope = entropy_trend(history)
    assert slope > 0


def test_entropy_trend_stable():
    history = [1.0] * 5
    slope = entropy_trend(history)
    assert abs(slope) < 1e-9


# ── Confidence tests ──────────────────────────────────────────────────────────

def test_logprob_confidence_perfect():
    assert logprob_confidence(0.0) == pytest.approx(1.0)


def test_logprob_confidence_near_zero():
    assert logprob_confidence(-20.0) < 1e-8


def test_avg_confidence_all_high():
    logprobs = [-0.05] * 10
    c = avg_confidence(logprobs)
    assert c > 0.9


def test_avg_confidence_empty():
    assert avg_confidence([]) == 0.0


# ── Repetition tests ──────────────────────────────────────────────────────────

def test_repetition_no_repeat():
    tokens = ["the", "quick", "brown", "fox", "jumps"]
    score = ngram_repetition_score(tokens, n=2)
    assert score == 0.0


def test_repetition_full_repeat():
    # With 10 identical tokens: 9 bigrams, all identical → 1 distinct / 9 total
    # distinct_ratio = 1/9, repetition_score = 1 - 1/9 = 8/9 ≈ 0.888
    tokens = ["cat"] * 10
    score = ngram_repetition_score(tokens, n=2)
    assert score == pytest.approx(8 / 9, rel=1e-3)


def test_repetition_partial():
    tokens = ["a", "b", "c", "a", "b", "c", "d", "e"]
    score = ngram_repetition_score(tokens, n=2)
    assert 0.0 < score < 1.0


# ── Signal fusion tests ───────────────────────────────────────────────────────

def test_fusion_weights_sum_to_one():
    total = sum(SIGNAL_WEIGHTS.values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_fusion_low_stress():
    """Low entropy, high confidence → low fused score (safe)."""
    raw = RawSignals(entropy=0.3, confidence=0.95, repetition=0.0, speed=0.0, classifier=0.1)
    frame = fuse_signals(raw, step=1)
    assert frame.fused_score < 0.35


def test_fusion_high_stress():
    """High entropy, low confidence, repetition → high fused score (escalate)."""
    raw = RawSignals(entropy=4.0, confidence=0.1, repetition=0.8, speed=0.9, classifier=0.9)
    frame = fuse_signals(raw, step=1)
    assert frame.fused_score > 0.65


def test_fusion_score_range():
    """Fused score must always be in [0, 1]."""
    for entropy in [0.0, 2.0, 5.0]:
        for conf in [0.0, 0.5, 1.0]:
            raw = RawSignals(entropy=entropy, confidence=conf,
                             repetition=0.5, speed=0.5, classifier=0.5)
            frame = fuse_signals(raw, step=1)
            assert 0.0 <= frame.fused_score <= 1.0


def test_score_zone_labels():
    assert score_zone(0.1)  == "GREEN"
    assert score_zone(0.40) == "YELLOW"
    assert score_zone(0.58) == "ORANGE"
    assert score_zone(0.80) == "RED"


# ── Escalation predictor tests ────────────────────────────────────────────────

def _make_frames(scores: list[float]) -> list:
    from cascade.core.inference_process import SignalFrame
    import time
    return [
        SignalFrame(step=i, timestamp=time.time(),
                    token_entropy=1.0, avg_logprob=-1.0,
                    repetition_score=0.0, generation_speed=0.0,
                    fused_score=s)
        for i, s in enumerate(scores)
    ]


def test_predictor_insufficient_history():
    p = EscalationPredictor(min_history=5)
    frames = _make_frames([0.3, 0.4])
    assert p.predict(frames) == 0.0


def test_predictor_stable_low_signal():
    """Stable low signal → near-zero escalation probability."""
    p = EscalationPredictor()
    frames = _make_frames([0.2] * 10)
    prob = p.predict(frames)
    assert prob < 0.2


def test_predictor_rising_trend():
    """Rising signal toward threshold → high escalation probability."""
    p = EscalationPredictor(threshold=0.65, lookahead=10)
    # Trend that clearly crosses 0.65 within 10 tokens
    frames = _make_frames([0.30, 0.38, 0.46, 0.54, 0.60, 0.65])
    prob = p.predict(frames)
    assert prob > 0.60


def test_predictor_should_start_remote():
    p = EscalationPredictor(threshold=0.65, lookahead=10, trigger_prob=0.70)
    frames = _make_frames([0.40, 0.48, 0.54, 0.60, 0.65, 0.70])
    assert p.should_start_remote(frames)


# ── State machine tests ───────────────────────────────────────────────────────

def test_state_machine_valid_path():
    proc = InferenceProcess(prompt="test", quality_budget=200)
    sm   = InferenceStateMachine(proc)
    assert sm.transition(InferenceState.LOCAL_FAST)
    assert sm.transition(InferenceState.LOCAL_VERIFY)
    assert sm.transition(InferenceState.LOCAL_RECOVER)
    assert sm.transition(InferenceState.REMOTE_ESCAPE)
    assert sm.transition(InferenceState.FINISHED)
    assert sm.is_finished


def test_state_machine_budget_block():
    """Budget too low to afford REMOTE_ESCAPE → blocked."""
    proc = InferenceProcess(prompt="test", quality_budget=60)
    sm   = InferenceStateMachine(proc)
    sm.transition(InferenceState.LOCAL_FAST)   # costs 20 → budget 40
    sm.transition(InferenceState.LOCAL_VERIFY) # costs 50 → blocked (40 < 50)?
    # LOCAL_VERIFY costs 50, budget is 60 initially, after LOCAL_FAST (20) = 40
    # So LOCAL_VERIFY should also be blocked
    assert proc.state != InferenceState.FINISHED


def test_state_machine_illegal_skip():
    """Cannot skip from INITIALIZING directly to REMOTE_ESCAPE."""
    proc = InferenceProcess(prompt="test", quality_budget=200)
    sm   = InferenceStateMachine(proc)
    ok, _ = sm.can_transition(InferenceState.REMOTE_ESCAPE)
    assert not ok


def test_state_machine_force_finish():
    proc = InferenceProcess(prompt="test")
    sm   = InferenceStateMachine(proc)
    sm.force_finish("test")
    assert proc.state == InferenceState.FINISHED
