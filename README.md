# Cascade: A Closed-Loop Runtime Inference Controller for Cost-Adaptive LLM Serving

> **AMD Developer Hackathon ACT II — Track 1 + Track 3 Submission**
> Built on AMD Developer Cloud · ROCm 6.2 · Fireworks AI API

---

## What is Cascade?

Cascade is **not a router**.

Every existing LLM cost-optimization system makes one decision — before inference begins:  
*"Should this query go to the local model or the cloud API?"*

Cascade makes continuous decisions **during** decoding, observing five runtime signals simultaneously,  
projecting trajectories forward, and launching a cloud model **before the local model fails** —  
so both run in parallel and neither GPU sits idle.

This is **closed-loop inference control**, not routing.

---

## The Core Innovation

### Traditional system
```
Prompt → [decide] → Model → Response
            ↑
      one decision, before any work
```

### Cascade
```
Prompt → Local starts immediately (Q2_K, AMD ROCm)
              ↓ (every 10 tokens)
         Observe: entropy + confidence + repetition + speed + classifier
              ↓
         EscalationPredictor: project trajectory → P(escalation in 20 tokens)
              ↓ (if P ≥ 70%)
         Remote starts in parallel (Fireworks AI)
         Local continues (no idle GPU)
              ↓
         Race merge: best quality wins
              ↓
         Response + cost log + token waste metric
```

The decision is not binary and not one-shot. The controller observes, predicts, and adapts  
every 10 tokens throughout the full decoding loop.

---

## Architecture

```
cascade/
├── core/
│   ├── inference_process.py    ← New primitive: InferenceProcess (like a Unix process)
│   ├── state_machine.py        ← Formal FSM: INIT→Q2→Q4→Q8→REMOTE→DONE
│   └── runtime_controller.py  ← OS kernel: closed-loop decoding control
│
├── signals/
│   ├── entropy.py              ← Shannon entropy of next-token distribution
│   ├── confidence.py           ← Per-token log-probability signals
│   ├── repetition.py           ← N-gram repetition detection
│   ├── speed.py                ← Tokens/sec (KV cache pressure proxy)
│   └── fusion.py               ← Weighted multi-objective signal combiner
│
├── policy/
│   ├── quality_budget.py       ← Abstract budget accounting
│   ├── escalation_predictor.py ← Linear trend projection + sigmoid smoothing
│   └── online_learning.py      ← Domain-aware policy that evolves per request
│
├── execution/
│   ├── local/
│   │   ├── engine.py           ← vLLM + AMD ROCm runner
│   │   └── quant_selector.py  ← Q2_K / Q4_K_M / Q8_0 dispatch
│   └── remote/
│       ├── fireworks.py        ← Async Fireworks AI client with draft injection
│       └── parallel_merge.py  ← Speculative parallel executor + race merge
│
├── dashboard/                  ← Real-time WebSocket dashboard
├── eval/benchmark.py           ← Cascade vs all-remote vs all-local benchmarks
├── Dockerfile
└── docker-compose.yml
```

---

## The New Primitive: `InferenceProcess`

```python
@dataclass
class InferenceProcess:
    prompt:            str
    quality_budget:    int    = 100    # Abstract spend units
    latency_budget_ms: int    = 5_000  # Hard deadline
    cost_budget_usd:   float  = 0.005  # Per-request API ceiling

    state:          InferenceState    # Managed by StateMachine
    signal_history: list[SignalFrame] # Every 10 tokens
    tokens_local:   int               # AMD GPU tokens
    tokens_remote:  int               # Fireworks AI tokens
    tokens_wasted:  int               # Local tokens discarded on escalation
    route_taken:    str
    cost_actual_usd: float
```

Like a Unix process, the InferenceProcess carries **state, budgets, and history** through its full lifecycle.  
The RuntimeController reads and mutates it continuously — not once at dispatch time.

---

## Runtime State Machine

```
INITIALIZING → LOCAL_FAST (Q2_K)
                    ↓ fused_score > 0.42
               LOCAL_VERIFY (Q4_K_M)
                    ↓ fused_score > 0.55
               LOCAL_RECOVER (Q8_0)
                    ↓ P(escalation) > 70%  ← EscalationPredictor fires
               REMOTE_ESCAPE  ←→  LOCAL continues in parallel
                    ↓
                FINISHED (race winner selected)
```

Every transition is guarded by the `InferenceStateMachine`: budget check → latency check → legal path.  
No transition bypasses the guard. This is a formal state machine, not an if-else chain.

---

## Multi-Signal Fusion

```python
fused_score = (
    0.35 * normalize(entropy)       +   # Primary: next-token distribution uncertainty
    0.25 * (1 - confidence)         +   # Token-level probability
    0.15 * repetition_score         +   # Degenerate generation detection
    0.10 * speed_signal             +   # KV cache / memory pressure
    0.15 * classifier_prior             # Query-level pre-classification
)
```

`fused_score ∈ [0, 1]` where `0 = confident local`, `1 = must escalate`.

---

## Speculative Parallel Execution

When `P(escalation) ≥ 70%`:

```python
local_task  = asyncio.create_task(local_engine.continue_stream())
remote_task = asyncio.create_task(fireworks.generate(draft_prefix=local_draft))

done, pending = await asyncio.wait([local_task, remote_task],
                                    return_when=FIRST_COMPLETED)
# Winner: first to pass quality_threshold
# Loser: task.cancel() — clean, no resource leak
```

- **Local draft injection**: the local model's partial output is sent as a partial assistant  
  message to Fireworks AI, reducing remote completion tokens needed.
- **No idle GPU**: AMD GPU runs Q8_0 inference while Fireworks AI generates.  
  Cloud latency overlaps local decoding.

---

## Quality Budget

Each request gets a budget (default: 100). Every state transition spends units:

| State | Spend | Meaning |
|-------|-------|---------|
| LOCAL_FAST (Q2_K) | 20 | Cheap, exploratory |
| LOCAL_VERIFY (Q4_K_M) | 50 | Moderate |
| LOCAL_RECOVER (Q8_0) | 80 | Premium local |
| REMOTE_ESCAPE | 200 | Cloud — beyond default budget |

A request with `quality_budget=60` can never reach REMOTE_ESCAPE — it stays local.  
A request with `quality_budget=200` can escalate freely. The budget is a **product-level control knob**.

---

## Online Learning

After each request, the router learns:

```python
learner.record(domain="legal", local_won=False, quality=0.42)
# → legal: local_success_rate drops → recommended_budget rises
# → next legal query: classifier_prior set higher → escalation fires earlier
```

No retraining. No model updates. Pure online policy adaptation.

---

## Live Dashboard

Real-time dashboard at `http://localhost:8000/`:

- **State badge** — live FSM state with color coding
- **Confidence heatmap** — per-token log-prob colored green → red
- **Escalation predictor bar** — rising probability curve
- **Signal gauges** — entropy, confidence, repetition, speed, fused score
- **Token waste tracker** — wasted vs reused vs total
- **Cost savings counter** — cumulative savings vs all-remote baseline

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Routing accuracy | ≥ 85% correct local/remote decisions |
| Cost vs all-remote | 60–80% savings on mixed workloads |
| Token waste rate | < 15% on escalated queries |
| Local route rate | 60–70% on standard benchmark tasks |

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/YOUR_HANDLE/cascade
cd cascade
cp .env.example .env
# Add FIREWORKS_API_KEY to .env

# 2. Run with Docker (AMD GPU)
docker-compose up --build

# 3. Dashboard
open http://localhost:8000

# 4. Benchmark
docker exec cascade-runtime python eval/benchmark.py

# 5. API
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformer attention", "quality_budget": 100}'
```

---

## AMD Platform Integration

| Component | AMD Technology |
|-----------|---------------|
| Local inference | AMD MI300X / RX 7900 via ROCm 6.2 |
| Model serving | vLLM with ROCm PyTorch backend |
| Quantization | Q2_K / Q4_K_M / Q8_0 on AMD GPU |
| Training baseline | AMD Developer Cloud |

---

## Why This Wins

**Routing is solved. Runtime inference control is not.**

Most hackathon submissions will build a router with a confidence threshold.  
Cascade introduces a new systems abstraction — the **InferenceProcess** — and a new execution model  
— **speculative parallel execution** — that makes fundamentally different engineering tradeoffs:

- Predictive rather than reactive
- Continuous rather than one-shot  
- Concurrent rather than sequential
- Measured by token waste, not just accuracy

These are the primitives that production LLM serving systems will converge on.  
Cascade builds them today, on AMD hardware, with real benchmarks.

---

## License

MIT — built for AMD Developer Hackathon ACT II.
