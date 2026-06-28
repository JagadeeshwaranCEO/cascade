"""
Cascade — Runtime Adaptive Inference Orchestrator
FastAPI server: REST API + live WebSocket dashboard.

Endpoints:
    POST /infer          — Run an InferenceProcess, stream events over WS
    GET  /health         — Liveness check
    GET  /metrics        — Aggregate runtime statistics
    GET  /learner        — Domain routing policy state
    WS   /ws             — Real-time dashboard WebSocket
    GET  /               — Live dashboard UI
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from cascade.core.inference_process import InferenceProcess
from cascade.core.runtime_controller import RuntimeController
from cascade.execution.local.engine import LocalInferenceEngine
from cascade.execution.remote.fireworks import FireworksClient
from cascade.policy.online_learning import OnlineLearner
from cascade.dashboard.websocket import manager, broadcast_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("cascade.main")

# ── Global state ──────────────────────────────────────────────────────────────
local_engine: Optional[LocalInferenceEngine] = None
learner       = OnlineLearner()
_metrics      = {"total_requests": 0, "total_saved_usd": 0.0,
                 "total_wasted_tokens": 0, "local_wins": 0, "remote_wins": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global local_engine
    log.info("Cascade starting — loading local model...")
    local_engine = LocalInferenceEngine()
    await local_engine.load(quants=["q2_k", "q4_k_m", "q8_0"])
    log.info("Cascade ready.")
    yield
    log.info("Cascade shutting down.")


app = FastAPI(
    title="Cascade",
    description="Runtime Adaptive Inference Orchestrator — AMD Developer Hackathon ACT II",
    version="1.0.0",
    lifespan=lifespan,
)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(ws)


# ── REST API ──────────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    prompt:            str
    quality_budget:    int   = Field(100, ge=20, le=200)
    latency_budget_ms: int   = Field(5_000, ge=500)
    cost_budget_usd:   float = Field(0.005, ge=0.0)


@app.post("/infer")
async def infer(req: InferRequest):
    """
    Run a prompt through the Cascade runtime inference controller.
    Returns the completed InferenceProcess as JSON.
    Events are pushed in real-time over /ws for the dashboard.
    """
    global _metrics
    process = InferenceProcess(
        prompt=req.prompt,
        quality_budget=req.quality_budget,
        latency_budget_ms=req.latency_budget_ms,
        cost_budget_usd=req.cost_budget_usd,
    )

    fw_key = os.environ.get("FIREWORKS_API_KEY")
    fw_client = FireworksClient(api_key=fw_key) if fw_key else None

    controller = RuntimeController(
        local_engine=local_engine,
        fireworks_client=fw_client,
        learner=learner,
        broadcast_fn=broadcast_fn,
    )

    result = await controller.run(process)

    # Update aggregate metrics
    _metrics["total_requests"] += 1
    _metrics["total_wasted_tokens"] += result.tokens_wasted
    if "remote" in result.route_taken:
        _metrics["remote_wins"] += 1
        # All-remote baseline cost estimate (Fireworks for all tokens)
        all_remote_cost = ((result.tokens_local + result.tokens_remote) / 1000) * 0.0009
        saved = max(0.0, all_remote_cost - result.cost_actual_usd)
        _metrics["total_saved_usd"] += saved
    else:
        _metrics["local_wins"] += 1
        # Full inference on remote would have cost:
        all_remote_cost = (result.tokens_local / 1000) * 0.0009
        _metrics["total_saved_usd"] += all_remote_cost

    return result.to_dict()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "cascade", "ts": time.time()}


@app.get("/metrics")
async def metrics():
    total = _metrics["total_requests"]
    return {
        **_metrics,
        "local_route_pct":  round(_metrics["local_wins"]  / max(total, 1) * 100, 1),
        "remote_route_pct": round(_metrics["remote_wins"] / max(total, 1) * 100, 1),
        "avg_saved_per_req": round(_metrics["total_saved_usd"] / max(total, 1), 6),
        "domain_policy":    learner.domain_summary(),
    }


@app.get("/learner")
async def learner_state():
    return learner.domain_summary()


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


# ── Live Dashboard HTML ───────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cascade — Runtime Inference Controller</title>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --green: #3fb950;
    --yellow: #d29922; --orange: #f0883e; --red: #f85149;
    --blue: #58a6ff; --purple: #bc8cff; --teal: #39d353;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }
  header { padding: 16px 24px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.25rem; font-weight: 700; }
  header .badge { background: var(--teal); color: #000; font-size: 0.7rem;
                  font-weight: 700; padding: 2px 8px; border-radius: 12px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; padding: 20px; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 10px; padding: 18px; }
  .card h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em;
             color: var(--muted); margin-bottom: 14px; }
  .stat-row { display: flex; justify-content: space-between; margin: 6px 0; font-size: 0.9rem; }
  .stat-val { font-weight: 600; font-variant-numeric: tabular-nums; }
  .green { color: var(--green); } .yellow { color: var(--yellow); }
  .orange { color: var(--orange); } .red { color: var(--red); } .blue { color: var(--blue); }

  /* State badge */
  .state-badge { display: inline-block; padding: 4px 14px; border-radius: 20px;
                 font-weight: 700; font-size: 0.85rem; margin-bottom: 12px; }
  .state-INITIALIZING  { background: #21262d; color: var(--muted); }
  .state-LOCAL_FAST    { background: #0d2818; color: var(--green); }
  .state-LOCAL_VERIFY  { background: #2e2008; color: var(--yellow); }
  .state-LOCAL_RECOVER { background: #2e1308; color: var(--orange); }
  .state-REMOTE_ESCAPE { background: #2e0808; color: var(--red); }
  .state-FINISHED      { background: #0a1929; color: var(--blue); }

  /* Confidence heatmap */
  #heatmap { display: flex; flex-wrap: wrap; gap: 3px; min-height: 48px; }
  .hmap-token { display: inline-block; padding: 2px 4px; border-radius: 3px;
                font-size: 0.75rem; font-family: monospace; cursor: default; }

  /* Escalation bar */
  .bar-track { background: #21262d; border-radius: 6px; height: 16px;
               overflow: hidden; margin: 8px 0; }
  .bar-fill  { height: 100%; border-radius: 6px; transition: width 0.3s ease,
               background 0.3s ease; }

  /* Signal gauges */
  .gauge { margin: 8px 0; }
  .gauge-label { font-size: 0.78rem; color: var(--muted); display: flex;
                 justify-content: space-between; }
  .gauge-track { background: #21262d; border-radius: 4px; height: 6px; margin-top: 4px; }
  .gauge-fill  { height: 100%; border-radius: 4px; transition: width 0.2s ease; }

  /* Prompt box */
  #prompt-input { width: 100%; background: #21262d; border: 1px solid var(--border);
                  border-radius: 8px; padding: 10px 14px; color: var(--text);
                  font-size: 0.9rem; resize: vertical; min-height: 80px; }
  #run-btn { margin-top: 10px; width: 100%; padding: 10px; background: var(--blue);
             color: #000; font-weight: 700; border: none; border-radius: 8px;
             cursor: pointer; font-size: 0.9rem; transition: opacity 0.2s; }
  #run-btn:hover { opacity: 0.85; } #run-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Response output */
  #response-box { background: #21262d; border-radius: 8px; padding: 14px;
                  font-family: monospace; font-size: 0.82rem; line-height: 1.6;
                  min-height: 120px; max-height: 300px; overflow-y: auto;
                  white-space: pre-wrap; word-break: break-word; color: var(--text); }

  /* Route trace */
  #route-trace { font-size: 0.78rem; color: var(--muted); margin-top: 10px; }

  /* Full-width bottom row */
  .full-width { grid-column: 1 / -1; }

  /* Savings ticker */
  .savings-big { font-size: 2rem; font-weight: 800; color: var(--green); margin: 8px 0; }
  .savings-sub { font-size: 0.8rem; color: var(--muted); }

  /* Log */
  #event-log { font-family: monospace; font-size: 0.75rem; color: var(--muted);
               max-height: 200px; overflow-y: auto; }
  .log-entry { padding: 2px 0; border-bottom: 1px solid #21262d; }
  .log-entry .ts { color: #484f58; margin-right: 8px; }
</style>
</head>
<body>
<header>
  <h1>⚡ Cascade</h1>
  <span class="badge">LIVE</span>
  <span style="margin-left:auto; font-size:0.8rem; color:var(--muted)">
    Runtime Adaptive Inference Orchestrator · AMD ROCm + Fireworks AI
  </span>
</header>

<div class="grid">

  <!-- Col 1: State + Signal gauges -->
  <div class="card">
    <h2>Runtime State</h2>
    <div id="state-badge" class="state-badge state-INITIALIZING">INITIALIZING</div>
    <div class="stat-row"><span>Process ID</span><span class="stat-val blue" id="pid">—</span></div>
    <div class="stat-row"><span>Latency</span><span class="stat-val" id="latency">—</span></div>
    <div class="stat-row"><span>Tokens local</span><span class="stat-val green" id="tok-local">0</span></div>
    <div class="stat-row"><span>Tokens remote</span><span class="stat-val orange" id="tok-remote">0</span></div>
    <div class="stat-row"><span>Tokens wasted</span><span class="stat-val red" id="tok-waste">0</span></div>
    <div class="stat-row"><span>Waste %</span><span class="stat-val" id="waste-pct">0%</span></div>
    <div class="stat-row"><span>Cost</span><span class="stat-val" id="cost">$0.000000</span></div>
    <div class="stat-row"><span>Budget left</span><span class="stat-val" id="budget">100</span></div>
  </div>

  <!-- Col 2: Signal gauges + escalation -->
  <div class="card">
    <h2>Signal Monitor</h2>
    <div class="gauge">
      <div class="gauge-label"><span>Entropy</span><span id="g-entropy">0.00</span></div>
      <div class="gauge-track"><div class="gauge-fill" id="gf-entropy" style="width:0%;background:var(--green)"></div></div>
    </div>
    <div class="gauge">
      <div class="gauge-label"><span>Confidence</span><span id="g-conf">0.00</span></div>
      <div class="gauge-track"><div class="gauge-fill" id="gf-conf" style="width:0%;background:var(--green)"></div></div>
    </div>
    <div class="gauge">
      <div class="gauge-label"><span>Repetition</span><span id="g-rep">0.00</span></div>
      <div class="gauge-track"><div class="gauge-fill" id="gf-rep" style="width:0%;background:var(--green)"></div></div>
    </div>
    <div class="gauge">
      <div class="gauge-label"><span>Speed</span><span id="g-speed">0.00</span></div>
      <div class="gauge-track"><div class="gauge-fill" id="gf-speed" style="width:0%;background:var(--green)"></div></div>
    </div>
    <h2 style="margin-top:16px">Escalation Predictor</h2>
    <div id="esc-pct" style="font-size:1.4rem;font-weight:800;color:var(--green)">0%</div>
    <div class="bar-track">
      <div class="bar-fill" id="esc-bar" style="width:0%;background:var(--green)"></div>
    </div>
    <div style="font-size:0.75rem;color:var(--muted)">
      Trigger at 70% — remote starts in parallel
    </div>
    <div class="stat-row" style="margin-top:10px">
      <span>Fused score</span><span class="stat-val" id="fused-score">0.000</span>
    </div>
    <div class="stat-row">
      <span>Zone</span><span class="stat-val" id="zone">—</span>
    </div>
  </div>

  <!-- Col 3: Prompt + response -->
  <div class="card">
    <h2>Inference</h2>
    <textarea id="prompt-input" placeholder="Enter a prompt…">Explain the difference between transformer self-attention and cross-attention mechanisms.</textarea>
    <button id="run-btn" onclick="runInference()">▶ Run Inference</button>
    <h2 style="margin-top:14px">Response</h2>
    <div id="response-box">Ready.</div>
    <div id="route-trace"></div>
  </div>

  <!-- Full row: Confidence heatmap -->
  <div class="card full-width">
    <h2>Confidence Heatmap — Per-Token Signal</h2>
    <div id="heatmap"></div>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:8px">
      🟢 &gt;0.85  🟡 0.65–0.85  🟠 0.45–0.65  🔴 &lt;0.45
    </div>
  </div>

  <!-- Savings + aggregate metrics -->
  <div class="card">
    <h2>Cost Savings vs All-Remote</h2>
    <div class="savings-big" id="savings">$0.000000</div>
    <div class="savings-sub">Cumulative savings this session</div>
    <div class="stat-row" style="margin-top:16px"><span>Total requests</span><span class="stat-val" id="m-total">0</span></div>
    <div class="stat-row"><span>Local route %</span><span class="stat-val green" id="m-local-pct">—</span></div>
    <div class="stat-row"><span>Remote route %</span><span class="stat-val orange" id="m-remote-pct">—</span></div>
  </div>

  <!-- Event log -->
  <div class="card" style="grid-column: 2 / -1;">
    <h2>Event Log</h2>
    <div id="event-log"></div>
  </div>

</div>

<script>
const WS_URL = `ws://${location.host}/ws`;
let ws, totalSaved = 0;

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onmessage = e => handleEvent(JSON.parse(e.data));
  ws.onclose   = () => setTimeout(connect, 2000);
}

function handleEvent(data) {
  const p = data.process || {};
  const ev = data.event;

  // State badge
  if (p.state) {
    const badge = document.getElementById('state-badge');
    badge.textContent = p.state;
    badge.className = `state-badge state-${p.state}`;
  }

  // Process stats
  setVal('pid',        p.process_id  || '—');
  setVal('tok-local',  p.tokens_local ?? 0);
  setVal('tok-remote', p.tokens_remote ?? 0);
  setVal('tok-waste',  p.tokens_wasted ?? 0);
  setVal('waste-pct',  (p.waste_pct ?? 0) + '%');
  setVal('cost',       '$' + (p.cost_usd ?? 0).toFixed(6));
  setVal('budget',     p.budget_remaining ?? '—');
  if (p.latency_ms) setVal('latency', p.latency_ms.toFixed(0) + 'ms');

  // Signals
  if (data.signal) {
    const s = data.signal;
    updateGauge('entropy', s.token_entropy ?? 0, 4);
    updateGauge('conf',    1 - (s.avg_logprob ?? 0), 1, true);
    updateGauge('rep',     s.repetition_score ?? 0, 1);
    updateGauge('speed',   s.generation_speed ?? 0, 1);
    setVal('fused-score', (s.fused_score ?? 0).toFixed(3));
    const zone = data.zone || '—';
    const el = document.getElementById('zone');
    el.textContent = zone;
    el.className = 'stat-val ' + {GREEN:'green',YELLOW:'yellow',ORANGE:'orange',RED:'red'}[zone] || '';

    // Heatmap token
    if (data.token) addHeatmapToken(data.token, s.avg_logprob ?? -1.5);
  }

  // Escalation bar
  if (data.escalation_prob !== undefined) {
    const pct = Math.round(data.escalation_prob * 100);
    const bar = document.getElementById('esc-bar');
    const lbl = document.getElementById('esc-pct');
    lbl.textContent = pct + '%';
    bar.style.width = pct + '%';
    const color = pct < 40 ? 'var(--green)' : pct < 70 ? 'var(--yellow)' : 'var(--red)';
    bar.style.background = color;
    lbl.style.color = color;
  }

  // Response
  if (p.state === 'FINISHED' && ev === 'finished') {
    document.getElementById('run-btn').disabled = false;
    document.getElementById('run-btn').textContent = '▶ Run Inference';
    document.getElementById('route-trace').textContent =
      `Route: ${p.route_taken || '—'} · Quality: ${p.quality_score ?? '—'} · Latency: ${(p.latency_ms||0).toFixed(0)}ms`;
    refreshMetrics();
  }

  // Log
  addLog(ev, p.state || '');
}

function updateGauge(id, val, maxVal, invert = false) {
  const pct = Math.min(100, (val / maxVal) * 100);
  const fill = document.getElementById('gf-' + id);
  if (!fill) return;
  fill.style.width = pct + '%';
  const stress = invert ? (100 - pct) / 100 : pct / 100;
  fill.style.background = stress < 0.35 ? 'var(--green)'
    : stress < 0.60 ? 'var(--yellow)'
    : stress < 0.80 ? 'var(--orange)' : 'var(--red)';
  const lbl = document.getElementById('g-' + id);
  if (lbl) lbl.textContent = val.toFixed(2);
}

function addHeatmapToken(token, logprob) {
  const conf = Math.exp(Math.max(logprob, -20));
  const color = conf > 0.85 ? '#1a4d1a' : conf > 0.65 ? '#4d3300' : conf > 0.45 ? '#4d1a00' : '#4d0000';
  const textColor = conf > 0.85 ? 'var(--green)' : conf > 0.65 ? 'var(--yellow)' : conf > 0.45 ? 'var(--orange)' : 'var(--red)';
  const el = document.createElement('span');
  el.className = 'hmap-token';
  el.textContent = token;
  el.style.background = color;
  el.style.color = textColor;
  el.title = `conf: ${conf.toFixed(3)}`;
  document.getElementById('heatmap').appendChild(el);
}

function addLog(event, state) {
  const log = document.getElementById('event-log');
  const d = document.createElement('div');
  d.className = 'log-entry';
  d.innerHTML = `<span class="ts">${new Date().toLocaleTimeString()}</span>${event} ${state ? '→ ' + state : ''}`;
  log.prepend(d);
  if (log.children.length > 80) log.removeChild(log.lastChild);
}

async function runInference() {
  const prompt = document.getElementById('prompt-input').value.trim();
  if (!prompt) return;
  document.getElementById('run-btn').disabled = true;
  document.getElementById('run-btn').textContent = '⏳ Running…';
  document.getElementById('heatmap').innerHTML = '';
  document.getElementById('response-box').textContent = 'Running…';
  document.getElementById('route-trace').textContent = '';

  try {
    const resp = await fetch('/infer', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, quality_budget: 100}),
    });
    const data = await resp.json();
    document.getElementById('response-box').textContent =
      data.response || '(no response)';
  } catch(e) {
    document.getElementById('response-box').textContent = 'Error: ' + e.message;
    document.getElementById('run-btn').disabled = false;
    document.getElementById('run-btn').textContent = '▶ Run Inference';
  }
}

async function refreshMetrics() {
  try {
    const r = await fetch('/metrics');
    const m = await r.json();
    setVal('m-total',       m.total_requests);
    setVal('m-local-pct',   m.local_route_pct  + '%');
    setVal('m-remote-pct',  m.remote_route_pct + '%');
    totalSaved = m.total_saved_usd || 0;
    document.getElementById('savings').textContent = '$' + totalSaved.toFixed(6);
  } catch(_) {}
}

function setVal(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}

connect();
refreshMetrics();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
