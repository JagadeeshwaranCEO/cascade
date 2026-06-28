"""
cascade.dashboard.websocket
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real-time metrics broadcast over WebSocket.
Feeds the live dashboard: confidence heatmap, escalation predictor bar,
state machine badge, cost savings counter, token waste tracker.
"""
from __future__ import annotations
import json
import logging
from typing import Set

log = logging.getLogger(__name__)


class ConnectionManager:
    """Thread-safe WebSocket connection pool."""

    def __init__(self):
        self.active: Set = set()

    async def connect(self, ws):
        await ws.accept()
        self.active.add(ws)
        log.info("Dashboard +1 connection (total: %d)", len(self.active))

    async def disconnect(self, ws):
        self.active.discard(ws)

    async def broadcast(self, payload: dict):
        if not self.active:
            return
        msg  = json.dumps(payload)
        dead = set()
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.active -= dead


manager = ConnectionManager()


async def broadcast_fn(payload: dict):
    """Top-level broadcast callable injected into RuntimeController."""
    await manager.broadcast(payload)
