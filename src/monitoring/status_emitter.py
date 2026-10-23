"""Writes the bot's runtime state to status.json for the read-only dashboard.

Atomic (tmp + rename) so a concurrently reading dashboard never sees partial
JSON, and throttled so the trading loop never spends meaningful time on I/O.
The dashboard treats a stale file (old ``ts``) as "bot down" — that behavior is
the liveness signal, so the emitter must never fake freshness.
"""
import json
import os
import threading
import time


class StatusEmitter:
    def __init__(self, path: str = "status.json", min_interval: float = 2.0):
        self.path = path
        self.min_interval = min_interval
        self._state: dict = {}
        self._lock = threading.Lock()
        self._last_write = 0.0

    def update(self, force: bool = False, **fields) -> None:
        """Merge fields into the state and write if the throttle allows.
        force=True bypasses the throttle (use for settlements/halts)."""
        with self._lock:
            self._state.update(fields)
            now = time.time()
            if not force and now - self._last_write < self.min_interval:
                return
            self._last_write = now
            self._state["ts"] = now
            tmp = f"{self.path}.tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(self._state, f)
                os.replace(tmp, self.path)
            except OSError:
                # Status emission must never break the trading loop
                pass
