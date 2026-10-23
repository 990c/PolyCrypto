import csv
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from monitoring.dashboard import build_layout, load_status, tail_log, trade_stats
from monitoring.status_emitter import StatusEmitter


@pytest.mark.unit
class TestStatusEmitter:
    def test_writes_valid_json_atomically(self, tmp_path):
        path = str(tmp_path / "status.json")
        em = StatusEmitter(path, min_interval=0)
        em.update(mode="PAPER", prob_up=0.42)
        data = json.load(open(path))
        assert data["mode"] == "PAPER"
        assert data["prob_up"] == 0.42
        assert "ts" in data
        assert not os.path.exists(path + ".tmp")

    def test_throttles_writes(self, tmp_path):
        path = str(tmp_path / "status.json")
        em = StatusEmitter(path, min_interval=60)
        em.update(a=1)
        first_mtime = os.path.getmtime(path)
        em.update(a=2)  # throttled — no write
        assert os.path.getmtime(path) == first_mtime
        assert json.load(open(path))["a"] == 1

    def test_force_bypasses_throttle(self, tmp_path):
        path = str(tmp_path / "status.json")
        em = StatusEmitter(path, min_interval=60)
        em.update(a=1)
        em.update(force=True, a=2)
        assert json.load(open(path))["a"] == 2

    def test_state_merges_across_updates(self, tmp_path):
        path = str(tmp_path / "status.json")
        em = StatusEmitter(path, min_interval=0)
        em.update(mode="PAPER")
        em.update(prob_up=0.3)
        data = json.load(open(path))
        assert data["mode"] == "PAPER" and data["prob_up"] == 0.3

    def test_write_failure_does_not_raise(self):
        em = StatusEmitter("/nonexistent-dir/status.json", min_interval=0)
        em.update(a=1)  # must not raise — trading loop safety


@pytest.mark.unit
class TestLoadStatus:
    def test_missing_file(self, tmp_path):
        status, age = load_status(str(tmp_path / "nope.json"))
        assert status is None and age is None

    def test_fresh_and_stale_age(self, tmp_path):
        path = str(tmp_path / "status.json")
        json.dump({"ts": time.time() - 30, "mode": "PAPER"}, open(path, "w"))
        status, age = load_status(path)
        assert status["mode"] == "PAPER"
        assert 29 < age < 35

    def test_corrupt_json(self, tmp_path):
        path = str(tmp_path / "status.json")
        open(path, "w").write("{not json")
        status, age = load_status(path)
        assert status is None


@pytest.mark.unit
class TestTradeStats:
    HEADER = ["trade_id", "timestamp", "market_id", "action", "entry_price",
              "size_usd", "tokens", "prob_up", "ev", "kelly_fraction",
              "outcome", "pnl", "balance_after", "cumulative_pnl", "settlement_source"]

    def _write(self, tmp_path, rows):
        path = str(tmp_path / "trades.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.HEADER)
            w.writerows(rows)
        return path

    def test_win_loss_pnl_recompute(self, tmp_path):
        now = time.time()
        path = self._write(tmp_path, [
            ["1", now, "m1", "BUY_DOWN", "0.5000", "10.00", "20", "0.3", "0.1", "", "PENDING", "", "90", "", ""],
            ["", now + 1, "m1", "SETTLE_DOWN", "", "", "20", "", "", "", "WIN", "20", "110", "10", "resolution"],
            ["2", now + 2, "m2", "BUY_UP", "0.2500", "10.00", "40", "0.7", "0.2", "", "PENDING", "", "100", "", ""],
            ["", now + 3, "m2", "SETTLE_UP", "", "", "40", "", "", "", "LOSS", "0", "100", "0", "heuristic"],
        ])
        s = trade_stats(path, session_start_ts=0)
        assert s["n"] == 2 and s["wins"] == 1
        assert s["pnl"] == pytest.approx(10.0 - 10.0)  # +$10 win, -$10 loss
        assert s["source_resolution"] == 1 and s["source_heuristic"] == 1
        assert s["last_trades"][0]["side"] == "UP"  # newest first

    def test_session_split(self, tmp_path):
        old, new = time.time() - 7200, time.time()
        path = self._write(tmp_path, [
            ["1", old, "m1", "BUY_DOWN", "0.5", "10", "20", "0.3", "0.1", "", "PENDING", "", "90", "", ""],
            ["", old + 1, "m1", "SETTLE_DOWN", "", "", "20", "", "", "", "WIN", "20", "110", "10", "resolution"],
            ["2", new, "m2", "BUY_DOWN", "0.5", "10", "20", "0.3", "0.1", "", "PENDING", "", "100", "", ""],
            ["", new + 1, "m2", "SETTLE_DOWN", "", "", "20", "", "", "", "WIN", "20", "120", "20", "resolution"],
        ])
        s = trade_stats(path, session_start_ts=new - 60)
        assert s["n"] == 2 and s["session_n"] == 1

    def test_pending_trades_excluded(self, tmp_path):
        path = self._write(tmp_path, [
            ["1", time.time(), "m1", "BUY_DOWN", "0.5", "10", "20", "0.3", "0.1", "", "PENDING", "", "90", "", ""],
        ])
        s = trade_stats(path, session_start_ts=0)
        assert s["n"] == 0

    def test_missing_file(self, tmp_path):
        s = trade_stats(str(tmp_path / "none.csv"))
        assert s["n"] == 0 and s["last_trades"] == []


@pytest.mark.unit
class TestTailLog:
    def test_filters_actionable_lines(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "bot_2026-07-09.log").write_text(
            "2026-07-09 10:00:00 | DEBUG | noise line\n"
            "2026-07-09 10:00:01 | INFO | Settled: DOWN | epoch PnL +$5\n"
            "2026-07-09 10:00:02 | WARNING | LIVE FOK rejected: BUY_UP @ 0.26\n"
            "2026-07-09 10:00:03 | INFO | routine tick\n"
        )
        lines = tail_log(str(logs))
        assert len(lines) == 2
        assert "Settled" in lines[0] and "rejected" in lines[1]

    def test_no_logs_dir(self, tmp_path):
        assert tail_log(str(tmp_path / "nope")) == []


@pytest.mark.unit
class TestRenderSmoke:
    def test_renders_with_live_status(self):
        from rich.console import Console
        status = {
            "ts": time.time(), "mode": "LIVE", "model_version": "v2",
            "prob_up": 0.31, "gate_state": "eligible", "best_action": "BUY_DOWN",
            "ev": 0.082, "target_price": 0.41, "up_bid": 0.58, "up_ask": 0.60,
            "epoch_end": time.time() + 120, "epoch_traded": False,
            "bankroll": 104.32, "wallet_usdc": 104.0, "binance_lag_ms": 42,
            "poly_connected": True, "exposure": 10, "exposure_cap": 20,
            "drawdown_pct": 3.1, "drawdown_limit_pct": 20,
            "consec_losses": 1, "tilt_threshold": 5, "spend_24h": 30, "spend_cap": 50,
        }
        stats = {"n": 2, "wins": 1, "pnl": 5.0, "staked": 20.0,
                 "session_n": 1, "session_wins": 1, "session_pnl": 10.0,
                 "last_trades": [{"time": "23:51", "side": "DOWN", "price": 0.37,
                                  "size": 10.0, "outcome": "WIN"}],
                 "source_resolution": 20, "source_heuristic": 1}
        layout = build_layout(status, 1.0, stats, ["INFO | Settled: DOWN"])
        out = Console(width=120, record=True)
        out.print(layout)
        text = out.export_text()
        assert "PolyCrypto" in text and "LIVE" in text and "BUY_DOWN" in text

    def test_renders_stale_without_status(self):
        from rich.console import Console
        stats = {"n": 0, "wins": 0, "pnl": 0.0, "staked": 0.0,
                 "session_n": 0, "session_wins": 0, "session_pnl": 0.0,
                 "last_trades": [], "source_resolution": 0, "source_heuristic": 0}
        layout = build_layout(None, None, stats, [])
        out = Console(width=120, record=True)
        out.print(layout)
        assert "STALE" in out.export_text()
