"""Read-only terminal dashboard for the PolyCrypto bot.

Runs as a separate process; reads status.json (written by the bot's
StatusEmitter), trades.csv, and logs/bot-*.log. It can never affect the
trading loop — attach and detach freely, locally or over SSH.

Usage:
  python src/monitoring/dashboard.py [--dir /path/to/project] [--refresh 1.0]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

STALE_AFTER_SECONDS = 10
LOG_TAIL_LINES = 9
LOG_KEEP = re.compile(
    r"WARNING|ERROR|CRITICAL|FOK filled|FOK rejected|Settled:|Settlement source|"
    r"V2 prediction|Trade #|preflight|halting"
)


# --------------------------- data layer (pure, testable) ---------------------------

def load_status(path: str) -> tuple:
    """Returns (status_dict | None, age_seconds | None)."""
    try:
        with open(path) as f:
            status = json.load(f)
        age = time.time() - float(status.get("ts", 0))
        return status, age
    except (OSError, ValueError):
        return None, None


def trade_stats(path: str, session_start_ts: float = 0.0) -> dict:
    """Win rate and PnL recomputed from first principles (entry price + outcome),
    plus the settlement-source mix that determines label quality."""
    stats = {
        "n": 0, "wins": 0, "pnl": 0.0, "staked": 0.0,
        "session_n": 0, "session_wins": 0, "session_pnl": 0.0,
        "last_trades": [], "source_resolution": 0, "source_heuristic": 0,
    }
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        return stats

    for r in rows:
        if r.get("action", "").startswith("SETTLE"):
            src = r.get("settlement_source", "")
            if src == "resolution":
                stats["source_resolution"] += 1
            elif src == "heuristic":
                stats["source_heuristic"] += 1

    for i, r in enumerate(rows):
        if not r.get("action", "").startswith("BUY"):
            continue
        try:
            ts = float(r["timestamp"])
            price = float(r["entry_price"])
            size = float(r["size_usd"])
        except (ValueError, KeyError, TypeError):
            continue
        outcome = None
        for r2 in rows[i + 1:]:
            if r2.get("action", "").startswith("SETTLE") and r2.get("outcome") in ("WIN", "LOSS"):
                outcome = r2["outcome"]
                break
        if outcome is None:
            continue
        win = outcome == "WIN"
        pnl = size * (1 - price) / price if win else -size
        stats["n"] += 1
        stats["wins"] += int(win)
        stats["pnl"] += pnl
        stats["staked"] += size
        if ts >= session_start_ts:
            stats["session_n"] += 1
            stats["session_wins"] += int(win)
            stats["session_pnl"] += pnl
        stats["last_trades"].append({
            "time": datetime.fromtimestamp(ts).strftime("%H:%M"),
            "side": r["action"].replace("BUY_", ""),
            "price": price, "size": size, "outcome": outcome,
        })

    stats["last_trades"] = stats["last_trades"][-5:][::-1]
    return stats


def tail_log(logs_dir: str, n: int = LOG_TAIL_LINES) -> list:
    """Last n actionable lines from the newest bot log."""
    files = sorted(glob.glob(os.path.join(logs_dir, "bot_*.log")))
    if not files:
        return []
    try:
        with open(files[-1], errors="replace") as f:
            lines = f.readlines()[-3000:]
    except OSError:
        return []
    kept = [ln.rstrip("\n") for ln in lines if LOG_KEEP.search(ln)]
    return kept[-n:]


# --------------------------- render layer ---------------------------

def _bar(value: float, limit: float, width: int = 12) -> str:
    from rich.markup import escape
    frac = 0.0 if limit <= 0 else min(1.0, max(0.0, value / limit))
    filled = int(round(frac * width))
    color = "green" if frac < 0.6 else ("yellow" if frac < 0.85 else "red")
    return f"[{color}]{escape('█' * filled)}{escape('░' * (width - filled))}[/]"


def _fmt_money(v) -> str:
    return "—" if v is None else f"${v:,.2f}"


def build_layout(status, age, stats, log_lines):
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    stale = status is None or age is None or age > STALE_AFTER_SECONDS
    mode = (status or {}).get("mode", "?")
    mode_style = "bold white on red" if mode == "LIVE" else "bold white on blue"

    # ---- header ----
    header = Text()
    header.append(" PolyCrypto ", style="bold")
    header.append(f" {mode} ", style=mode_style)
    header.append(f"  model={(status or {}).get('model_version', '?')}")
    epoch_end = (status or {}).get("epoch_end") or 0
    left = max(0, epoch_end - time.time())
    header.append(f"  epoch ends {int(left // 60)}:{int(left % 60):02d}")
    if stale:
        shown = f"{age:.0f}s" if age is not None else "no status.json"
        header.append(f"  ● STALE ({shown}) — bot down?", style="bold red")
    else:
        lag = (status or {}).get("binance_lag_ms", 0) or 0
        lag_style = "green" if lag < 150 else ("yellow" if lag < 500 else "red")
        header.append("  ● live", style="bold green")
        header.append(f"  lag {lag:.0f}ms", style=lag_style)
        header.append(f"  poly {'✓' if status.get('poly_connected') else '✗'}")

    # ---- signal ----
    s = status or {}
    sig = Table.grid(padding=(0, 1))
    sig.add_column(style="dim", width=12)
    sig.add_column()
    prob = s.get("prob_up")
    sig.add_row("P(UP)", "—" if prob is None else f"{prob:.3f}")
    gate = s.get("gate_state", "—")
    gate_style = "green" if gate == "eligible" else "yellow"
    sig.add_row("state", f"[{gate_style}]{gate}[/]")
    act = s.get("best_action", "—")
    act_style = {"BUY_UP": "green", "BUY_DOWN": "red"}.get(act, "dim")
    ev = s.get("ev") or 0
    sig.add_row("action", f"[{act_style}]{act}[/] @ {s.get('target_price') or 0:.3f}  EV {ev * 100:.1f}%")
    bid, ask = s.get("up_bid"), s.get("up_ask")
    sig.add_row("market UP", f"{bid or 0:.2f} / {ask or 0:.2f}"
                             f"   DOWN ~{(1 - bid) if bid else 0:.2f}")
    sig.add_row("traded", "yes" if s.get("epoch_traded") else "no")

    # ---- risk ----
    risk = Table.grid(padding=(0, 1))
    risk.add_column(style="dim", width=12)
    risk.add_column()
    risk.add_row("exposure", f"{_bar(s.get('exposure') or 0, s.get('exposure_cap') or 1)} "
                             f"{_fmt_money(s.get('exposure') or 0)} / {_fmt_money(s.get('exposure_cap'))}")
    dd, ddl = s.get("drawdown_pct") or 0, s.get("drawdown_limit_pct") or 20
    risk.add_row("drawdown", f"{_bar(dd, ddl)} {dd:.1f}% / {ddl:.0f}% halt")
    cl, tt = s.get("consec_losses") or 0, s.get("tilt_threshold") or 5
    risk.add_row("tilt", f"{_bar(cl, tt)} {cl} / {tt} losses")
    if mode == "LIVE":
        risk.add_row("spend 24h", f"{_bar(s.get('spend_24h') or 0, s.get('spend_cap') or 1)} "
                                  f"{_fmt_money(s.get('spend_24h') or 0)} / {_fmt_money(s.get('spend_cap'))}")
    tot_src = stats["source_resolution"] + stats["source_heuristic"]
    if tot_src:
        heur_pct = stats["source_heuristic"] / tot_src * 100
        style = "green" if heur_pct <= 10 else "yellow"
        risk.add_row("settle src", f"[{style}]resolution {100 - heur_pct:.0f}% / heuristic {heur_pct:.0f}%[/]")

    # ---- money ----
    money = Table.grid(padding=(0, 1))
    money.add_column(style="dim", width=14)
    money.add_column()
    money.add_row("bankroll", _fmt_money(s.get("bankroll")))
    if mode == "LIVE":
        wallet = s.get("wallet_usdc")
        drift = None if (wallet is None or s.get("bankroll") is None) else wallet - s["bankroll"]
        money.add_row("wallet USDC", f"{_fmt_money(wallet)}"
                                     + (f"  drift {drift:+.2f}" if drift is not None else ""))
    wr = (stats["wins"] / stats["n"] * 100) if stats["n"] else 0
    money.add_row("session PnL", f"{stats['session_pnl']:+,.2f}  "
                                 f"({stats['session_wins']}/{stats['session_n']})")
    money.add_row("all-time", f"{stats['pnl']:+,.2f} on {_fmt_money(stats['staked'])} "
                              f"| WR {wr:.1f}% (n={stats['n']})")

    trades_tbl = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
    for col in ("time", "side", "px", "size", "out"):
        trades_tbl.add_column(col)
    for t in stats["last_trades"]:
        out_style = "green" if t["outcome"] == "WIN" else "red"
        side_style = "green" if t["side"] == "UP" else "red"
        trades_tbl.add_row(t["time"], f"[{side_style}]{t['side']}[/]", f"{t['price']:.2f}",
                           f"${t['size']:.0f}", f"[{out_style}]{t['outcome']}[/]")

    # ---- log ----
    log_text = Text()
    for ln in log_lines:
        style = ("red" if ("ERROR" in ln or "CRITICAL" in ln)
                 else "yellow" if ("WARNING" in ln or "rejected" in ln)
                 else "")
        log_text.append(ln[-160:] + "\n", style=style)
    if not log_lines:
        log_text.append("(no log lines yet — logs/bot_*.log)", style="dim")

    layout = Layout()
    layout.split_column(
        Layout(Panel(header, height=3), name="header", size=3),
        Layout(name="body"),
        Layout(Panel(log_text, title="log", border_style="dim"), name="log", size=LOG_TAIL_LINES + 2),
    )
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))
    layout["left"].split_column(
        Layout(Panel(sig, title="signal", border_style="cyan")),
        Layout(Panel(risk, title="risk", border_style="magenta")),
    )
    money_grid = Table.grid()
    money_grid.add_row(money)
    money_grid.add_row(Text())
    money_grid.add_row(trades_tbl)
    layout["right"].update(Panel(money_grid, title="money", border_style="green"))
    return layout


def main():
    parser = argparse.ArgumentParser(description="PolyCrypto terminal dashboard (read-only)")
    parser.add_argument("--dir", default=".", help="project directory (status.json, trades.csv, logs/)")
    parser.add_argument("--refresh", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="render one frame and exit (smoke test)")
    args = parser.parse_args()

    from rich.console import Console
    from rich.live import Live

    status_path = os.path.join(args.dir, "status.json")
    trades_path = os.path.join(args.dir, "trades.csv")
    logs_dir = os.path.join(args.dir, "logs")
    session_start = time.time() - 12 * 3600  # "session" = last 12h

    def frame():
        status, age = load_status(status_path)
        stats = trade_stats(trades_path, session_start_ts=session_start)
        return build_layout(status, age, stats, tail_log(logs_dir))

    console = Console()
    if args.once:
        console.print(frame())
        return
    with Live(frame(), console=console, screen=True, refresh_per_second=4) as live:
        while True:
            time.sleep(args.refresh)
            live.update(frame())


if __name__ == "__main__":
    main()
