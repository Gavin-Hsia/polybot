"""
Paper trade tracker.
Logs picks to data/paper_trades.json, updates outcomes, reports performance.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

TRADES_FILE = Path(__file__).parent.parent / "data" / "paper_trades.json"


def _load() -> list[dict]:
    if not TRADES_FILE.exists():
        return []
    with open(TRADES_FILE) as f:
        return json.load(f)


def _save(trades: list[dict]):
    TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def log_pick(event_ticker: str, title: str, bet_player: str, bet_side: str,
             model_prob: float, kalshi_price: float, edge: float,
             kelly_fraction: float, close_time: str):
    """Record a new paper trade pick."""
    trades = _load()
    trade = {
        "id":             f"{event_ticker}-{bet_side}",
        "event_ticker":   event_ticker,
        "title":          title,
        "bet_player":     bet_player,
        "bet_side":       bet_side,           # "YES" or "NO"
        "model_prob":     round(model_prob, 4),
        "kalshi_price":   round(kalshi_price, 4),  # price paid (ask for YES, 1-bid for NO)
        "edge":           round(edge, 4),
        "kelly_fraction": round(kelly_fraction, 4),
        "close_time":     close_time,
        "logged_at":      datetime.now(timezone.utc).isoformat(),
        "outcome":        None,   # filled in after match resolves
        "pnl":            None,
    }
    # Avoid duplicates
    existing = {t["id"] for t in trades}
    if trade["id"] not in existing:
        trades.append(trade)
        _save(trades)
        return trade
    return None


def resolve(event_ticker: str, winner_name: str):
    """Mark a trade as won or lost after the match result is known."""
    trades = _load()
    updated = 0
    for t in trades:
        if t["event_ticker"] != event_ticker or t["outcome"] is not None:
            continue
        player_won = winner_name.lower() in t["bet_player"].lower() or t["bet_player"].lower() in winner_name.lower()
        if t["bet_side"] == "YES":
            won = player_won
            pnl = (1 - t["kalshi_price"]) if won else -t["kalshi_price"]
        else:  # NO bet = betting opponent wins
            won = not player_won
            pnl = t["kalshi_price"] if won else -(1 - t["kalshi_price"])

        t["outcome"] = "WIN" if won else "LOSS"
        t["pnl"]     = round(pnl, 4)
        updated += 1

    if updated:
        _save(trades)
    return updated


def report() -> dict:
    """Print and return performance summary."""
    trades  = _load()
    settled = [t for t in trades if t["outcome"] is not None]
    pending = [t for t in trades if t["outcome"] is None]

    if not settled:
        print("No settled trades yet.")
        return {}

    wins   = [t for t in settled if t["outcome"] == "WIN"]
    losses = [t for t in settled if t["outcome"] == "LOSS"]
    total_pnl = sum(t["pnl"] for t in settled)

    # Expected value: compare model_prob vs price paid
    avg_edge = sum(t["edge"] for t in settled) / len(settled)

    print(f"\n{'='*50}")
    print(f"  PAPER TRADING PERFORMANCE")
    print(f"{'='*50}")
    print(f"  Settled : {len(settled):>4}  ({len(wins)} W / {len(losses)} L)")
    print(f"  Win rate: {len(wins)/len(settled):.1%}")
    print(f"  Total P&L (per unit): {total_pnl:+.4f}")
    print(f"  Avg edge at entry   : {avg_edge:+.1%}")
    print(f"  Pending picks       : {len(pending)}")

    # ROI by edge bucket
    if len(settled) >= 5:
        import pandas as pd
        df = pd.DataFrame(settled)
        df["edge_bucket"] = pd.cut(df["edge"], bins=[-1, 0.03, 0.07, 0.12, 1.0],
                                   labels=["<3%", "3-7%", "7-12%", ">12%"])
        summary = df.groupby("edge_bucket", observed=True).agg(
            n=("pnl", "count"),
            win_rate=("outcome", lambda x: (x == "WIN").mean()),
            avg_pnl=("pnl", "mean"),
        )
        print(f"\n  Performance by edge bucket:\n{summary.to_string()}")
    print(f"{'='*50}\n")

    return {"settled": len(settled), "wins": len(wins), "pnl": total_pnl, "avg_edge": avg_edge}
