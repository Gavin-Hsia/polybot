"""
Kalshi API client — read-only market data, no auth required.
"""

import time
import requests
from typing import Optional

REQUEST_DELAY = 0.5  # seconds between API calls to avoid 429s

BASE = "https://api.elections.kalshi.com/trade-api/v2"

ATP_SERIES   = ["KXATPMATCH", "KXATPCHALLENGERMATCH"]
GRAND_SLAMS  = ["KXATPGRANDSLAM", "KXFOMEN", "KXFOMENSINGLES", "KXWMENSINGLES"]


def get_open_atp_matches() -> list[dict]:
    """Return all open ATP match events with their market prices."""
    events = []
    for series in ATP_SERIES:
        resp = requests.get(f"{BASE}/events", params={"limit": 200, "status": "open", "series_ticker": series}, timeout=10)
        resp.raise_for_status()
        events.extend(resp.json().get("events", []))
    return events


def get_markets_for_event(event_ticker: str) -> list[dict]:
    time.sleep(REQUEST_DELAY)
    resp = requests.get(f"{BASE}/markets", params={"event_ticker": event_ticker, "limit": 50}, timeout=10)
    resp.raise_for_status()
    return resp.json().get("markets", [])


def parse_match(event: dict) -> Optional[dict]:
    """
    Turn a Kalshi ATP match event + its markets into a structured dict.
    Returns None if market data is missing or unparseable.
    """
    ticker   = event["event_ticker"]
    title    = event.get("title", "")
    markets  = get_markets_for_event(ticker)

    if len(markets) < 2:
        return None

    players = []
    for m in markets:
        yes_bid = float(m.get("yes_bid_dollars", 0) or 0)
        yes_ask = float(m.get("yes_ask_dollars", 0) or 0)
        mid     = (yes_bid + yes_ask) / 2 if yes_ask > 0 else None
        name    = m.get("yes_sub_title") or m.get("title", "")
        players.append({
            "name":      name,
            "ticker":    m["ticker"],
            "yes_bid":   yes_bid,
            "yes_ask":   yes_ask,
            "mid":       mid,
            "spread":    round(yes_ask - yes_bid, 4) if yes_ask > 0 else None,
            "volume":    float(m.get("volume_fp", 0) or 0),
        })

    if len(players) < 2 or players[0]["mid"] is None:
        return None

    competition = event.get("product_metadata", {}).get("competition", "")

    return {
        "event_ticker": ticker,
        "title":        title,
        "competition":  competition,
        "close_time":   markets[0].get("close_time", ""),
        "players":      players,
    }


def get_all_open_matches() -> list[dict]:
    """Fetch and parse all open ATP matches. Skips any with missing prices."""
    events  = get_open_atp_matches()
    matches = []
    for e in events:
        parsed = parse_match(e)
        if parsed:
            matches.append(parsed)
    return matches
