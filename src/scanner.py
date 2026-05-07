"""
Main scanner: polls Kalshi for open ATP matches, runs model predictions,
identifies value bets, and logs paper trade picks.
"""

import pandas as pd
from rapidfuzz import process, fuzz
from typing import Optional

from kalshi import get_all_open_matches
from features import load_matches, build_live_features
from model import load_model, load_cache
from tracker import log_pick

MIN_EDGE   = 0.04   # minimum model edge to log a pick (4%)
MAX_KELLY  = 0.20   # cap Kelly fraction
MIN_VOLUME = 100    # skip illiquid markets


def fuzzy_match_player(kalshi_name: str, atp_names: list[str], threshold: int = 75) -> Optional[str]:
    # token_sort_ratio handles name-order mismatches (e.g. "Bu Yunchaokete" vs "Yunchaokete Bu")
    result = process.extractOne(kalshi_name, atp_names, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= threshold:
        return result[0]
    # Fallback: try partial ratio for players with extra middle names
    result2 = process.extractOne(kalshi_name, atp_names, scorer=fuzz.partial_ratio)
    if result2 and result2[1] >= 90:
        return result2[0]
    return None


def kelly_fraction(model_prob: float, price: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    b = (1 / price) - 1
    q = 1 - model_prob
    f = (b * model_prob - q) / b
    return max(0.0, min(f, MAX_KELLY))


CLAY_TOURNAMENTS = [
    "rome", "roland", "french", "monte carlo", "monte-carlo", "madrid",
    "hamburg", "barcelona", "geneva", "lyon", "estoril", "bucharest",
    "marrakech", "munich", "houston", "rio", "buenos aires", "cordoba",
    "santiago", "lima", "casablanca", "geneva"
]
GRASS_TOURNAMENTS = [
    "wimbledon", "halle", "queens", "nottingham", "eastbourne",
    "s-hertogenbosch", "hertogenbosch", "mallorca", "newport", "stuttgart"
]

def infer_surface(title: str, competition: str = "") -> str:
    combined = (competition + " " + title).lower()
    if any(k in combined for k in CLAY_TOURNAMENTS):
        return "Clay"
    if any(k in combined for k in GRASS_TOURNAMENTS):
        return "Grass"
    return "Hard"


def analyze_match(match: dict, df: pd.DataFrame, model, cache, atp_names: list[str]) -> list[dict]:
    players = match["players"]
    if len(players) < 2:
        return []

    p1_kalshi = players[0]["name"]
    p2_kalshi = players[1]["name"]
    surface   = infer_surface(match["title"], match.get("competition", ""))

    p1_atp = fuzzy_match_player(p1_kalshi, atp_names) or p1_kalshi
    p2_atp = fuzzy_match_player(p2_kalshi, atp_names) or p2_kalshi

    X = build_live_features(p1_atp, p2_atp, surface, df, cache)
    prob_p1 = float(model.predict_proba(X)[0, 1])
    prob_p2 = 1 - prob_p1

    bets = []
    for prob, player_kalshi, player_atp, opponent_kalshi, mkt in [
        (prob_p1, p1_kalshi, p1_atp, p2_kalshi, players[0]),
        (prob_p2, p2_kalshi, p2_atp, p1_kalshi, players[1]),
    ]:
        ask = mkt["yes_ask"]
        if ask <= 0 or mkt["volume"] < MIN_VOLUME:
            continue
        edge = prob - ask
        if edge >= MIN_EDGE:
            kalshi_mid  = mkt["mid"] or ask
            is_upset    = kalshi_mid < 0.45 and prob > 0.50
            bets.append({
                "player":         player_kalshi,
                "atp_name":       player_atp,
                "opponent":       opponent_kalshi,
                "side":           "YES",
                "model_prob":     prob,
                "kalshi_mid":     kalshi_mid,
                "kalshi_price":   ask,
                "edge":           edge,
                "kelly":          kelly_fraction(prob, ask),
                "volume":         mkt["volume"],
                "surface":        surface,
                "is_upset_pick":  is_upset,
            })
    return bets


def scan(log_picks: bool = True) -> list[dict]:
    """Fetch open matches, run predictions, return all value bets."""
    print("Loading ATP data, model, and cache...")
    df    = load_matches()
    model = load_model()
    cache = load_cache()

    atp_names = list(set(
        df["winner_name"].dropna().tolist() +
        df["loser_name"].dropna().tolist()
    ))

    print("Fetching open Kalshi ATP matches...")
    matches = get_all_open_matches()
    print(f"Found {len(matches)} open matches\n")

    all_bets = []
    skipped  = []

    for match in matches:
        bets = analyze_match(match, df, model, cache, atp_names)
        if bets:
            for b in bets:
                all_bets.append((match, b))
        else:
            p1 = match["players"][0]
            p2 = match["players"][1]
            # Check if it was a name mismatch vs genuine no-edge
            p1_matched = fuzzy_match_player(p1["name"], atp_names)
            p2_matched = fuzzy_match_player(p2["name"], atp_names)
            if not p1_matched or not p2_matched:
                skipped.append(match["title"])

    # Sort: upset picks first (more interesting), then by edge within each group
    upset_bets = [(m, b) for m, b in all_bets if b["is_upset_pick"]]
    value_bets = [(m, b) for m, b in all_bets if not b["is_upset_pick"]]
    upset_bets.sort(key=lambda x: x[1]["edge"], reverse=True)
    value_bets.sort(key=lambda x: x[1]["edge"], reverse=True)
    all_bets_sorted = upset_bets + value_bets

    # Print results
    _print_results(all_bets_sorted, matches, skipped)

    # Log to paper trades
    if log_picks:
        for match, bet in all_bets_sorted:
            log_pick(
                event_ticker   = match["event_ticker"],
                title          = match["title"],
                bet_player     = bet["player"],
                bet_side       = bet["side"],
                model_prob     = bet["model_prob"],
                kalshi_price   = bet["kalshi_price"],
                edge           = bet["edge"],
                kelly_fraction = bet["kelly"],
                close_time     = match["close_time"],
            )

    return all_bets_sorted


def _print_results(all_bets: list, matches: list, skipped: list):
    upset_bets = [(m, b) for m, b in all_bets if b["is_upset_pick"]]
    value_bets = [(m, b) for m, b in all_bets if not b["is_upset_pick"]]

    print(f"\n{'='*65}")
    print(f"  UPSET PICKS  — model disagrees with Kalshi favourite")
    print(f"{'='*65}")
    if not upset_bets:
        print("  None today.")
    for match, bet in upset_bets:
        _print_bet(match, bet)

    print(f"\n{'='*65}")
    print(f"  VALUE PICKS  — both sides agree on favourite, but model finds edge")
    print(f"{'='*65}")
    if not value_bets:
        print("  None today.")
    for match, bet in value_bets:
        _print_bet(match, bet)

    print(f"\n{'='*65}")
    print(f"  {len(all_bets)} pick(s) across {len(matches)} matches")
    if skipped:
        print(f"  {len(skipped)} match(es) skipped (players not in ATP dataset)")
    print(f"{'='*65}\n")


def _print_bet(match: dict, bet: dict):
    tag = "★ UPSET" if bet["is_upset_pick"] else "  VALUE"
    print(f"\n  {tag} | {match['title']}")
    print(f"         Bet     : YES on {bet['player']} (vs {bet['opponent']})")
    print(f"         Surface : {bet['surface']}")
    print(f"         Model   : {bet['model_prob']:.1%}  |  Kalshi: {bet['kalshi_mid']:.2f}  |  Edge: {bet['edge']:+.1%}")
    print(f"         Kelly   : {bet['kelly']:.1%} of bankroll  |  Volume: ${bet['volume']:,.0f}")
    print(f"         Closes  : {match['close_time'][:16]}")
