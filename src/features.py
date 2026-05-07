"""
Builds match features from historical ATP data.
Uses precomputed lookup tables so training is O(n) not O(n²).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from bisect import bisect_left

DATA_DIR    = Path(__file__).parent.parent / "data" / "tennis_atp"
FORM_WINDOW = 15   # last N matches for recent form
SURF_WINDOW = 100  # last N surface matches for surface win rate


@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    frames = []
    for yr in range(2005, 2026):
        path = DATA_DIR / f"atp_matches_{yr}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df["winner_rank"]  = pd.to_numeric(df["winner_rank"], errors="coerce")
    df["loser_rank"]   = pd.to_numeric(df["loser_rank"],  errors="coerce")
    return df.sort_values("tourney_date").reset_index(drop=True)


class StatsCache:
    """
    Precomputes all player stats in one pass so per-match feature
    building is just dictionary lookups — no more O(n²) scanning.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        print("  Precomputing form stats...")
        self._form, self._surf_wr, self._rank = self._precompute_player_stats(df)
        print("  Precomputing H2H stats...")
        self._h2h = self._precompute_h2h(df)
        print("  Done.")

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    @staticmethod
    def _precompute_player_stats(df):
        """
        Build long-form player-match table, then compute rolling
        form and surface win rates per player using vectorized pandas ops.
        Returns dicts keyed by (player, match_idx).
        """
        w = df[["tourney_date", "winner_name", "surface", "winner_rank"]].copy()
        w.columns = ["date", "player", "surface", "rank"]
        w["won"] = 1
        w["orig_idx"] = df.index  # which match row this came from

        l = df[["tourney_date", "loser_name", "surface", "loser_rank"]].copy()
        l.columns = ["date", "player", "surface", "rank"]
        l["won"] = 0
        l["orig_idx"] = df.index

        long = pd.concat([w, l], ignore_index=True).sort_values(["player", "date"])

        form_dict    = {}
        surf_wr_dict = {}
        rank_dict    = {}

        # --- Form and rank: per player ---
        for player, grp in long.groupby("player", sort=False):
            grp = grp.sort_values("date").reset_index(drop=True)
            rolled_form = grp["won"].shift(1).rolling(FORM_WINDOW, min_periods=3).mean()
            for i in range(len(grp)):
                row = grp.iloc[i]
                key = (player, int(row["orig_idx"]))
                fv  = rolled_form.iloc[i]
                form_dict[key] = float(fv) if pd.notna(fv) else 0.5
                r = row["rank"]
                rank_dict[key] = float(r) if pd.notna(r) else 150.0

        # --- Surface win rate: per (player, surface) ---
        for (player, surface), grp in long.groupby(["player", "surface"], sort=False):
            grp = grp.sort_values("date").reset_index(drop=True)
            rolled_surf = grp["won"].shift(1).rolling(SURF_WINDOW, min_periods=5).mean()
            for i in range(len(grp)):
                row = grp.iloc[i]
                key = (player, int(row["orig_idx"]))
                sv  = rolled_surf.iloc[i]
                surf_wr_dict[key] = float(sv) if pd.notna(sv) else 0.5

        return form_dict, surf_wr_dict, rank_dict

    @staticmethod
    def _precompute_h2h(df):
        """
        Process matches chronologically once, maintaining running H2H counts.
        Stores the count BEFORE each match so there's no lookahead.
        Returns dict: (player_a, player_b, match_idx) -> (a_wins, b_wins).
        """
        h2h_running = {}  # {frozenset({p1, p2}): {p1: n, p2: n}}
        h2h_dict    = {}

        for idx, row in df.iterrows():
            w, l = row["winner_name"], row["loser_name"]
            key  = frozenset({w, l})

            counts = h2h_running.get(key, {w: 0, l: 0})
            h2h_dict[(w, l, idx)] = (counts.get(w, 0), counts.get(l, 0))
            h2h_dict[(l, w, idx)] = (counts.get(l, 0), counts.get(w, 0))

            # Update AFTER recording
            if key not in h2h_running:
                h2h_running[key] = {w: 0, l: 0}
            h2h_running[key][w] += 1

        return h2h_dict

    # ------------------------------------------------------------------
    # Per-match lookup
    # ------------------------------------------------------------------

    def form(self, player: str, match_idx: int) -> float:
        return self._form.get((player, match_idx), 0.5)

    def surf_wr(self, player: str, match_idx: int) -> float:
        return self._surf_wr.get((player, match_idx), 0.5)

    def rank(self, player: str, match_idx: int) -> float:
        return self._rank.get((player, match_idx), 150.0)

    def h2h(self, p1: str, p2: str, match_idx: int):
        w1, w2 = self._h2h.get((p1, p2, match_idx), (0, 0))
        total  = w1 + w2
        rate   = w1 / total if total > 0 else 0.5
        return w1, w2, rate

    # ------------------------------------------------------------------
    # Live lookup (for scan.py — current date, no match_idx)
    # ------------------------------------------------------------------

    def live_form(self, player: str, df: pd.DataFrame) -> float:
        played = df[(df["winner_name"] == player) | (df["loser_name"] == player)].tail(FORM_WINDOW)
        if len(played) < 3:
            return 0.5
        return float((played["winner_name"] == player).mean())

    def live_surf_wr(self, player: str, surface: str, df: pd.DataFrame) -> float:
        played = df[
            ((df["winner_name"] == player) | (df["loser_name"] == player)) &
            (df["surface"] == surface)
        ].tail(SURF_WINDOW)
        if len(played) < 5:
            return 0.5
        return float((played["winner_name"] == player).mean())

    def live_rank(self, player: str, df: pd.DataFrame) -> float:
        w = df[df["winner_name"] == player]["winner_rank"].dropna()
        l = df[df["loser_name"]  == player]["loser_rank"].dropna()
        combined = pd.concat([w, l])
        return float(combined.iloc[-1]) if len(combined) else 150.0

    def live_h2h(self, p1: str, p2: str, df: pd.DataFrame):
        hist = df[
            ((df["winner_name"] == p1) & (df["loser_name"] == p2)) |
            ((df["winner_name"] == p2) & (df["loser_name"] == p1))
        ]
        w1 = (hist["winner_name"] == p1).sum()
        w2 = (hist["winner_name"] == p2).sum()
        total = w1 + w2
        rate  = w1 / total if total > 0 else 0.5
        return int(w1), int(w2), rate


def _make_feature_dict(r1, r2, form1, form2, sw1, sw2, h2h1, h2h2, h2h_rate, h2h_total, surface):
    return {
        "rank_p1":        r1,
        "rank_p2":        r2,
        "rank_diff":      r2 - r1,
        "rank_ratio":     r2 / max(r1, 1),
        "log_rank_ratio": np.log(max(r2, 1) / max(r1, 1)),
        "form_p1":        form1,
        "form_p2":        form2,
        "form_diff":      form1 - form2,
        "surf_wr_p1":     sw1,
        "surf_wr_p2":     sw2,
        "surf_wr_diff":   sw1 - sw2,
        "h2h_wins_p1":    h2h1,
        "h2h_wins_p2":    h2h2,
        "h2h_rate_p1":    h2h_rate,
        "h2h_total":      h2h_total,
        "surface_clay":   int(surface == "Clay"),
        "surface_grass":  int(surface == "Grass"),
        "surface_hard":   int(surface == "Hard"),
    }


FEATURE_COLS = [
    "rank_p1", "rank_p2", "rank_diff", "rank_ratio", "log_rank_ratio",
    "form_p1", "form_p2", "form_diff",
    "surf_wr_p1", "surf_wr_p2", "surf_wr_diff",
    "h2h_wins_p1", "h2h_wins_p2", "h2h_rate_p1", "h2h_total",
    "surface_clay", "surface_grass", "surface_hard",
]


def build_training_rows(df: pd.DataFrame, cache: StatsCache, rng=None):
    """
    Vectorized: build all training feature rows using precomputed cache.
    Returns list of (feature_dict, label) tuples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    ranked = df.dropna(subset=["winner_rank", "loser_rank"])
    rows, labels = [], []

    for idx, row in ranked.iterrows():
        winner, loser = row["winner_name"], row["loser_name"]
        surface       = row.get("surface", "Hard") or "Hard"

        swap = rng.random() < 0.5
        p1, p2 = (winner, loser) if not swap else (loser, winner)
        label  = 1 if not swap else 0

        r1   = cache.rank(p1, idx)
        r2   = cache.rank(p2, idx)
        f1   = cache.form(p1, idx)
        f2   = cache.form(p2, idx)
        sw1  = cache.surf_wr(p1, idx)
        sw2  = cache.surf_wr(p2, idx)
        h1, h2, hr = cache.h2h(p1, p2, idx)

        rows.append(_make_feature_dict(r1, r2, f1, f2, sw1, sw2, h1, h2, hr, h1 + h2, surface))
        labels.append(label)

    return pd.DataFrame(rows)[FEATURE_COLS], pd.Series(labels)


def build_live_features(p1: str, p2: str, surface: str, df: pd.DataFrame, cache: StatsCache) -> pd.DataFrame:
    """Build features for a live match using the most recent data."""
    r1  = cache.live_rank(p1, df)
    r2  = cache.live_rank(p2, df)
    f1  = cache.live_form(p1, df)
    f2  = cache.live_form(p2, df)
    sw1 = cache.live_surf_wr(p1, surface, df)
    sw2 = cache.live_surf_wr(p2, surface, df)
    h1, h2, hr = cache.live_h2h(p1, p2, df)

    feats = _make_feature_dict(r1, r2, f1, f2, sw1, sw2, h1, h2, hr, h1 + h2, surface)
    return pd.DataFrame([feats])[FEATURE_COLS]
