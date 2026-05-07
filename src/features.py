"""
Builds match features from historical ATP data.
Improvements over v1:
  - Elo ratings (overall + surface-specific) computed from match history
  - Recency-weighted form (recent matches count more via exponential decay)
  - O(n) vectorized precomputation via StatsCache
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

DATA_DIR     = Path(__file__).parent.parent / "data" / "tennis_atp"
FORM_WINDOW  = 20    # matches considered for recent form
SURF_WINDOW  = 100   # surface-specific matches window
ELO_START    = 1500  # default starting Elo
ELO_K        = 32    # Elo K-factor
ELO_K_SURF   = 24    # surface Elo K-factor (less data per surface)
DECAY_HALF   = 10    # half-life in matches for recency weighting


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


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def _decay_weights(n: int, half_life: float) -> np.ndarray:
    """Exponential decay weights: most recent match has weight 1, older matches decay."""
    indices = np.arange(n)
    weights = 0.5 ** (indices[::-1] / half_life)
    return weights / weights.sum()


class StatsCache:
    """
    Precomputes all player stats in one O(n) pass.
    Features: Elo (overall + surface), recency-weighted form, surface WR, H2H.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        print("  Precomputing Elo ratings...")
        self._elo, self._elo_surf = self._precompute_elo(df)
        print("  Precomputing form and surface stats...")
        self._form, self._surf_wr, self._rank = self._precompute_player_stats(df)
        print("  Precomputing H2H stats...")
        self._h2h = self._precompute_h2h(df)
        print("  Done.")

    # ------------------------------------------------------------------
    # Elo precomputation
    # ------------------------------------------------------------------

    @staticmethod
    def _precompute_elo(df):
        """
        Process matches chronologically, maintaining running Elo ratings.
        Records Elo BEFORE each match (no lookahead).
        Returns:
          elo_dict:      {(player, match_idx): elo_rating}
          elo_surf_dict: {(player, match_idx): elo_rating_on_surface}
        """
        elo         = {}       # player -> current overall Elo
        elo_surf    = {}       # (player, surface) -> current surface Elo
        elo_dict      = {}
        elo_surf_dict = {}

        for idx, row in df.iterrows():
            w       = row["winner_name"]
            l       = row["loser_name"]
            surface = row.get("surface", "Hard") or "Hard"

            # Record BEFORE update
            elo_w = elo.get(w, ELO_START)
            elo_l = elo.get(l, ELO_START)
            elo_dict[(w, idx)] = elo_w
            elo_dict[(l, idx)] = elo_l

            esw = elo_surf.get((w, surface), ELO_START)
            esl = elo_surf.get((l, surface), ELO_START)
            elo_surf_dict[(w, idx)] = esw
            elo_surf_dict[(l, idx)] = esl

            # Update overall Elo
            exp_w = _elo_expected(elo_w, elo_l)
            elo[w] = elo_w + ELO_K * (1 - exp_w)
            elo[l] = elo_l + ELO_K * (0 - (1 - exp_w))

            # Update surface Elo
            exp_ws = _elo_expected(esw, esl)
            elo_surf[(w, surface)] = esw + ELO_K_SURF * (1 - exp_ws)
            elo_surf[(l, surface)] = esl + ELO_K_SURF * (0 - (1 - exp_ws))

        return elo_dict, elo_surf_dict

    # ------------------------------------------------------------------
    # Form, surface WR, rank precomputation
    # ------------------------------------------------------------------

    @staticmethod
    def _precompute_player_stats(df):
        w = df[["tourney_date", "winner_name", "surface", "winner_rank"]].copy()
        w.columns = ["date", "player", "surface", "rank"]
        w["won"] = 1
        w["orig_idx"] = df.index

        l = df[["tourney_date", "loser_name", "surface", "loser_rank"]].copy()
        l.columns = ["date", "player", "surface", "rank"]
        l["won"] = 0
        l["orig_idx"] = df.index

        long = pd.concat([w, l], ignore_index=True)

        form_dict    = {}
        surf_wr_dict = {}
        rank_dict    = {}

        # Recency-weighted form per player
        for player, grp in long.groupby("player", sort=False):
            grp = grp.sort_values("date").reset_index(drop=True)
            wins = grp["won"].values
            for i in range(len(grp)):
                row = grp.iloc[i]
                key = (player, int(row["orig_idx"]))
                rank_dict[key] = float(row["rank"]) if pd.notna(row["rank"]) else 150.0

                # Use matches BEFORE current (shift by 1)
                past = wins[:i]
                if len(past) >= 3:
                    n = min(len(past), FORM_WINDOW)
                    w_arr = _decay_weights(n, DECAY_HALF)
                    form_dict[key] = float(np.dot(past[-n:], w_arr))
                else:
                    form_dict[key] = 0.5

        # Surface win rate per (player, surface)
        for (player, surface), grp in long.groupby(["player", "surface"], sort=False):
            grp = grp.sort_values("date").reset_index(drop=True)
            rolled_surf = grp["won"].shift(1).rolling(SURF_WINDOW, min_periods=5).mean()
            for i in range(len(grp)):
                row = grp.iloc[i]
                key = (player, int(row["orig_idx"]))
                sv  = rolled_surf.iloc[i]
                surf_wr_dict[key] = float(sv) if pd.notna(sv) else 0.5

        return form_dict, surf_wr_dict, rank_dict

    # ------------------------------------------------------------------
    # H2H precomputation
    # ------------------------------------------------------------------

    @staticmethod
    def _precompute_h2h(df):
        h2h_running = {}
        h2h_dict    = {}
        for idx, row in df.iterrows():
            w, l = row["winner_name"], row["loser_name"]
            key  = frozenset({w, l})
            counts = h2h_running.get(key, {w: 0, l: 0})
            h2h_dict[(w, l, idx)] = (counts.get(w, 0), counts.get(l, 0))
            h2h_dict[(l, w, idx)] = (counts.get(l, 0), counts.get(w, 0))
            if key not in h2h_running:
                h2h_running[key] = {w: 0, l: 0}
            h2h_running[key][w] += 1
        return h2h_dict

    # ------------------------------------------------------------------
    # Batch lookups (training)
    # ------------------------------------------------------------------

    def elo(self, player: str, match_idx: int) -> float:
        return self._elo.get((player, match_idx), ELO_START)

    def elo_surf(self, player: str, match_idx: int) -> float:
        return self._elo_surf.get((player, match_idx), ELO_START)

    def form(self, player: str, match_idx: int) -> float:
        return self._form.get((player, match_idx), 0.5)

    def surf_wr(self, player: str, match_idx: int) -> float:
        return self._surf_wr.get((player, match_idx), 0.5)

    def rank(self, player: str, match_idx: int) -> float:
        return self._rank.get((player, match_idx), 150.0)

    def h2h(self, p1: str, p2: str, match_idx: int):
        w1, w2 = self._h2h.get((p1, p2, match_idx), (0, 0))
        total  = w1 + w2
        return w1, w2, (w1 / total if total > 0 else 0.5)

    # ------------------------------------------------------------------
    # Live lookups (scan.py — current date)
    # ------------------------------------------------------------------

    def live_elo(self, player: str) -> float:
        """Most recent overall Elo for a player."""
        matches = [(idx, v) for (p, idx), v in self._elo.items() if p == player]
        return max(matches, key=lambda x: x[0])[1] if matches else ELO_START

    def live_elo_surf(self, player: str, surface: str) -> float:
        """Most recent surface Elo — uses overall Elo as fallback."""
        surf_matches = [(idx, v) for (p, idx), v in self._elo_surf.items()
                        if p == player]
        if not surf_matches:
            return self.live_elo(player)
        return max(surf_matches, key=lambda x: x[0])[1]

    def live_form(self, player: str, df: pd.DataFrame) -> float:
        played = df[(df["winner_name"] == player) | (df["loser_name"] == player)].tail(FORM_WINDOW)
        if len(played) < 3:
            return 0.5
        wins = (played["winner_name"] == player).values.astype(float)
        w_arr = _decay_weights(len(wins), DECAY_HALF)
        return float(np.dot(wins, w_arr))

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
        w1 = int((hist["winner_name"] == p1).sum())
        w2 = int((hist["winner_name"] == p2).sum())
        total = w1 + w2
        return w1, w2, (w1 / total if total > 0 else 0.5)


# ------------------------------------------------------------------
# Feature construction
# ------------------------------------------------------------------

def _make_feature_dict(r1, r2, elo1, elo2, elo_s1, elo_s2,
                       form1, form2, sw1, sw2, h2h1, h2h2, h2h_rate, h2h_total, surface):
    return {
        "rank_p1":          r1,
        "rank_p2":          r2,
        "rank_diff":        r2 - r1,
        "rank_ratio":       r2 / max(r1, 1),
        "log_rank_ratio":   np.log(max(r2, 1) / max(r1, 1)),
        "elo_p1":           elo1,
        "elo_p2":           elo2,
        "elo_diff":         elo1 - elo2,
        "elo_surf_p1":      elo_s1,
        "elo_surf_p2":      elo_s2,
        "elo_surf_diff":    elo_s1 - elo_s2,
        "form_p1":          form1,
        "form_p2":          form2,
        "form_diff":        form1 - form2,
        "surf_wr_p1":       sw1,
        "surf_wr_p2":       sw2,
        "surf_wr_diff":     sw1 - sw2,
        "h2h_wins_p1":      h2h1,
        "h2h_wins_p2":      h2h2,
        "h2h_rate_p1":      h2h_rate,
        "h2h_total":        h2h_total,
        "surface_clay":     int(surface == "Clay"),
        "surface_grass":    int(surface == "Grass"),
        "surface_hard":     int(surface == "Hard"),
    }


FEATURE_COLS = [
    "rank_p1", "rank_p2", "rank_diff", "rank_ratio", "log_rank_ratio",
    "elo_p1", "elo_p2", "elo_diff",
    "elo_surf_p1", "elo_surf_p2", "elo_surf_diff",
    "form_p1", "form_p2", "form_diff",
    "surf_wr_p1", "surf_wr_p2", "surf_wr_diff",
    "h2h_wins_p1", "h2h_wins_p2", "h2h_rate_p1", "h2h_total",
    "surface_clay", "surface_grass", "surface_hard",
]


def build_training_rows(df: pd.DataFrame, cache: StatsCache, rng=None):
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

        h1, h2, hr = cache.h2h(p1, p2, idx)
        rows.append(_make_feature_dict(
            cache.rank(p1, idx),     cache.rank(p2, idx),
            cache.elo(p1, idx),      cache.elo(p2, idx),
            cache.elo_surf(p1, idx), cache.elo_surf(p2, idx),
            cache.form(p1, idx),     cache.form(p2, idx),
            cache.surf_wr(p1, idx),  cache.surf_wr(p2, idx),
            h1, h2, hr, h1 + h2, surface,
        ))
        labels.append(label)

    return pd.DataFrame(rows)[FEATURE_COLS], pd.Series(labels)


def build_live_features(p1: str, p2: str, surface: str,
                        df: pd.DataFrame, cache: StatsCache) -> pd.DataFrame:
    h1, h2, hr = cache.live_h2h(p1, p2, df)
    feats = _make_feature_dict(
        cache.live_rank(p1, df),      cache.live_rank(p2, df),
        cache.live_elo(p1),           cache.live_elo(p2),
        cache.live_elo_surf(p1, surface), cache.live_elo_surf(p2, surface),
        cache.live_form(p1, df),      cache.live_form(p2, df),
        cache.live_surf_wr(p1, surface, df), cache.live_surf_wr(p2, surface, df),
        h1, h2, hr, h1 + h2, surface,
    )
    return pd.DataFrame([feats])[FEATURE_COLS]
