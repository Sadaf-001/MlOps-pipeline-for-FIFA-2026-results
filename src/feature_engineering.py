"""
feature_engineering.py
=======================
Builds time-series features from historical match results.
All features are computed using only matches BEFORE the current match date
to prevent data leakage.

Features generated per match:
  Rolling (last N matches per team):
    - win rate, draw rate, loss rate
    - goals scored avg, goals conceded avg
    - home-specific win rate (home team only)
    - away-specific win rate (away team only)
  Head-to-head:
    - home team win rate vs this specific away team
  Streak:
    - current consecutive win/loss streak going into the match

Usage:
    python src/feature_engineering.py \
        --input  data/results.csv \
        --output data/results_engineered.csv \
        --window 10
"""

import argparse
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd


# ─── Rolling stats tracker ────────────────────────────────────────────────────

class TeamHistory:
    """Maintains a rolling window of past match outcomes for one team."""

    def __init__(self, window: int):
        self.window = window
        self.all_matches   = deque()   # (goals_for, goals_against, result)  W/D/L
        self.home_matches  = deque()   # only matches played as home team
        self.away_matches  = deque()   # only matches played as away team

    def _trim(self, q: deque):
        while len(q) > self.window:
            q.popleft()

    def add(self, goals_for: int, goals_against: int, is_home: bool):
        if goals_for > goals_against:
            result = "W"
        elif goals_for < goals_against:
            result = "L"
        else:
            result = "D"

        entry = (goals_for, goals_against, result)
        self.all_matches.append(entry)
        self._trim(self.all_matches)

        if is_home:
            self.home_matches.append(entry)
            self._trim(self.home_matches)
        else:
            self.away_matches.append(entry)
            self._trim(self.away_matches)

    def _stats(self, q: deque) -> dict:
        n = len(q)
        if n == 0:
            return dict(win_rate=np.nan, draw_rate=np.nan, loss_rate=np.nan,
                        goals_scored_avg=np.nan, goals_conceded_avg=np.nan,
                        n_matches=0)
        wins   = sum(1 for *_, r in q if r == "W")
        draws  = sum(1 for *_, r in q if r == "D")
        losses = sum(1 for *_, r in q if r == "L")
        gf     = sum(gf for gf, *_ in q)
        ga     = sum(ga for _, ga, *_ in q)
        return dict(
            win_rate=round(wins / n, 4),
            draw_rate=round(draws / n, 4),
            loss_rate=round(losses / n, 4),
            goals_scored_avg=round(gf / n, 4),
            goals_conceded_avg=round(ga / n, 4),
            n_matches=n,
        )

    def overall_stats(self) -> dict:
        return self._stats(self.all_matches)

    def home_stats(self) -> dict:
        return self._stats(self.home_matches)

    def away_stats(self) -> dict:
        return self._stats(self.away_matches)

    def streak(self) -> int:
        """
        Positive = current winning streak length.
        Negative = current losing streak length.
        0 = last match was a draw or no history.
        """
        if not self.all_matches:
            return 0
        last_result = self.all_matches[-1][2]
        if last_result == "D":
            return 0
        count = 0
        for entry in reversed(self.all_matches):
            if entry[2] == last_result:
                count += 1
            else:
                break
        return count if last_result == "W" else -count


# ─── Head-to-head tracker ─────────────────────────────────────────────────────

class H2HHistory:
    """Tracks historical results between two specific teams."""

    def __init__(self):
        # keyed by frozenset({teamA, teamB}) → list of outcomes from teamA's perspective
        self.records: dict[frozenset, list] = defaultdict(list)

    def add(self, home: str, away: str, home_goals: int, away_goals: int):
        key = frozenset({home, away})
        if home_goals > away_goals:
            outcome_home, outcome_away = "W", "L"
        elif home_goals < away_goals:
            outcome_home, outcome_away = "L", "W"
        else:
            outcome_home, outcome_away = "D", "D"
        self.records[key].append((home, outcome_home))
        self.records[key].append((away, outcome_away))

    def win_rate(self, team: str, opponent: str) -> float:
        key = frozenset({team, opponent})
        matches = [(t, r) for t, r in self.records[key] if t == team]
        if not matches:
            return np.nan
        wins = sum(1 for _, r in matches if r == "W")
        return round(wins / len(matches), 4)


# ─── Main feature builder ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Iterate through matches in chronological order.
    For each match, snapshot features BEFORE updating the history.
    This guarantees zero leakage.
    """
    df = df.sort_values("date").reset_index(drop=True)

    team_histories: dict[str, TeamHistory] = defaultdict(lambda: TeamHistory(window))
    h2h = H2HHistory()

    rows = []

    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        hg   = match["home_score"]
        ag   = match["away_score"]

        # ── Snapshot features BEFORE this match ──────────────────────────────
        h_stats  = team_histories[home].overall_stats()
        a_stats  = team_histories[away].overall_stats()
        hh_stats = team_histories[home].home_stats()
        aa_stats = team_histories[away].away_stats()
        h_streak = team_histories[home].streak()
        a_streak = team_histories[away].streak()
        h2h_rate = h2h.win_rate(home, away)

        row = {
            # Original columns kept for reference / encoding
            "date":        match["date"],
            "home_team":   home,
            "away_team":   away,
            "tournament":  match["tournament"],
            "neutral":     match["neutral"],
            "home_score":  hg,
            "away_score":  ag,

            # ── Home team rolling features ────────────────────────────────
            "home_win_rate":            h_stats["win_rate"],
            "home_draw_rate":           h_stats["draw_rate"],
            "home_loss_rate":           h_stats["loss_rate"],
            "home_goals_scored_avg":    h_stats["goals_scored_avg"],
            "home_goals_conceded_avg":  h_stats["goals_conceded_avg"],
            "home_n_matches":           h_stats["n_matches"],

            # Home team when playing AT HOME specifically
            "home_home_win_rate":       hh_stats["win_rate"],
            "home_home_goals_avg":      hh_stats["goals_scored_avg"],

            # Home team streak
            "home_streak":              h_streak,

            # ── Away team rolling features ────────────────────────────────
            "away_win_rate":            a_stats["win_rate"],
            "away_draw_rate":           a_stats["draw_rate"],
            "away_loss_rate":           a_stats["loss_rate"],
            "away_goals_scored_avg":    a_stats["goals_scored_avg"],
            "away_goals_conceded_avg":  a_stats["goals_conceded_avg"],
            "away_n_matches":           a_stats["n_matches"],

            # Away team when playing AWAY specifically
            "away_away_win_rate":       aa_stats["win_rate"],
            "away_away_goals_avg":      aa_stats["goals_scored_avg"],

            # Away team streak
            "away_streak":              a_streak,

            # ── Head-to-head ──────────────────────────────────────────────
            "h2h_home_win_rate":        h2h_rate,

            # ── Derived differentials ─────────────────────────────────────
            # Positive = home team advantage
            "win_rate_diff":   _safe_diff(h_stats["win_rate"],   a_stats["win_rate"]),
            "goals_diff":      _safe_diff(h_stats["goals_scored_avg"], a_stats["goals_scored_avg"]),
            "streak_diff":     float(h_streak - a_streak),
        }
        rows.append(row)

        # ── Update histories AFTER snapshotting ───────────────────────────────
        team_histories[home].add(hg, ag, is_home=True)
        team_histories[away].add(ag, hg, is_home=False)
        h2h.add(home, away, hg, ag)

    return pd.DataFrame(rows)


def _safe_diff(a, b) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return round(float(a) - float(b), 4)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/results.csv")
    p.add_argument("--output", default="data/results_engineered.csv")
    p.add_argument("--window", type=int, default=10,
                   help="Rolling window size (number of past matches).")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[1/3] Loading '{args.input}' …")
    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].str.upper().map({"TRUE": 1, "FALSE": 0})
    else:
        df["neutral"] = df["neutral"].astype(int)
    print(f"  {len(df)} matches loaded ({df['date'].min().date()} → {df['date'].max().date()})")

    print(f"\n[2/3] Building time-series features (window={args.window}) …")
    engineered = build_features(df, window=args.window)

    # Drop the first few rows where teams have < 3 historical matches (too noisy)
    before = len(engineered)
    engineered = engineered[engineered["home_n_matches"] >= 3]
    engineered = engineered[engineered["away_n_matches"] >= 3]
    print(f"  Dropped {before - len(engineered)} rows with < 3 match history")
    print(f"  Final dataset: {len(engineered)} rows")

    # Fill remaining NaNs (h2h with no prior meetings) with 0.5 (neutral prior)
    h2h_col = "h2h_home_win_rate"
    engineered[h2h_col] = engineered[h2h_col].fillna(0.5)
    # Fill any other NaNs with column median
    engineered = engineered.fillna(engineered.median(numeric_only=True))

    print(f"\n[3/3] Saving to '{args.output}' …")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    engineered.to_csv(args.output, index=False)

    print("\n  Feature summary:")
    ts_cols = [c for c in engineered.columns
               if c not in ("date","home_team","away_team","tournament",
                             "neutral","home_score","away_score")]
    for col in ts_cols:
        print(f"    {col:<35s}  mean={engineered[col].mean():.3f}  "
              f"null={engineered[col].isna().sum()}")

    print(f"\n✅  Done. Engineered CSV → {args.output}")


if __name__ == "__main__":
    main()
