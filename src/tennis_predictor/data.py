"""Download and clean ATP/WTA match data from Jeff Sackmann's public repositories.

Data source: https://github.com/JeffSackmann (tennis_atp, tennis_wta), CC BY-NC-SA 4.0.
"""
from __future__ import annotations

import glob
import os
import re
import time
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config

# Round ordering: qualifying precedes the main draw, which progresses to the final.
ROUND_ORD = {
    "Q1": -3, "Q2": -2, "Q3": -1, "Q4": -1, "ER": 0,
    "R128": 1, "RR": 1, "BR": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
}

# Serve/box-score columns we preserve from the raw CSVs for later feature work.
_SERVE_COLS = [
    "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]


def _download_one(tour: str, kind: str, year: int, data_dir: str) -> Optional[str]:
    """Fetch one season file if not already on disk. Returns the path or None."""
    fn = {
        "atp": {"matches": f"atp_matches_{year}.csv",
                "lower": f"atp_matches_qual_chall_{year}.csv"},
        "wta": {"matches": f"wta_matches_{year}.csv",
                "lower": f"wta_matches_qual_itf_{year}.csv"},
    }[tour][kind]
    path = os.path.join(data_dir, fn)
    if os.path.exists(path):
        return path
    base = f"https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master"
    try:
        urllib.request.urlretrieve(f"{base}/{fn}", path)
        time.sleep(0.15)  # be polite to the raw.githubusercontent host
        return path
    except Exception:
        return None


def download_data(config: Config) -> None:
    """Download all configured seasons for both tours (a no-op for cached files)."""
    os.makedirs(config.data_dir, exist_ok=True)
    for tour in config.tours:
        got = 0
        for year in range(config.start_year, config.end_year + 1):
            for kind in ("matches", "lower"):
                if _download_one(tour, kind, year, config.data_dir):
                    got += 1
        print(f"{tour}: {got} files available")


def refresh_current_year(config: Config) -> None:
    """Delete current-year files so the next download pulls fresh, live-updating data."""
    import datetime as _dt
    year = _dt.date.today().year
    removed = 0
    for f in os.listdir(config.data_dir):
        if str(year) in f and f.endswith(".csv"):
            os.remove(os.path.join(config.data_dir, f))
            removed += 1
    print(f"Removed {removed} current-year ({year}) file(s); they will re-download.")


def parse_score(score: object) -> tuple[int, int, bool]:
    """Parse a score string into (winner_games, loser_games, is_clean).

    ``is_clean`` is False for walkovers, retirements and other non-completed results,
    which carry no signal about on-court dominance.
    """
    if not isinstance(score, str):
        return (0, 0, False)
    s = score.strip()
    if any(t in s.upper() for t in ["W/O", "WALKOVER", "DEF", "ABD", "ABN", "UNFINISHED"]):
        return (0, 0, False)
    had_ret = ("RET" in s.upper()) or ("RTD" in s.upper())
    s = re.sub(r"(RET|RTD)", "", s, flags=re.IGNORECASE).strip()
    wg = lg = sets = 0
    for st in s.split():
        m = re.match(r"(\d+)-(\d+)", st)
        if not m:
            continue
        wg += int(m.group(1))
        lg += int(m.group(2))
        sets += 1
    if sets == 0 or (wg == 0 and lg == 0):
        return (0, 0, False)
    return (wg, lg, not had_ret)


def _load_tour(tour: str, data_dir: str) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(f"{data_dir}/{tour}_matches_*.csv")):
        try:
            frames.append(pd.read_csv(p, low_memory=False))
        except Exception as e:  # pragma: no cover - corrupt download
            print(f"skip {p}: {e}")
    d = pd.concat(frames, ignore_index=True)
    d["tour"] = tour
    return d


def load_matches(config: Config) -> pd.DataFrame:
    """Load every cached CSV, clean it, and return a tidy, chronologically-sorted frame.

    Adds: round ordering, a round-adjusted ``match_date`` (so qualifying matches are
    dated before the main draw), parsed game counts, and a ``dominance_w`` margin
    signal. Walkovers/retirements are kept but flagged via ``score_clean``.
    """
    df = pd.concat([_load_tour(t, config.data_dir) for t in config.tours], ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["winner_name", "loser_name", "surface", "tourney_date"])
    df = df[df["surface"].isin(config.surfaces)]

    df["round_ord"] = (
        df["round"].astype(str).str.upper().str.strip().map(ROUND_ORD).fillna(0).astype(int)
    )
    df = df.sort_values(["tourney_date", "round_ord", "match_num"]).reset_index(drop=True)

    # Round-adjusted estimated date: qualifying rounds (negative round_ord) fall before
    # main-draw start, so rest-days for a qualifier are correctly positive, not zero.
    df["match_date"] = df["tourney_date"] + pd.to_timedelta(df["round_ord"] * 1.3, unit="D")

    ps = df["score"].apply(parse_score)
    df["w_games"] = ps.apply(lambda x: x[0])
    df["l_games"] = ps.apply(lambda x: x[1])
    df["score_clean"] = ps.apply(lambda x: x[2])
    tot = (df["w_games"] + df["l_games"]).replace(0, np.nan)
    df["dominance_w"] = (df["w_games"] / tot).fillna(0.5)

    for col in _SERVE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    print(f"{len(df):,} matches  {df.match_date.min().date()} -> {df.match_date.max().date()}")
    print(f"clean (non WO/RET) scores: {df['score_clean'].mean() * 100:.1f}%")
    print(f"serve stats available: {df['w_ace'].notna().mean() * 100:.1f}%")
    return df
