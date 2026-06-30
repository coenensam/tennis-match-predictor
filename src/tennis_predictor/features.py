"""Leak-free feature engineering on top of the Elo replay.

Every feature is a player's state *entering* the match: rolling form, head-to-head,
rest days, and trailing serve/return and strength-of-schedule rollups. Each rollup is
recorded *before* the current match is folded into the player's history, so the exact
same construction runs at train and inference time — this is what killed an earlier
version that accidentally fed post-match box scores into training.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config

# 18 model features, in the order XGBoost expects them.
FEATURES = [
    "elo_diff", "ov_diff", "p_elo",
    "form_p1", "form_p2", "h2h",
    "rest_p1", "rest_p2",
    "rank_diff", "min_nm",
    "surface_clay", "surface_grass", "tour_wta",
    # serve/return trailing diffs (NaN until a player has serve history; XGBoost routes
    # NaN natively, and the value is built identically at train and inference time).
    "spw_diff", "rpw_diff", "ace_diff", "bps_diff",
    # opponent quality: recent average opponent rank (strength of schedule). Uses the
    # opponent's rank, so it is immune to the inactive-star own-rank collapse.
    "sos_diff",
]

_SERVE_COLS = [
    "w_svpt", "w_1stWon", "w_2ndWon", "w_ace", "w_bpSaved", "w_bpFaced",
    "l_svpt", "l_1stWon", "l_2ndWon", "l_ace", "l_bpSaved", "l_bpFaced",
]


def _avg(dq) -> float:
    return float(np.mean(dq)) if len(dq) else np.nan


@dataclass
class FeatureHistory:
    """Trailing per-player rollups, also reused by the live predictor."""

    form: Dict = field(default_factory=lambda: defaultdict(list))
    h2h: Dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    last_date: Dict = field(default_factory=dict)
    spw: Dict = None  # serve points won %
    rpw: Dict = None  # return points won %
    ace: Dict = None  # ace rate per serve point
    bps: Dict = None  # break points saved %
    sos: Dict = None  # opponent ranks faced

    @classmethod
    def new(cls, config: Config) -> "FeatureHistory":
        sw = config.serve_window
        return cls(
            spw=defaultdict(lambda: deque(maxlen=sw)),
            rpw=defaultdict(lambda: deque(maxlen=sw)),
            ace=defaultdict(lambda: deque(maxlen=sw)),
            bps=defaultdict(lambda: deque(maxlen=sw)),
            sos=defaultdict(lambda: deque(maxlen=config.sos_window)),
        )


def build_features(elo: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, FeatureHistory]:
    """Turn the per-match Elo frame into a model-ready matrix plus the live history.

    Match perspective is randomised (p1/p2) so the label is balanced and the model can
    never exploit a "winner is always player 1" shortcut.
    """
    has_serve = all(c in elo.columns for c in _SERVE_COLS)
    if has_serve:
        elo[_SERVE_COLS] = elo[_SERVE_COLS].apply(pd.to_numeric, errors="coerce")

    hist = FeatureHistory.new(config)
    fw = config.form_window
    records = []

    for r in elo.itertuples():
        tour = r.tour
        w, l = (tour, r.winner), (tour, r.loser)
        d, s = r.date, r.surface

        form_w = np.mean(hist.form[w][-fw:]) if hist.form[w] else 0.5
        form_l = np.mean(hist.form[l][-fw:]) if hist.form[l] else 0.5
        hw, hl = hist.h2h[(w, l)]
        rest_w = (d - hist.last_date[w]).days if w in hist.last_date else 30
        rest_l = (d - hist.last_date[l]).days if l in hist.last_date else 30

        records.append({
            "date": d, "tour": tour, "level": r.level, "surface": s,
            "winner": r.winner, "loser": r.loser,
            "elo_diff_w": r.elo_w - r.elo_l, "ov_diff_w": r.ov_w - r.ov_l, "p_elo_w": r.p_w_elo,
            "form_w": form_w, "form_l": form_l, "h2h_w": hw - hl,
            "rest_w": min(rest_w, 60), "rest_l": min(rest_l, 60),
            "rank_diff_w": (r.rank_l - r.rank_w)
                if not (np.isnan(r.rank_w) or np.isnan(r.rank_l)) else 0,
            "nm_w": r.nm_w, "nm_l": r.nm_l, "min_nm": min(r.nm_w, r.nm_l),
            "spw_w": _avg(hist.spw[w]), "spw_l": _avg(hist.spw[l]),
            "rpw_w": _avg(hist.rpw[w]), "rpw_l": _avg(hist.rpw[l]),
            "ace_w": _avg(hist.ace[w]), "ace_l": _avg(hist.ace[l]),
            "bps_w": _avg(hist.bps[w]), "bps_l": _avg(hist.bps[l]),
            "sos_w": _avg(hist.sos[w]), "sos_l": _avg(hist.sos[l]),
        })

        # --- update histories AFTER recording (so the row above stayed pre-match) ---
        hist.form[w].append(1)
        hist.form[l].append(0)
        hist.h2h[(w, l)][0] += 1
        hist.h2h[(l, w)][1] += 1
        hist.last_date[w] = d
        hist.last_date[l] = d

        if has_serve:
            wsv, lsv = r.w_svpt, r.l_svpt
            if wsv and lsv and not (np.isnan(wsv) or np.isnan(lsv)) and wsv > 0 and lsv > 0:
                hist.spw[w].append((r.w_1stWon + r.w_2ndWon) / wsv)
                hist.spw[l].append((r.l_1stWon + r.l_2ndWon) / lsv)
                hist.rpw[w].append((lsv - r.l_1stWon - r.l_2ndWon) / lsv)
                hist.rpw[l].append((wsv - r.w_1stWon - r.w_2ndWon) / wsv)
                hist.ace[w].append(r.w_ace / wsv)
                hist.ace[l].append(r.l_ace / lsv)
                if r.w_bpFaced and not np.isnan(r.w_bpFaced) and r.w_bpFaced > 0:
                    hist.bps[w].append(r.w_bpSaved / r.w_bpFaced)
                if r.l_bpFaced and not np.isnan(r.l_bpFaced) and r.l_bpFaced > 0:
                    hist.bps[l].append(r.l_bpSaved / r.l_bpFaced)

        if pd.notna(r.rank_l) and float(r.rank_l) > 0:
            hist.sos[w].append(float(r.rank_l))
        if pd.notna(r.rank_w) and float(r.rank_w) > 0:
            hist.sos[l].append(float(r.rank_w))

    F = pd.DataFrame(records)
    data = _randomise_perspective(F, config)
    cov = data["spw_diff"].notna().mean() * 100
    print(f"Feature matrix: {data.shape}  (serve features present on {cov:.0f}% of rows)")
    return data, hist


def _randomise_perspective(F: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Flip a random half of rows to player-2's perspective; y=1 means player 1 won."""
    rng = np.random.default_rng(42)
    flip = rng.random(len(F)) < 0.5

    data = F[["date", "tour", "surface", "level", "min_nm", "winner", "loser"]].copy()
    data["flip"] = flip
    data["elo_diff"] = np.where(flip, -F["elo_diff_w"], F["elo_diff_w"])
    data["ov_diff"] = np.where(flip, -F["ov_diff_w"], F["ov_diff_w"])
    data["p_elo"] = np.where(flip, 1 - F["p_elo_w"], F["p_elo_w"])
    data["form_p1"] = np.where(flip, F["form_l"], F["form_w"])
    data["form_p2"] = np.where(flip, F["form_w"], F["form_l"])
    data["h2h"] = np.where(flip, -F["h2h_w"], F["h2h_w"])
    data["rest_p1"] = np.where(flip, F["rest_l"], F["rest_w"])
    data["rest_p2"] = np.where(flip, F["rest_w"], F["rest_l"])
    data["rank_diff"] = np.where(flip, -F["rank_diff_w"], F["rank_diff_w"])
    data["spw_diff"] = np.where(flip, F["spw_l"] - F["spw_w"], F["spw_w"] - F["spw_l"])
    data["rpw_diff"] = np.where(flip, F["rpw_l"] - F["rpw_w"], F["rpw_w"] - F["rpw_l"])
    data["ace_diff"] = np.where(flip, F["ace_l"] - F["ace_w"], F["ace_w"] - F["ace_l"])
    data["bps_diff"] = np.where(flip, F["bps_l"] - F["bps_w"], F["bps_w"] - F["bps_l"])
    data["sos_diff"] = np.where(flip, F["sos_l"] - F["sos_w"], F["sos_w"] - F["sos_l"])
    data["surface_clay"] = (F["surface"] == "Clay").astype(int)
    data["surface_grass"] = (F["surface"] == "Grass").astype(int)
    data["tour_wta"] = (F["tour"] == "wta").astype(int)
    data["y"] = np.where(flip, 0, 1)
    return data
