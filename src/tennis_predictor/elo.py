"""Surface-adjusted Elo ratings with idle decay and an adaptive surface blend.

Each player keeps an overall rating plus one rating per surface, all keyed by
``(tour, name)`` so ATP and WTA never mix. The match rating is a shrinkage blend of
the two: a player with little history on the match surface leans on their overall
rating instead of a barely-moved 1500 seed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config


def k_factor(n: int, config: Config) -> float:
    """Experience-decayed K: large early, shrinking as the sample grows."""
    return config.k_base / (n + config.k_offset) ** config.k_exp


def expected(ra: float, rb: float) -> float:
    """Standard Elo expectation that A beats B."""
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def surf_weight(n_surface: int, config: Config) -> float:
    """Weight on the surface rating, shrunk toward 0 when surface history is thin."""
    if not config.use_adaptive_surf:
        return config.blend_surf
    return config.blend_surf * n_surface / (n_surface + config.surf_shrink_c)


def blend_elo(overall_r: float, surf_r: float, n_surface: int, config: Config) -> float:
    """Blend overall and surface ratings using the adaptive surface weight."""
    w = surf_weight(n_surface, config)
    return w * surf_r + (1 - w) * overall_r


@dataclass
class EloState:
    """Mutable rating bookkeeping, also reused by the live predictor.

    All dicts are keyed by ``(tour, player_name)``.
    """

    overall: Dict = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    surf: Dict = None  # surface -> {key: rating}
    n_all: Dict = field(default_factory=lambda: defaultdict(int))
    n_surf: Dict = None  # surface -> {key: count}
    last_played: Dict = field(default_factory=dict)

    @classmethod
    def new(cls, surfaces) -> "EloState":
        return cls(
            surf={s: defaultdict(lambda: 1500.0) for s in surfaces},
            n_surf={s: defaultdict(int) for s in surfaces},
        )

    def decay(self, key, today, config: Config) -> None:
        """Regress a player's ratings toward 1500 after a long layoff (in place)."""
        if key not in self.last_played:
            return
        idle = (today - self.last_played[key]).days
        if idle <= config.decay_grace:
            return
        f = min(config.decay_cap, config.decay_per_day * (idle - config.decay_grace))
        self.overall[key] = 1500 + (self.overall[key] - 1500) * (1 - f)
        for s in self.surf:
            self.surf[s][key] = 1500 + (self.surf[s][key] - 1500) * (1 - f)


def replay_elo(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, EloState]:
    """Replay every match chronologically, recording each player's PRE-match ratings.

    Returns a per-match DataFrame (one row per historical match, ratings as they stood
    *entering* that match — so it is leakage-free for downstream training) and the
    final ``EloState`` for live prediction.
    """
    state = EloState.new(config.surfaces)
    rows = []

    for r in df.itertuples():
        s, d, tour = r.surface, r.match_date, r.tour
        w, l = (tour, r.winner_name), (tour, r.loser_name)
        state.decay(w, d, config)
        state.decay(l, d, config)

        rw = blend_elo(state.overall[w], state.surf[s][w], state.n_surf[s][w], config)
        rl = blend_elo(state.overall[l], state.surf[s][l], state.n_surf[s][l], config)
        p_w = expected(rw, rl)

        rows.append({
            "date": d, "tour": tour, "level": r.tourney_level, "surface": s,
            "winner": r.winner_name, "loser": r.loser_name,
            "elo_w": rw, "elo_l": rl, "p_w_elo": p_w,
            "ov_w": state.overall[w], "ov_l": state.overall[l],
            "nm_w": state.n_all[w], "nm_l": state.n_all[l],
            "rank_w": getattr(r, "winner_rank", np.nan),
            "rank_l": getattr(r, "loser_rank", np.nan),
            "dominance_w": r.dominance_w, "score_clean": r.score_clean,
            "w_svpt": getattr(r, "w_svpt", np.nan), "l_svpt": getattr(r, "l_svpt", np.nan),
            "w_1stWon": getattr(r, "w_1stWon", np.nan), "l_1stWon": getattr(r, "l_1stWon", np.nan),
            "w_2ndWon": getattr(r, "w_2ndWon", np.nan), "l_2ndWon": getattr(r, "l_2ndWon", np.nan),
            "w_ace": getattr(r, "w_ace", np.nan), "l_ace": getattr(r, "l_ace", np.nan),
            "w_bpSaved": getattr(r, "w_bpSaved", np.nan), "l_bpSaved": getattr(r, "l_bpSaved", np.nan),
            "w_bpFaced": getattr(r, "w_bpFaced", np.nan), "l_bpFaced": getattr(r, "l_bpFaced", np.nan),
        })

        # Update ratings toward the realised result (winner gained 1 - p_w of an "upset").
        mult = 1.0
        if config.use_margin_elo and r.score_clean:
            mult = 1.0 + config.margin_strength * (2 * r.dominance_w - 1)
        kw = k_factor(state.n_all[w], config) * mult
        kl = k_factor(state.n_all[l], config) * mult
        ksw = k_factor(state.n_surf[s][w], config) * mult
        ksl = k_factor(state.n_surf[s][l], config) * mult
        delta = 1 - p_w
        state.overall[w] += kw * delta
        state.overall[l] -= kl * delta
        state.surf[s][w] += ksw * delta
        state.surf[s][l] -= ksl * delta
        state.n_all[w] += 1
        state.n_all[l] += 1
        state.n_surf[s][w] += 1
        state.n_surf[s][l] += 1
        state.last_played[w] = d
        state.last_played[l] = d

    elo = pd.DataFrame(rows)
    print(f"Replayed {len(elo):,} matches; {len(state.n_all):,} players rated.")
    return elo, state
