"""Single-match inference and model persistence.

``MatchPredictor`` bundles the trained XGBoost model with the final Elo and feature
histories so a fresh win probability can be produced for any pairing, reconstructing
each feature exactly as it was built during training (no train/inference skew).
"""
from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .elo import EloState, blend_elo, expected, surf_weight
from .features import FEATURES, FeatureHistory, _avg


@dataclass
class MatchPredictor:
    model: object
    elo: EloState
    hist: FeatureHistory
    config: Config

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #
    def predict(self, p1: str, p2: str, surface: str, tour: str = "atp",
                match_date: str | None = None) -> dict:
        """Return P(p1 beats p2) plus the diagnostic components behind it.

        Names must match the Sackmann 'First Last' spelling. The result dict carries a
        ``low_confidence`` reason list so callers can widen their edge or skip thin matches.
        """
        cfg = self.config
        assert surface in cfg.surfaces, f"surface must be one of {cfg.surfaces}"
        today = pd.Timestamp(match_date or dt.date.today())
        k1, k2 = (tour, p1), (tour, p2)

        def decayed(key):
            g = 1.0
            if key in self.elo.last_played:
                idle = (today - self.elo.last_played[key]).days
                if idle > cfg.decay_grace:
                    g = 1.0 - min(cfg.decay_cap, cfg.decay_per_day * (idle - cfg.decay_grace))
            ov = 1500 + (self.elo.overall[key] - 1500) * g
            sr = 1500 + (self.elo.surf[surface][key] - 1500) * g
            return ov, sr

        ov1, s1 = decayed(k1)
        ov2, s2 = decayed(k2)
        r1 = blend_elo(ov1, s1, self.elo.n_surf[surface][k1], cfg)
        r2 = blend_elo(ov2, s2, self.elo.n_surf[surface][k2], cfg)

        h = self.hist
        feat = {
            "elo_diff": r1 - r2, "ov_diff": ov1 - ov2, "p_elo": expected(r1, r2),
            "form_p1": np.mean(h.form[k1][-cfg.form_window:]) if h.form[k1] else 0.5,
            "form_p2": np.mean(h.form[k2][-cfg.form_window:]) if h.form[k2] else 0.5,
            "h2h": h.h2h[(k1, k2)][0] - h.h2h[(k1, k2)][1],
            "rest_p1": min((today - h.last_date[k1]).days, 60) if k1 in h.last_date else 30,
            "rest_p2": min((today - h.last_date[k2]).days, 60) if k2 in h.last_date else 30,
            "rank_diff": 0,
            "min_nm": min(self.elo.n_all[k1], self.elo.n_all[k2]),
            "surface_clay": int(surface == "Clay"),
            "surface_grass": int(surface == "Grass"),
            "tour_wta": int(tour == "wta"),
            "spw_diff": _avg(h.spw[k1]) - _avg(h.spw[k2]),
            "rpw_diff": _avg(h.rpw[k1]) - _avg(h.rpw[k2]),
            "ace_diff": _avg(h.ace[k1]) - _avg(h.ace[k2]),
            "bps_diff": _avg(h.bps[k1]) - _avg(h.bps[k2]),
            "sos_diff": _avg(h.sos[k1]) - _avg(h.sos[k2]),
        }
        X = pd.DataFrame([feat])[FEATURES]
        p1_win = float(self.model.predict_proba(X)[:, 1][0])

        reasons = []
        if np.isnan(_avg(h.sos[k1])) or np.isnan(_avg(h.sos[k2])):
            reasons.append("no ranked-opponent history")
        if np.isnan(feat["spw_diff"]):
            reasons.append("no serve history")
        if feat["min_nm"] < cfg.reliable_min_nm:
            reasons.append(f"thin history (min_nm={feat['min_nm']})")

        return {
            "p1": p1, "p2": p2, "surface": surface, "tour": tour,
            "p1_win": p1_win, "p2_win": 1 - p1_win,
            "elo": (r1, r2), "n": (self.elo.n_all[k1], self.elo.n_all[k2]),
            "features": feat, "low_confidence": reasons,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, path: str | None = None) -> str:
        path = path or self.config.model_path
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved predictor -> {path} "
              f"({len(self.elo.n_all):,} players, {len(FEATURES)} features)")
        return path

    @staticmethod
    def load(path: str) -> "MatchPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)


def format_prediction(out: dict) -> str:
    """Human-readable one-block summary of a prediction dict."""
    lines = [
        f"{out['p1']} vs {out['p2']}  ({out['surface']}, {out['tour'].upper()})",
        f"  Elo: {out['elo'][0]:.0f} vs {out['elo'][1]:.0f}  "
        f"(matches: {out['n'][0]} vs {out['n'][1]})",
        f"  P({out['p1']}) = {out['p1_win']:.3f}   "
        f"P({out['p2']}) = {out['p2_win']:.3f}",
    ]
    if out["low_confidence"]:
        lines.append("  [!] LOW CONFIDENCE: " + "; ".join(out["low_confidence"]))
    return "\n".join(lines)
