"""Central configuration for the tennis match predictor.

Every tunable constant lives here so experiments are reproducible and the rest of
the package stays free of magic numbers. The defaults reproduce the values used in
the original research notebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """All tunable knobs for the data, Elo replay, features and model.

    The defaults match the notebook that this package was refactored from. Override
    any field at construction time, e.g. ``Config(k_base=300, start_year=2010)``.
    """

    # -- Data range --------------------------------------------------------
    data_dir: str = "tennis_data"
    start_year: int = 2000
    end_year: int = 2026
    burnin_until: str = "2003-01-01"  # ignore matches before ratings stabilise
    surfaces: List[str] = field(default_factory=lambda: ["Hard", "Clay", "Grass"])
    tours: List[str] = field(default_factory=lambda: ["atp", "wta"])

    # -- Elo --------------------------------------------------------------
    # K-factor decays with experience: K(n) = k_base / (n + k_offset) ** k_exp
    k_base: float = 250.0
    k_offset: int = 5
    k_exp: float = 0.4
    # Idle-time rating decay toward 1500 (regression to the mean during layoffs).
    decay_per_day: float = 0.0008
    decay_grace: int = 30      # days of inactivity before decay starts
    decay_cap: float = 0.30    # max fraction of rating-above-1500 that can decay away

    # Optional margin-of-victory weighting of the Elo update (off by default).
    use_margin_elo: bool = False
    margin_strength: float = 0.5

    # -- Adaptive surface blend -------------------------------------------
    # The match rating blends overall and surface Elo. The surface weight shrinks
    # toward zero when a player has little history on that surface, so an unplayed
    # surface (rating still seeded at 1500) cannot poison the blend.
    blend_surf: float = 0.7      # max weight on the surface rating
    use_adaptive_surf: bool = True
    surf_shrink_c: int = 20      # half-weight reached at this many surface matches

    # -- Feature engineering ----------------------------------------------
    form_window: int = 10        # matches in the rolling win-rate "form" feature
    serve_window: int = 20       # matches in the trailing serve/return rollups
    sos_window: int = 20         # matches in the strength-of-schedule rollup
    min_history: int = 5         # min matches per player to enter train/eval pools
    reliable_min_nm: int = 10    # below this -> low-confidence flag on a prediction

    # -- XGBoost ----------------------------------------------------------
    test_fraction: float = 0.2   # most-recent share of matches held out for testing
    xgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
        eval_metric="logloss",
    ))

    # -- Closing-line-value backtest --------------------------------------
    # tennis-data.co.uk season files (xlsx). Empty -> CLV step is skipped.
    clv_odds_files: List[str] = field(default_factory=list)
    clv_max_date_gap: int = 16   # days tolerance when joining odds to predictions
    clv_levels: List[str] = field(default_factory=lambda: ["G", "M", "A", "F", "D"])

    @property
    def model_path(self) -> str:
        return "tennis_elo_xgb_model.pkl"
