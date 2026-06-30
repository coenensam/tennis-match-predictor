"""End-to-end orchestration: data -> Elo -> features -> model -> (optional) CLV.

``run_pipeline`` ties the modules together and returns a ready-to-use predictor. It is
the single entry point used by ``scripts/train.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .clv import run_clv_backtest
from .config import Config
from .data import download_data, load_matches
from .elo import replay_elo
from .features import build_features
from .model import TrainResult, train
from .predict import MatchPredictor


@dataclass
class PipelineResult:
    predictor: MatchPredictor
    training: TrainResult


def run_pipeline(config: Config | None = None, run_clv: bool = True) -> PipelineResult:
    """Run the full training pipeline and return the fitted predictor + metrics."""
    config = config or Config()

    print("\n[1/5] Downloading data ...")
    download_data(config)

    print("\n[2/5] Loading & cleaning matches ...")
    df = load_matches(config)

    print("\n[3/5] Replaying surface-adjusted Elo ...")
    elo_df, elo_state = replay_elo(df, config)

    print("\n[4/5] Building leak-free features ...")
    data, hist = build_features(elo_df, config)

    print("\n[5/5] Training XGBoost ...")
    result = train(data, config)

    if run_clv and config.clv_odds_files:
        print("\n[+] Closing-line-value backtest ...")
        try:
            run_clv_backtest(data, config)
        except FileNotFoundError as e:
            print(f"  CLV skipped: {e}")

    predictor = MatchPredictor(model=result.model, elo=elo_state, hist=hist, config=config)
    return PipelineResult(predictor=predictor, training=result)
