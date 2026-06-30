"""Surface-adjusted Elo + XGBoost tennis match-outcome predictor.

Public API:
    >>> from tennis_predictor import Config, run_pipeline
    >>> result = run_pipeline(Config())
    >>> result.predictor.predict("Carlos Alcaraz", "Jannik Sinner", surface="Clay")
"""
from .clv import run_clv_backtest, shin_devig
from .config import Config
from .features import FEATURES
from .model import accuracy_by_tier, train
from .pipeline import PipelineResult, run_pipeline
from .predict import MatchPredictor, format_prediction

__version__ = "1.0.0"

__all__ = [
    "Config",
    "run_pipeline",
    "PipelineResult",
    "MatchPredictor",
    "format_prediction",
    "train",
    "accuracy_by_tier",
    "run_clv_backtest",
    "shin_devig",
    "FEATURES",
]
