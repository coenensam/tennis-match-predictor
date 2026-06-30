"""Train and evaluate the XGBoost win-probability model on a time-ordered split."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss

from .config import Config
from .features import FEATURES


@dataclass
class TrainResult:
    model: xgb.XGBClassifier
    test: pd.DataFrame          # held-out rows with a "pred" column attached
    metrics: dict               # headline log-loss / brier / accuracy vs the Elo baseline


def _eligible(data: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Mature matches only: both players past the min-history burn-in."""
    return (
        data[(data["min_nm"] >= config.min_history)
             & (data["date"] >= config.burnin_until)]
        .sort_values("date")
        .reset_index(drop=True)
    )


def time_split(data: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the most recent ``test_fraction`` of matches is the test set."""
    d2 = _eligible(data, config)
    cut = d2["date"].quantile(1 - config.test_fraction)
    return d2[d2["date"] < cut], d2[d2["date"] >= cut]


def train(data: pd.DataFrame, config: Config) -> TrainResult:
    """Fit XGBoost on the training split and report test metrics vs the raw-Elo baseline.

    The raw-Elo probability (``p_elo``) is the honest baseline: the model has to add
    information *beyond* Elo to justify itself.
    """
    tr, te = time_split(data, config)
    print(f"train={len(tr):,}  test={len(te):,}  cut={te['date'].min().date()}  "
          f"features={len(FEATURES)}")

    model = xgb.XGBClassifier(**config.xgb_params)
    model.fit(tr[FEATURES], tr["y"], eval_set=[(te[FEATURES], te["y"])], verbose=False)
    pred = model.predict_proba(te[FEATURES])[:, 1]

    te = te.copy()
    te["pred"] = pred
    metrics = {
        "xgb_log_loss": log_loss(te["y"], pred),
        "elo_log_loss": log_loss(te["y"], te["p_elo"]),
        "xgb_brier": brier_score_loss(te["y"], pred),
        "elo_brier": brier_score_loss(te["y"], te["p_elo"]),
        "xgb_accuracy": float(((pred > 0.5) == te["y"]).mean()),
    }
    metrics["xgb_beats_elo"] = metrics["xgb_log_loss"] < metrics["elo_log_loss"]

    print(f"\nTest log_loss: XGB={metrics['xgb_log_loss']:.4f}  "
          f"raw-Elo={metrics['elo_log_loss']:.4f}")
    print(f"Test accuracy: {metrics['xgb_accuracy']:.4f}  "
          f"| XGB beats Elo: {metrics['xgb_beats_elo']}")
    for t in config.tours:
        m = (te["tour"] == t).values
        if m.any():
            print(f"  {t}: XGB log_loss={log_loss(te['y'][m], pred[m]):.4f}  n={m.sum():,}")

    return TrainResult(model=model, test=te, metrics=metrics)


def accuracy_by_tier(result: TrainResult, min_n: int = 100) -> pd.DataFrame:
    """Segment test accuracy / log-loss by tour and tournament level."""
    te = result.test
    g = te.groupby(["tour", "level"]).agg(
        n=("y", "size"),
        acc=("y", lambda s: ((te.loc[s.index, "pred"] > 0.5) == s).mean()),
        logloss=("y", lambda s: log_loss(s, te.loc[s.index, "pred"], labels=[0, 1])
                 if s.nunique() > 1 else np.nan),
    )
    return g[g["n"] >= min_n].sort_values(["tour", "n"], ascending=[True, False])
