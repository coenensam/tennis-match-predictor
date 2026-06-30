"""Closing-line-value backtest: does the model know anything the market does not?

Calibration tells you the model agrees with itself. This is the harder test — it
benchmarks the model's probabilities against the de-vigged Pinnacle closing line,
the gold-standard measure of market efficiency. The sharpest single statistic is the
encompassing regression coefficient ``b_model``: if it is ~0 (or negative) the model
adds no information beyond the close, and there is no edge at this tier.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from .config import Config
from .features import FEATURES


# --------------------------------------------------------------------------- #
# Name matching between Sackmann ("First Last") and tennis-data ("Last F.")    #
# --------------------------------------------------------------------------- #
def _norm(s: object) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def keys_sackmann(name: str) -> list[str]:
    """Candidate match keys for a Sackmann 'First Last' name (surname|initial)."""
    toks = str(name).split()
    if len(toks) < 2:
        return [_norm(name) + "|"]
    init = _norm(toks[0])[:1]
    surname_toks = toks[1:]
    keys = [_norm(" ".join(surname_toks)) + "|" + init]
    if len(surname_toks) > 1:
        keys.append(_norm(surname_toks[0]) + "|" + init)
        keys.append(_norm(surname_toks[-1]) + "|" + init)
    return keys


def key_tennisdata(name: str) -> str:
    """Single match key for a tennis-data 'Surname F.' name."""
    toks = str(name).replace(".", " ").split()
    initials = []
    while (toks and len(toks[-1]) <= 2 and toks[-1].replace(".", "").isalpha()
           and len(toks[-1]) > 0 and toks[-1][0].isupper()):
        initials.insert(0, toks.pop())
    last = _norm(" ".join(toks))
    init = _norm(initials[0])[:1] if initials else ""
    return last + "|" + init


# --------------------------------------------------------------------------- #
# De-vigging                                                                   #
# --------------------------------------------------------------------------- #
def shin_devig(odds_w: float, odds_l: float, tol: float = 1e-10, iters: int = 100) -> float:
    """Shin de-vig: remove the bookmaker overround while correcting favourite-longshot bias.

    Solves Shin's insider-trading parameter z by bisection so the two implied
    probabilities sum to exactly 1, then returns the winner-side probability.
    """
    pi_w, pi_l = 1 / odds_w, 1 / odds_l
    booksum = pi_w + pi_l
    if booksum <= 1:  # no overround -> fall back to proportional
        return pi_w / booksum
    lo, hi = 0.0, booksum - 1
    pw = pi_w / booksum
    for _ in range(iters):
        z = (lo + hi) / 2
        pw = (np.sqrt(z * z + 4 * (1 - z) * pi_w * pi_w / booksum) - z) / (2 * (1 - z))
        pl = (np.sqrt(z * z + 4 * (1 - z) * pi_l * pi_l / booksum) - z) / (2 * (1 - z))
        s = pw + pl
        if abs(s - 1) < tol:
            break
        lo, hi = (z, hi) if s > 1 else (lo, z)
    return pw / (pw + pl)


def proportional_devig(odds_w: float, odds_l: float) -> float:
    iw, il = 1 / odds_w, 1 / odds_l
    return iw / (iw + il)


# --------------------------------------------------------------------------- #
# Backtest                                                                     #
# --------------------------------------------------------------------------- #
def _load_odds(config: Config) -> pd.DataFrame:
    frames = []
    for f in config.clv_odds_files:
        try:
            frames.append(pd.read_excel(f))
            print(f"  loaded {f}: {len(frames[-1]):,} rows")
        except FileNotFoundError:
            print(f"  WARNING: {f} not found -- skipping")
    if not frames:
        raise FileNotFoundError("No odds files found; check Config.clv_odds_files.")
    odds = pd.concat(frames, ignore_index=True)
    if not pd.api.types.is_datetime64_any_dtype(odds["Date"]):
        odds["Date"] = pd.to_datetime(odds["Date"], unit="D", origin="1899-12-30")
    odds = odds[odds["Comment"].astype(str).str.lower().eq("completed")]
    odds = odds.dropna(subset=["PSW", "PSL"]).reset_index(drop=True)
    odds["kw"] = odds["Winner"].map(key_tennisdata)
    odds["kl"] = odds["Loser"].map(key_tennisdata)
    return odds


def run_clv_backtest(data: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Retrain strictly out-of-sample on the pre-odds window, then benchmark vs the close.

    Returns the matched prediction/odds frame and prints the headline comparison, the
    per-tier gap, the encompassing regression, and a significance-tested ROI sim.
    """
    odds = _load_odds(config)
    print(f"Odds rows (completed, Pinnacle present): {len(odds):,}")

    # Out-of-sample retrain: train only on matches before the odds window opens.
    d2 = data[(data["min_nm"] >= config.min_history)
              & (data["date"] >= config.burnin_until)].sort_values("date")
    odds_start = odds["Date"].min()
    tr_clv = d2[d2["date"] < odds_start]
    te_clv = d2[d2["date"] >= odds_start]
    print(f"CLV retrain: train<{odds_start.date()} ({len(tr_clv):,}), "
          f"eval>= ({len(te_clv):,})")

    model = xgb.XGBClassifier(**config.xgb_params)
    model.fit(tr_clv[FEATURES], tr_clv["y"], verbose=False)
    pred = model.predict_proba(te_clv[FEATURES])[:, 1]

    preds = te_clv[["date", "tour", "level", "surface", "winner", "loser",
                    "flip", "min_nm"]].copy()
    preds["p_w_xgb"] = np.where(preds["flip"], 1 - pred, pred)
    preds["p_w_elo"] = np.where(preds["flip"], 1 - te_clv["p_elo"].values,
                                te_clv["p_elo"].values)
    preds = preds[preds["level"].isin(config.clv_levels)].copy()

    clv = _join_odds_to_preds(odds, preds, config)
    print(f"\nMatched {len(clv):,} / {len(odds):,} odds rows "
          f"({len(clv) / len(odds) * 100:.1f}%)")

    _report_headline(clv)
    _report_encompassing(clv)
    _report_roi(clv)
    return clv


def _join_odds_to_preds(odds, preds, config) -> pd.DataFrame:
    """Fuzzy-join each odds row to the nearest-dated prediction for the same pairing."""
    p_idx: dict = {}
    for i, r in preds.iterrows():
        for kw in keys_sackmann(r["winner"]):
            for kl in keys_sackmann(r["loser"]):
                p_idx.setdefault(tuple(sorted([kw, kl])), []).append(i)

    rows, used = [], set()
    for _, r in odds.iterrows():
        cands = p_idx.get(tuple(sorted([r["kw"], r["kl"]])), [])
        best, bd = None, 99
        for i in cands:
            if i in used:
                continue
            dd = abs((preds.at[i, "date"] - r["Date"]).days)
            if dd <= config.clv_max_date_gap and dd < bd:
                best, bd = i, dd
        if best is None:
            continue
        used.add(best)
        pr = preds.loc[best]
        same = any(k == r["kw"] for k in keys_sackmann(pr["winner"]))
        rows.append({
            "date": r["Date"], "series": r.get("Series"),
            "winner": r["Winner"], "loser": r["Loser"],
            "p_model": pr["p_w_xgb"] if same else 1 - pr["p_w_xgb"],
            "p_elo": pr["p_w_elo"] if same else 1 - pr["p_w_elo"],
            "p_pinn": proportional_devig(r["PSW"], r["PSL"]),
            "p_shin": shin_devig(r["PSW"], r["PSL"]),
            "PSW": r["PSW"], "PSL": r["PSL"],
            "MaxW": r.get("MaxW"), "MaxL": r.get("MaxL"),
        })
    return pd.DataFrame(rows)


def _report_headline(clv: pd.DataFrame) -> None:
    ll = lambda p: -np.log(np.clip(p, 1e-9, 1)).mean()
    acc = lambda p: (p > 0.5).mean()
    sep = "─" * 56
    print(f"\n{sep}\n{'':18s} {'log_loss':>10s} {'accuracy':>10s}\n{sep}")
    for label, col in [("Pinnacle de-vig", "p_pinn"), ("Shin de-vig", "p_shin"),
                       ("XGBoost", "p_model"), ("raw Elo", "p_elo")]:
        print(f"{label:18s} {ll(clv[col]):10.4f} {acc(clv[col]):10.4f}")


def _report_encompassing(clv: pd.DataFrame) -> None:
    """outcome ~ b_close*logit(close) + b_model*logit(model). b_model is the edge test."""
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
    p_sym = np.concatenate([clv.p_model, 1 - clv.p_model])
    pp_sym = np.concatenate([clv.p_pinn, 1 - clv.p_pinn])
    y_sym = np.concatenate([np.ones(len(clv)), np.zeros(len(clv))])
    lr = LogisticRegression(C=1e6).fit(np.column_stack([lg(pp_sym), lg(p_sym)]), y_sym)
    b_close, b_model = lr.coef_[0]
    print(f"\nEncompassing regression:  b_close={b_close:.3f}  b_model={b_model:.3f}")
    if abs(b_model) < 0.05:
        print("  -> b_model ~ 0: no information beyond the close. Look to softer markets.")
    elif b_model > 0:
        print("  -> b_model > 0: possible residual edge. Investigate further.")
    else:
        print("  -> b_model < 0: worse than the close even as a secondary signal.")


def _report_roi(clv: pd.DataFrame) -> None:
    """Flat-stake value-bet ROI at several edge thresholds, with a t-stat on each."""
    print("\nFlat 1-unit value bets settled at best-of-market odds:")
    print(f"{'threshold':>10s} {'bets':>6s} {'ROI%':>8s} {'t-stat':>8s} {'significant':>12s}")
    for t_val in [0.02, 0.04, 0.06, 0.08, 0.10]:
        pnl = []
        for r in clv.itertuples():
            if pd.isna(r.MaxW) or pd.isna(r.MaxL):
                continue
            if r.p_model - 1 / r.PSW > t_val:
                pnl.append(r.MaxW - 1)
            if (1 - r.p_model) - 1 / r.PSL > t_val:
                pnl.append(-1)
        pnl = np.array(pnl)
        if len(pnl) > 1:
            roi = pnl.mean() * 100
            tstat = pnl.mean() / (pnl.std() / np.sqrt(len(pnl)))
            sig = "YES" if abs(tstat) >= 2 else "no (noise)"
            print(f"{t_val:10.2f} {len(pnl):6d} {roi:8.2f} {tstat:8.2f} {sig:>12s}")
