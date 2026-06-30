# Tennis Match Predictor — Surface-Adjusted Elo + XGBoost

A probabilistic model for ATP & WTA singles outcomes, built end-to-end and **benchmarked
against the sharp bookmaker closing line** — the gold-standard test of whether a sports
model actually knows something the market does not.

The headline result is an honest one: the model is well-calibrated and beats a strong Elo
baseline out-of-sample, **but it does *not* beat the Pinnacle closing line at tour level.**
Knowing *where* a model has edge — and being able to prove it doesn't — is the point.

---

## Results

Trained on **801k ATP+WTA matches (2000–2026)**, evaluated on a strictly chronological hold-out.

| Model | Test log-loss | Test accuracy |
|---|---|---|
| Surface-adjusted Elo (baseline) | 0.5979 | 67.1% |
| **XGBoost (18 features)** | **0.5825** | **68.4%** |

The XGBoost layer adds real information on top of Elo (lower log-loss and Brier on 132k
held-out matches), and is well-calibrated across the probability range.

### Does it beat the market? (Closing-Line-Value backtest)

Benchmarked against the de-vigged **Pinnacle** close on 2,409 matched tour-level matches (2025–26):

| Benchmark | log-loss | accuracy |
|---|---|---|
| Pinnacle de-vig (the market) | **0.5980** | 67.6% |
| Betfair Exchange | 0.5901 | 67.2% |
| XGBoost (this model) | 0.6145 | 65.9% |
| raw Elo | 0.6271 | 65.3% |

**Encompassing regression** `outcome ~ b_close·logit(close) + b_model·logit(model)`:

```
b_close = 1.128    b_model = -0.141
```

`b_model < 0` ⇒ at ATP/WTA tour level the model carries **no information beyond the closing
line**, and a significance-tested flat-stake betting simulation confirms no edge survives
(`|t| < 2` at every threshold). The market is efficient here. The interesting question this
raises — and the model's design is built to chase — is whether edge exists in *softer* markets
(Challengers, ITF, low-liquidity exchanges) where the line is less sharp.

> Reporting a negative result honestly is deliberate. The encompassing coefficient is a far
> more rigorous success criterion than "accuracy went up," and it's the criterion that matters
> for any real-money or research use.

---

## What's interesting here (engineering & modelling)

- **Leakage discipline.** Every feature is a player's state *entering* the match. Serve/return
  rollups are recorded *before* the current match is folded into history, so the exact same
  construction runs at train and inference time. (An earlier version silently fed post-match
  box scores into training — caught by the inference-time distribution mismatch and fixed.)
- **Surface-adjusted Elo with adaptive shrinkage.** Each player has an overall rating plus a
  rating per surface; the surface weight shrinks toward zero when surface history is thin, so an
  unplayed surface (still seeded at 1500) can't poison the blend.
- **Idle-time rating decay** toward the mean during layoffs, computed locally at inference so a
  prediction never mutates global state.
- **Proper market benchmarking.** Shin de-vig (corrects favourite-longshot bias, solved by
  bisection for exact complementarity), an encompassing regression, and a t-stat'd ROI sim.
- **Honest evaluation throughout** — chronological splits, a strong baseline the model must beat,
  per-tier breakdowns, and calibration curves.

---

## Project layout

```
tennis-match-predictor/
├── src/tennis_predictor/
│   ├── config.py      # all tunable constants in one dataclass
│   ├── data.py        # download + clean Sackmann ATP/WTA data
│   ├── elo.py         # surface-adjusted Elo replay + decay + adaptive blend
│   ├── features.py    # leak-free feature engineering (18 features)
│   ├── model.py       # XGBoost training + time-split evaluation
│   ├── clv.py         # closing-line-value backtest, de-vig, encompassing test
│   ├── predict.py     # single-match inference + model persistence
│   └── pipeline.py    # end-to-end orchestration
├── scripts/           # train.py / predict.py CLIs
├── tests/             # de-vig + name-matching unit tests
└── notebooks/         # original research notebook (full exploration)
```

---

## Quickstart

```bash
pip install -e .

# Train end-to-end (downloads ~800k matches on first run) and save a predictor
python scripts/train.py --no-clv -o tennis_elo_xgb_model.pkl

# Add the closing-line-value backtest (needs tennis-data.co.uk season xlsx files)
python scripts/train.py --clv-odds 2025.xlsx 2026.xlsx

# Predict a single match from the saved model
python scripts/predict.py "Carlos Alcaraz" "Jannik Sinner" --surface Clay --tour atp
```

Or from Python:

```python
from tennis_predictor import Config, run_pipeline

result = run_pipeline(Config())
print(result.training.metrics)

out = result.predictor.predict("Iga Swiatek", "Aryna Sabalenka",
                               surface="Clay", tour="wta")
print(out["p1_win"], out["low_confidence"])
```

```bash
pytest        # run the unit tests
```

---

## Method notes

**Features (18).** Elo difference, overall-rating difference, Elo win-prob, rolling form (both
players), head-to-head, rest days (both), rank difference, min match count, surface one-hots,
tour flag, and four leak-free serve/return diffs (serve-points-won, return-points-won, ace rate,
break-points-saved) plus a strength-of-schedule diff (recent average opponent rank).

**Why a low-confidence flag, not a hard cutoff.** A calibration analysis (ECE across
strength-of-schedule mismatch) showed the model stays calibrated on cross-level matchups; the
only badly-calibrated slice is genuine data poverty (no ranked-opponent history). So predictions
on thin histories are *flagged*, not silently trusted.

## Data & license

Match data: [Jeff Sackmann's tennis_atp / tennis_wta](https://github.com/JeffSackmann)
(CC BY-NC-SA 4.0), downloaded at runtime — **not** redistributed here. Closing odds:
[tennis-data.co.uk](http://www.tennis-data.co.uk/) (supply your own season files). Code: MIT.
