"""CLI: run the full pipeline and save a reusable predictor.

    python scripts/train.py --clv-odds 2025.xlsx 2026.xlsx
    python scripts/train.py --start-year 2010 --no-clv -o model.pkl
"""
from __future__ import annotations

import argparse

from tennis_predictor import Config, run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the tennis match predictor.")
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--data-dir", default="tennis_data")
    ap.add_argument("--clv-odds", nargs="*", default=[],
                    help="tennis-data.co.uk season xlsx files for the CLV backtest")
    ap.add_argument("--no-clv", action="store_true", help="skip the CLV backtest")
    ap.add_argument("-o", "--out", default="tennis_elo_xgb_model.pkl")
    args = ap.parse_args()

    config = Config(
        start_year=args.start_year,
        end_year=args.end_year,
        data_dir=args.data_dir,
        clv_odds_files=args.clv_odds,
    )
    result = run_pipeline(config, run_clv=not args.no_clv)
    result.predictor.save(args.out)


if __name__ == "__main__":
    main()
