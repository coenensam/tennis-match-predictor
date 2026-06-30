"""CLI: predict a single match from a saved model.

    python scripts/predict.py "Carlos Alcaraz" "Jannik Sinner" --surface Clay --tour atp
"""
from __future__ import annotations

import argparse

from tennis_predictor import MatchPredictor, format_prediction


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict one tennis match.")
    ap.add_argument("player1")
    ap.add_argument("player2")
    ap.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass"])
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--date", default=None, help="match date YYYY-MM-DD (default: today)")
    ap.add_argument("-m", "--model", default="tennis_elo_xgb_model.pkl")
    args = ap.parse_args()

    predictor = MatchPredictor.load(args.model)
    out = predictor.predict(args.player1, args.player2, surface=args.surface,
                            tour=args.tour, match_date=args.date)
    print(format_prediction(out))


if __name__ == "__main__":
    main()
