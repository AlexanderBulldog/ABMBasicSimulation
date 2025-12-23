from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


FEATURES = [
    "inventory",
    "last_demand",
    "cash",
    "debt",
    "price",
    "n_workers",
    "bank_available_credit",
    "wage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a firm markup policy model.")
    parser.add_argument(
        "--data",
        type=str,
        default="output/ml_datasets/firm_policy.csv",
        help="Input dataset CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/models/firm_policy.pkl",
        help="Output model path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out_path = Path(args.out)

    df = pd.read_csv(data_path)
    missing = [c for c in FEATURES + ["realized_markup"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {data_path}: {', '.join(missing)}")

    X = df[FEATURES].to_numpy()
    y = df["realized_markup"].clip(0.0, 0.5).to_numpy()

    model = GradientBoostingRegressor(random_state=0)
    model.fit(X, y)
    model.feature_order_ = list(FEATURES)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
