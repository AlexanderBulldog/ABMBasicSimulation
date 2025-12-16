from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_operator import PARAM_BOUNDS  # noqa: E402


@dataclass(frozen=True)
class YTransform:
    name: str  # "identity" | "log1p" | "signed_log1p"

    def forward(self, y: np.ndarray) -> np.ndarray:
        if self.name == "identity":
            return y
        if self.name == "log1p":
            return np.log1p(np.clip(y, 0.0, None))
        if self.name == "signed_log1p":
            return np.sign(y) * np.log1p(np.abs(y))
        raise ValueError(f"Unknown transform: {self.name}")

    def inverse(self, t: np.ndarray) -> np.ndarray:
        if self.name == "identity":
            return t
        if self.name == "log1p":
            return np.expm1(t)
        if self.name == "signed_log1p":
            return np.sign(t) * np.expm1(np.abs(t))
        raise ValueError(f"Unknown transform: {self.name}")

    def inverse_sigma(self, mu_t: np.ndarray, sigma_t: np.ndarray) -> np.ndarray:
        """Approximate std in original space via delta method."""
        if self.name == "identity":
            return sigma_t
        if self.name == "log1p":
            return sigma_t * np.exp(mu_t)
        if self.name == "signed_log1p":
            return sigma_t * np.exp(np.abs(mu_t))
        raise ValueError(f"Unknown transform: {self.name}")


def choose_transform(metric: str, y: pd.Series) -> YTransform:
    """Pick a simple, robust transform for heavy-tailed outputs."""
    name = "identity"
    metric_l = metric.lower()
    is_financial = any(k in metric_l for k in ("debt", "deposit", "loans", "equity", "cash"))
    if is_financial:
        y_np = y.to_numpy(dtype=float)
        if np.nanmin(y_np) < 0:
            name = "signed_log1p"
        else:
            name = "log1p"
    return YTransform(name=name)


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV/Parquet dataset produced by run_lhs.py."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def default_metrics(df: pd.DataFrame) -> List[str]:
    """Select metric columns heuristically."""
    cols = []
    for c in df.columns:
        if c in ("bad_run", "error", "seed"):
            continue
        if c in PARAM_BOUNDS:
            continue
        if c.endswith(("_mean", "_rate", "_share")):
            cols.append(c)
    return cols


def train_gpr(X_train, y_train):
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gpr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    random_state=42,
                    n_restarts_optimizer=2,
                ),
            ),
        ]
    )
    gpr.fit(X_train, y_train)
    return gpr


def train_rf(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_test, y_test) -> Tuple[float, float]:
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    return r2, rmse


def cv_r2_scores(train_fn, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[float, float]:
    """Compute mean/std of R2 across KFold CV."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: List[float] = []
    for train_idx, test_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        model = train_fn(X_train, y_train)
        r2, _ = evaluate_model(model, X_test, y_test)
        scores.append(float(r2))
    return float(np.mean(scores)), float(np.std(scores))


def cv_r2_scores_transformed(
    train_fn,
    X: pd.DataFrame,
    y: pd.Series,
    transform: YTransform,
    n_splits: int = 5,
) -> Tuple[float, float]:
    """Compute mean/std of R2 across KFold CV (trained on transformed y, scored on original y)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: List[float] = []
    y_np = y.to_numpy(dtype=float)
    for train_idx, test_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_np[train_idx]
        y_test = y_np[test_idx]
        y_train_t = transform.forward(y_train)
        model = train_fn(X_train, y_train_t)
        preds_t = model.predict(X_test)
        preds = transform.inverse(np.asarray(preds_t, dtype=float))
        scores.append(float(r2_score(y_test, preds)))
    return float(np.mean(scores)), float(np.std(scores))


def load_targets(path: Path) -> Dict[str, Dict[str, float]]:
    """Load target specs from JSON: {metric: {target: val, sigma_obs: x, sigma_model: y}}."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    out: Dict[str, Dict[str, float]] = {}
    for k, v in cfg.items():
        target = v.get("target")
        if target is None and "low" in v and "high" in v:
            target = 0.5 * (v["low"] + v["high"])
        sigma_obs = v.get("sigma_obs", 0.0)
        sigma_model = v.get("sigma_model", 0.0)
        out[k] = {"target": target, "sigma_obs": sigma_obs, "sigma_model": sigma_model}
    return out


def history_matching(
    X: pd.DataFrame,
    gpr_models: Dict[str, Pipeline],
    targets: Dict[str, Dict[str, float]],
    improb_threshold: float = 3.0,
    min_r2: float = 0.2,
    metric_quality: Dict[str, float] | None = None,
    metric_transforms: Dict[str, YTransform] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute improbability I(θ) and return dataframe with flags + NROY subset."""
    records = []
    for idx in range(len(X)):
        row_params = X.iloc[idx]
        row_res = {**row_params.to_dict()}
        max_I = -np.inf
        used = 0
        for metric, cfg in targets.items():
            model = gpr_models.get(metric)
            if model is None or cfg.get("target") is None:
                continue
            if metric_quality is not None and metric_quality.get(metric, -np.inf) < min_r2:
                continue
            transform = metric_transforms.get(metric, YTransform("identity")) if metric_transforms else YTransform("identity")
            pred_t, std_t = model.predict(row_params.to_frame().T, return_std=True)
            mu_t = float(pred_t[0])
            sigma_emul_t = float(std_t[0])
            mu = float(transform.inverse(np.array([mu_t]))[0])
            sigma_emul = float(transform.inverse_sigma(np.array([mu_t]), np.array([sigma_emul_t]))[0])
            sigma_obs = cfg.get("sigma_obs", 0.0)
            sigma_model = cfg.get("sigma_model", 0.0)
            denom = np.sqrt(sigma_obs**2 + sigma_model**2 + sigma_emul**2)
            I = np.abs(mu - cfg["target"]) / max(denom, 1e-9)
            row_res[f"{metric}_pred"] = mu
            row_res[f"{metric}_sigma_emul"] = sigma_emul
            row_res[f"{metric}_I"] = I
            if I > max_I:
                max_I = I
            used += 1
        row_res["metrics_used"] = used
        if used == 0:
            row_res["I_max"] = np.nan
            row_res["nroy"] = False
        else:
            row_res["I_max"] = max_I
            row_res["nroy"] = max_I < improb_threshold
        records.append(row_res)
    df = pd.DataFrame(records)
    nroy_df = df[df["nroy"]].copy()
    return df, nroy_df


def refined_intervals(nroy_df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """Compute refined parameter intervals from NROY subset."""
    rows = []
    for p, (low, high) in bounds.items():
        if p not in nroy_df:
            continue
        series = nroy_df[p].dropna()
        if series.empty:
            continue
        p05 = float(series.quantile(0.05))
        p95 = float(series.quantile(0.95))
        rmin = float(series.min())
        rmax = float(series.max())
        width0 = high - low
        width1 = rmax - rmin
        shrink = 0.0 if width0 <= 0 else max(0.0, (1.0 - width1 / width0) * 100.0)
        rows.append(
            {
                "param": p,
                "initial_low": low,
                "initial_high": high,
                "nroy_min": rmin,
                "nroy_max": rmax,
                "nroy_p05": p05,
                "nroy_p95": p95,
                "shrink_pct": shrink,
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Train emulator (GPR + RF baseline) and optional history matching.")
    parser.add_argument(
        "--data",
        type=str,
        default="output/datasets/lhs_runs.csv",
        help="Path to dataset from run_lhs.py (CSV or Parquet)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Comma-separated metric columns; default: *_mean, *_rate, *_share",
    )
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split size")
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help="JSON file with target, sigma_obs, sigma_model per metric for history matching",
    )
    parser.add_argument(
        "--improb-threshold",
        type=float,
        default=3.0,
        help="Threshold for NROY (I_max < threshold)",
    )
    parser.add_argument(
        "--min-r2-for-history-matching",
        type=float,
        default=0.2,
        help="Skip target metrics whose GPR R2 is below this threshold",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="output/results",
        help="Directory to save results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    # Filter bad runs / errors
    if "bad_run" in df:
        df = df[df["bad_run"] == False]  # noqa: E712
    if "error" in df:
        df = df[df["error"].isna() | (df["error"] == "")]

    param_cols = list(PARAM_BOUNDS.keys())
    metric_cols = [m for m in args.metrics.split(",") if m] if args.metrics else default_metrics(df)

    X = df[param_cols]
    metrics_scores = []
    gpr_models: Dict[str, Pipeline] = {}
    rf_models: Dict[str, RandomForestRegressor] = {}
    gpr_quality: Dict[str, float] = {}
    metric_transforms: Dict[str, YTransform] = {}

    X_train, X_test = train_test_split(X, test_size=args.test_size, random_state=42)
    idx_train = X_train.index
    idx_test = X_test.index

    for metric in metric_cols:
        y = df[metric]
        # Skip constant / near-constant outputs (common for DefaultsFirm_*)
        if y.dropna().nunique() < 5 or float(np.nanstd(y.to_numpy())) < 1e-12:
            metrics_scores.append(
                {
                    "metric": metric,
                    "skipped": True,
                    "reason": "low_variance",
                    "y_transform": "",
                    "gpr_r2": np.nan,
                    "gpr_rmse": np.nan,
                    "gpr_r2_cv_mean": np.nan,
                    "gpr_r2_cv_std": np.nan,
                    "rf_r2": np.nan,
                    "rf_rmse": np.nan,
                    "rf_r2_cv_mean": np.nan,
                    "rf_r2_cv_std": np.nan,
                }
            )
            continue

        transform = choose_transform(metric, y)
        metric_transforms[metric] = transform
        y_train = y.loc[idx_train]
        y_test = y.loc[idx_test]

        y_train_t = transform.forward(y_train.to_numpy(dtype=float))
        y_test_np = y_test.to_numpy(dtype=float)

        gpr = train_gpr(X_train, y_train_t)
        preds_gpr_t = gpr.predict(X_test)
        preds_gpr = transform.inverse(np.asarray(preds_gpr_t, dtype=float))
        gpr_r2 = float(r2_score(y_test_np, preds_gpr))
        gpr_rmse = float(np.sqrt(mean_squared_error(y_test_np, preds_gpr)))
        gpr_r2_cv_mean, gpr_r2_cv_std = cv_r2_scores_transformed(train_gpr, X, y, transform)
        gpr_models[metric] = gpr
        # Use CV mean R2 as a more reliable quality proxy than a single holdout split.
        gpr_quality[metric] = float(gpr_r2_cv_mean)

        rf = train_rf(X_train, y_train_t)
        preds_rf_t = rf.predict(X_test)
        preds_rf = transform.inverse(np.asarray(preds_rf_t, dtype=float))
        rf_r2 = float(r2_score(y_test_np, preds_rf))
        rf_rmse = float(np.sqrt(mean_squared_error(y_test_np, preds_rf)))
        rf_r2_cv_mean, rf_r2_cv_std = cv_r2_scores_transformed(train_rf, X, y, transform)
        rf_models[metric] = rf

        metrics_scores.append(
            {
                "metric": metric,
                "skipped": False,
                "reason": "",
                "y_transform": transform.name,
                "gpr_r2": gpr_r2,
                "gpr_rmse": gpr_rmse,
                "gpr_r2_cv_mean": gpr_r2_cv_mean,
                "gpr_r2_cv_std": gpr_r2_cv_std,
                "rf_r2": rf_r2,
                "rf_rmse": rf_rmse,
                "rf_r2_cv_mean": rf_r2_cv_mean,
                "rf_r2_cv_std": rf_r2_cv_std,
            }
        )

    scores_df = pd.DataFrame(metrics_scores)
    scores_path = outdir / "emulator_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved emulator scores to {scores_path}")

    # History matching if targets provided
    if args.targets:
        targets = load_targets(Path(args.targets))
        hm_df, nroy_df = history_matching(
            X,
            gpr_models,
            targets,
            improb_threshold=args.improb_threshold,
            min_r2=args.min_r2_for_history_matching,
            metric_quality=gpr_quality,
            metric_transforms=metric_transforms,
        )
        hm_path = outdir / "history_matching.csv"
        hm_df.to_csv(hm_path, index=False)
        print(f"Saved history matching table to {hm_path}")

        intervals_df = refined_intervals(nroy_df, PARAM_BOUNDS)
        intervals_path = outdir / "refined_intervals.csv"
        intervals_df.to_csv(intervals_path, index=False)
        print(f"Saved refined intervals to {intervals_path}")
    else:
        print("Targets not provided; skipping history matching.")


if __name__ == "__main__":
    main()
