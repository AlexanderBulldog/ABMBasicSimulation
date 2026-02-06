
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
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
    name: str

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
        if self.name == "identity":
            return sigma_t
        if self.name == "log1p":
            return sigma_t * np.exp(mu_t)
        if self.name == "signed_log1p":
            return sigma_t * np.exp(np.abs(mu_t))
        raise ValueError(f"Unknown transform: {self.name}")


@dataclass(frozen=True)
class TargetSpec:
    target: float | None
    sigma_obs: float
    sigma_model: float
    sigma_ev: float | None
    sigma_ev_quantile: float | None
    enabled_for_sa: bool


@dataclass(frozen=True)
class UncertaintySpec:
    var_obs: float
    var_ev: float
    var_md: float
    source_ev: str
    ev_quantile: float
    replicate_groups: int
    notes: str
    enabled_for_sa: bool


def choose_transform(metric: str, y: pd.Series) -> YTransform:
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
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def default_metrics(df: pd.DataFrame) -> List[str]:
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


def cv_r2_scores_transformed(
    train_fn,
    X: pd.DataFrame,
    y: pd.Series,
    transform: YTransform,
    n_splits: int = 5,
) -> Tuple[float, float]:
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


def parse_reductions(raw: str) -> List[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = float(tok)
        if not (0.0 < v < 1.0):
            raise ValueError(f"Reduction must be in (0, 1): {v}")
        vals.append(v)
    if not vals:
        raise ValueError("At least one reduction must be provided")
    return vals

def load_targets(path: Path) -> Dict[str, TargetSpec]:
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    out: Dict[str, TargetSpec] = {}
    for metric, v in cfg.items():
        target = v.get("target")
        if target is None and "low" in v and "high" in v:
            target = 0.5 * (v["low"] + v["high"])
        sigma_obs = float(v.get("sigma_obs", 0.0))
        sigma_model = float(v.get("sigma_model", 0.0))
        sigma_ev = v.get("sigma_ev", None)
        sigma_ev_f = float(sigma_ev) if sigma_ev is not None else None
        sigma_ev_quantile = v.get("sigma_ev_quantile", None)
        sigma_ev_q_f = float(sigma_ev_quantile) if sigma_ev_quantile is not None else None
        if sigma_ev_q_f is not None and not (0.0 <= sigma_ev_q_f <= 1.0):
            raise ValueError(f"Invalid sigma_ev_quantile for {metric}: {sigma_ev_q_f}")
        enabled_for_sa = bool(v.get("enabled_for_sa", True))
        out[metric] = TargetSpec(
            target=target,
            sigma_obs=max(0.0, sigma_obs),
            sigma_model=max(0.0, sigma_model),
            sigma_ev=sigma_ev_f if sigma_ev_f is None else max(0.0, sigma_ev_f),
            sigma_ev_quantile=sigma_ev_q_f,
            enabled_for_sa=enabled_for_sa,
        )
    return out


def estimate_ev_from_seed_replicates(
    df: pd.DataFrame,
    param_cols: List[str],
    metric: str,
    quantile: float,
) -> Tuple[float | None, int, str]:
    if metric not in df.columns:
        return None, 0, "metric_missing"
    grouped = df.groupby(param_cols, dropna=False, sort=False)[metric].agg(["count", "var"])
    valid = grouped[(grouped["count"] >= 2) & np.isfinite(grouped["var"])]["var"]
    if valid.empty:
        return None, 0, "insufficient_replicates"
    q = float(np.quantile(valid.to_numpy(dtype=float), quantile))
    return max(0.0, q), int(valid.shape[0]), ""


def build_uncertainty_specs(
    df: pd.DataFrame,
    targets: Dict[str, TargetSpec],
    param_cols: List[str],
    ev_mode: str,
    ev_default_quantile: float,
) -> Tuple[Dict[str, UncertaintySpec], pd.DataFrame]:
    specs: Dict[str, UncertaintySpec] = {}
    rows: List[Dict[str, object]] = []
    for metric, t in targets.items():
        var_obs = float(t.sigma_obs**2)
        var_md = float(t.sigma_model**2)
        notes = ""
        source_ev = "fixed_zero"
        replicate_groups = 0
        q = float(t.sigma_ev_quantile if t.sigma_ev_quantile is not None else ev_default_quantile)

        var_ev = 0.0
        if t.sigma_ev is not None:
            var_ev = float(t.sigma_ev**2)
            source_ev = "fixed_target"
        elif ev_mode == "seed_replicates":
            ev_est, n_groups, ev_note = estimate_ev_from_seed_replicates(df, param_cols, metric, q)
            replicate_groups = n_groups
            if ev_est is not None:
                var_ev = float(ev_est)
                source_ev = "seed_replicates"
            else:
                var_ev = 0.0
                source_ev = "fallback_zero"
                notes = ev_note or "ev_unavailable"
        else:
            var_ev = 0.0
            source_ev = "fixed_zero"
            notes = "ev_mode_fixed_without_sigma_ev"

        spec = UncertaintySpec(
            var_obs=var_obs,
            var_ev=var_ev,
            var_md=var_md,
            source_ev=source_ev,
            ev_quantile=q,
            replicate_groups=replicate_groups,
            notes=notes,
            enabled_for_sa=t.enabled_for_sa,
        )
        specs[metric] = spec
        rows.append(
            {
                "metric": metric,
                "target": t.target,
                "sigma_obs": t.sigma_obs,
                "sigma_model": t.sigma_model,
                "sigma_ev": t.sigma_ev if t.sigma_ev is not None else np.nan,
                "sigma_ev_quantile": q,
                "var_obs": var_obs,
                "var_md": var_md,
                "var_ev": var_ev,
                "source_ev": source_ev,
                "replicate_groups": replicate_groups,
                "enabled_for_sa": t.enabled_for_sa,
                "notes": notes,
            }
        )
    return specs, pd.DataFrame(rows)


def history_matching(
    X: pd.DataFrame,
    gpr_models: Dict[str, Pipeline],
    targets: Dict[str, TargetSpec],
    uncertainty_specs: Dict[str, UncertaintySpec],
    improb_threshold: float = 3.0,
    min_r2: float = 0.2,
    metric_quality: Dict[str, float] | None = None,
    metric_transforms: Dict[str, YTransform] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    records = []
    used_metrics: set[str] = set()
    for idx in range(len(X)):
        row_params = X.iloc[idx]
        row_res = {**row_params.to_dict()}
        max_I = -np.inf
        used = 0
        for metric, cfg in targets.items():
            model = gpr_models.get(metric)
            if model is None or cfg.target is None:
                continue
            if metric_quality is not None and metric_quality.get(metric, -np.inf) < min_r2:
                continue
            transform = metric_transforms.get(metric, YTransform("identity")) if metric_transforms else YTransform("identity")
            pred_t, std_t = model.predict(row_params.to_frame().T, return_std=True)
            mu_t = float(pred_t[0])
            sigma_emul_t = float(std_t[0])
            mu = float(transform.inverse(np.array([mu_t]))[0])
            sigma_emul = float(transform.inverse_sigma(np.array([mu_t]), np.array([sigma_emul_t]))[0])
            var_cu = float(max(0.0, sigma_emul**2))

            unc = uncertainty_specs.get(metric)
            var_obs = float(unc.var_obs) if unc is not None else 0.0
            var_ev = float(unc.var_ev) if unc is not None else 0.0
            var_md = float(unc.var_md) if unc is not None else 0.0
            var_total = max(var_obs + var_ev + var_md + var_cu, 1e-12)
            I = abs(mu - cfg.target) / np.sqrt(var_total)

            row_res[f"{metric}_pred"] = mu
            row_res[f"{metric}_sigma_emul"] = sigma_emul
            row_res[f"{metric}_var_obs"] = var_obs
            row_res[f"{metric}_var_ev"] = var_ev
            row_res[f"{metric}_var_md"] = var_md
            row_res[f"{metric}_var_cu"] = var_cu
            row_res[f"{metric}_var_total"] = var_total
            row_res[f"{metric}_I"] = I
            if I > max_I:
                max_I = I
            used += 1
            used_metrics.add(metric)
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
    return df, nroy_df, sorted(used_metrics)


def refined_intervals(nroy_df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
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

def recompute_i_max_with_component_reduction(
    df: pd.DataFrame,
    metrics: List[str],
    targets: Dict[str, TargetSpec],
    component: str,
    reduction: float,
) -> pd.Series:
    vals = []
    for _, row in df.iterrows():
        max_i = -np.inf
        used = 0
        for metric in metrics:
            target = targets.get(metric)
            if target is None or target.target is None:
                continue
            pred_key = f"{metric}_pred"
            if pred_key not in row:
                continue
            var_obs = float(row.get(f"{metric}_var_obs", 0.0))
            var_ev = float(row.get(f"{metric}_var_ev", 0.0))
            var_md = float(row.get(f"{metric}_var_md", 0.0))
            var_cu = float(row.get(f"{metric}_var_cu", 0.0))
            if component == "OU":
                var_obs *= 1.0 - reduction
            elif component == "EV":
                var_ev *= 1.0 - reduction
            elif component == "MD":
                var_md *= 1.0 - reduction
            elif component == "CU":
                var_cu *= 1.0 - reduction
            var_total = max(var_obs + var_ev + var_md + var_cu, 1e-12)
            i_val = abs(float(row[pred_key]) - float(target.target)) / np.sqrt(var_total)
            max_i = max(max_i, i_val)
            used += 1
        vals.append(np.nan if used == 0 else max_i)
    return pd.Series(vals, index=df.index)


def run_sensitivity_analysis(
    hm_df: pd.DataFrame,
    targets: Dict[str, TargetSpec],
    uncertainty_specs: Dict[str, UncertaintySpec],
    used_metrics: List[str],
    reductions: List[float],
    improb_threshold: float,
    domain: str,
) -> pd.DataFrame:
    domain_df = hm_df[hm_df["nroy"] == True].copy() if domain == "nroy" else hm_df.copy()  # noqa: E712
    if domain_df.empty:
        rows = []
        for comp in ("EV", "OU", "MD", "CU"):
            for red in reductions:
                rows.append(
                    {
                        "component": comp,
                        "reduction": red,
                        "n_points": 0,
                        "newly_implausible_count": 0,
                        "newly_implausible_share": np.nan,
                    }
                )
        return pd.DataFrame(rows)

    default_unc = UncertaintySpec(0.0, 0.0, 0.0, "", 0.9, 0, "", True)
    sa_metrics = [m for m in used_metrics if uncertainty_specs.get(m, default_unc).enabled_for_sa]
    rows = []
    for comp in ("EV", "OU", "MD", "CU"):
        for red in reductions:
            new_i_max = recompute_i_max_with_component_reduction(
                domain_df,
                sa_metrics,
                targets,
                component=comp,
                reduction=red,
            )
            newly_impl = (new_i_max >= improb_threshold).fillna(False)
            n_points = int(domain_df.shape[0])
            n_impl = int(newly_impl.sum())
            share = float(n_impl / max(n_points, 1))
            rows.append(
                {
                    "component": comp,
                    "reduction": float(red),
                    "n_points": n_points,
                    "newly_implausible_count": n_impl,
                    "newly_implausible_share": share,
                }
            )
    sa_df = pd.DataFrame(rows)
    if sa_df.empty:
        return sa_df

    max_red = max(reductions)
    rank_tbl = []
    for comp in sorted(sa_df["component"].unique()):
        comp_df = sa_df[sa_df["component"] == comp]
        share_rmax = float(comp_df.loc[comp_df["reduction"] == max_red, "newly_implausible_share"].mean())
        share_mean = float(comp_df["newly_implausible_share"].mean())
        rank_tbl.append({"component": comp, "share_at_max_reduction": share_rmax, "share_mean": share_mean})
    rank_df = pd.DataFrame(rank_tbl)
    rank_df["rank_at_max_reduction"] = rank_df["share_at_max_reduction"].rank(ascending=False, method="dense").astype(int)
    rank_df["rank_by_mean"] = rank_df["share_mean"].rank(ascending=False, method="dense").astype(int)
    sa_df = sa_df.merge(rank_df, on="component", how="left")
    return sa_df.sort_values(["component", "reduction"]).reset_index(drop=True)


def generate_sensitivity_report(
    out_path: Path,
    data_path: Path,
    hm_df: pd.DataFrame,
    unc_df: pd.DataFrame,
    sa_df: pd.DataFrame,
    targets: Dict[str, TargetSpec],
    used_metrics: List[str],
    reductions: List[float],
    improb_threshold: float,
    domain: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_df = hm_df[hm_df["nroy"] == True].copy() if domain == "nroy" else hm_df.copy()  # noqa: E712
    baseline_n = int(baseline_df.shape[0])
    dominant_component = "n/a"
    if not sa_df.empty and "rank_at_max_reduction" in sa_df.columns:
        top = sa_df[sa_df["rank_at_max_reduction"] == 1]["component"].dropna().unique().tolist()
        if top:
            dominant_component = ", ".join(sorted([str(x) for x in top]))

    lines: List[str] = []
    lines.append(f"# Uncertainty Sensitivity Report ({date.today().isoformat()})\n\n")
    lines.append("## 1) Setup\n")
    lines.append(f"- Dataset: `{data_path}`\n")
    lines.append(f"- Implausibility threshold: `{improb_threshold}`\n")
    lines.append(f"- SA domain: `{domain}`\n")
    lines.append(f"- Baseline points used: `{baseline_n}`\n")
    lines.append(f"- Reductions: `{', '.join([f'{int(100*r)}%' for r in reductions])}`\n\n")

    lines.append("## 2) Baseline Uncertainty Decomposition\n")
    lines.append("| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for metric in used_metrics:
        if metric not in targets:
            continue
        has_row = (unc_df["metric"] == metric).any()
        var_obs = float(unc_df.loc[unc_df["metric"] == metric, "var_obs"].iloc[0]) if has_row else 0.0
        var_ev = float(unc_df.loc[unc_df["metric"] == metric, "var_ev"].iloc[0]) if has_row else 0.0
        var_md = float(unc_df.loc[unc_df["metric"] == metric, "var_md"].iloc[0]) if has_row else 0.0
        cu_col = f"{metric}_var_cu"
        total_col = f"{metric}_var_total"
        var_cu_med = float(baseline_df[cu_col].median()) if cu_col in baseline_df.columns and not baseline_df.empty else np.nan
        var_total_med = float(baseline_df[total_col].median()) if total_col in baseline_df.columns and not baseline_df.empty else np.nan
        lines.append(f"| {metric} | {var_obs:.6g} | {var_ev:.6g} | {var_md:.6g} | {var_cu_med:.6g} | {var_total_med:.6g} |\n")
    lines.append("\n")

    lines.append("## 3) SA Results (newly implausible share)\n")
    lines.append("| Component | Reduction | Points | Newly implausible | Share |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for _, row in sa_df.iterrows():
        lines.append(
            f"| {row['component']} | {int(round(float(row['reduction']) * 100))}% | {int(row['n_points'])} | "
            f"{int(row['newly_implausible_count'])} | {float(row['newly_implausible_share']):.4f} |\n"
        )
    lines.append("\n")

    lines.append("## 4) Interpretation\n")
    lines.append(f"- Dominant component (rank at max reduction): `{dominant_component}`\n")
    lines.append("- Investment guidance:\n")
    lines.append("  - `CU` dominant: increase design density / improve GP specification.\n")
    lines.append("  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.\n")
    lines.append("  - `OU` dominant: improve measurement quality / uncertainty model of observations.\n")
    lines.append("  - `MD` dominant: revisit model structure or discrepancy assumptions.\n\n")

    lines.append("## 5) Limitations\n")
    lines.append("- EV is treated as metric-level scalar (not x-dependent).\n")
    lines.append("- SA is local to selected baseline domain.\n")
    lines.append("- MD remains expert-specified and is not inferred from data.\n")

    out_path.write_text("".join(lines), encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Train emulator (GPR + RF baseline) and optional history matching.")
    parser.add_argument("--data", type=str, default="output/datasets/lhs_runs.csv", help="Path to dataset from run_lhs.py")
    parser.add_argument("--metrics", type=str, default="", help="Comma-separated metric columns")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split size")
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help="JSON file with target, sigma_obs, sigma_model (plus optional sigma_ev/sigma_ev_quantile/enabled_for_sa)",
    )
    parser.add_argument("--improb-threshold", type=float, default=3.0, help="Threshold for NROY")
    parser.add_argument("--min-r2-for-history-matching", type=float, default=0.2)
    parser.add_argument(
        "--ev-mode",
        type=str,
        choices=["seed_replicates", "fixed"],
        default="seed_replicates",
        help="How to set EV uncertainty",
    )
    parser.add_argument("--ev-default-quantile", type=float, default=0.9)
    parser.add_argument(
        "--save-uncertainty-breakdown",
        dest="save_uncertainty_breakdown",
        action="store_true",
        help="Save uncertainty_components.csv",
    )
    parser.add_argument(
        "--no-save-uncertainty-breakdown",
        dest="save_uncertainty_breakdown",
        action="store_false",
        help="Disable saving uncertainty_components.csv",
    )
    parser.set_defaults(save_uncertainty_breakdown=True)
    parser.add_argument("--sa-enable", action="store_true", help="Enable EV/OU/MD/CU sensitivity analysis")
    parser.add_argument("--sa-reductions", type=str, default="0.1,0.2,0.3,0.4")
    parser.add_argument("--sa-domain", type=str, choices=["nroy", "all"], default="nroy")
    parser.add_argument("--sa-out", type=str, default="")
    parser.add_argument("--sa-report", type=str, default="")
    parser.add_argument("--outdir", type=str, default="output/results")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (0.0 <= args.ev_default_quantile <= 1.0):
        raise SystemExit("--ev-default-quantile must be in [0, 1]")
    reductions = parse_reductions(args.sa_reductions)

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    if "bad_run" in df:
        df = df[df["bad_run"] == False]  # noqa: E712
    if "error" in df:
        df = df[df["error"].isna() | (df["error"] == "")]

    param_cols = list(PARAM_BOUNDS.keys())
    metric_cols = [m for m in args.metrics.split(",") if m] if args.metrics else default_metrics(df)
    X = df[param_cols]

    metrics_scores = []
    gpr_models: Dict[str, Pipeline] = {}
    gpr_quality: Dict[str, float] = {}
    metric_transforms: Dict[str, YTransform] = {}

    X_train, X_test = train_test_split(X, test_size=args.test_size, random_state=42)
    idx_train = X_train.index
    idx_test = X_test.index

    for metric in metric_cols:
        y = df[metric]
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
        gpr_quality[metric] = float(gpr_r2_cv_mean)

        rf = train_rf(X_train, y_train_t)
        preds_rf_t = rf.predict(X_test)
        preds_rf = transform.inverse(np.asarray(preds_rf_t, dtype=float))
        rf_r2 = float(r2_score(y_test_np, preds_rf))
        rf_rmse = float(np.sqrt(mean_squared_error(y_test_np, preds_rf)))
        rf_r2_cv_mean, rf_r2_cv_std = cv_r2_scores_transformed(train_rf, X, y, transform)

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

    if args.targets:
        targets = load_targets(Path(args.targets))
        uncertainty_specs, uncertainty_df = build_uncertainty_specs(
            df=df,
            targets=targets,
            param_cols=param_cols,
            ev_mode=args.ev_mode,
            ev_default_quantile=float(args.ev_default_quantile),
        )
        if args.save_uncertainty_breakdown:
            unc_path = outdir / "uncertainty_components.csv"
            uncertainty_df.to_csv(unc_path, index=False)
            print(f"Saved uncertainty breakdown to {unc_path}")
            warn_rows = uncertainty_df[uncertainty_df["notes"] != ""]
            if not warn_rows.empty:
                for _, wr in warn_rows.iterrows():
                    print(f"[WARN] {wr['metric']}: {wr['notes']}")

        hm_df, nroy_df, used_metrics = history_matching(
            X,
            gpr_models,
            targets,
            uncertainty_specs,
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

        if args.sa_enable:
            sa_df = run_sensitivity_analysis(
                hm_df=hm_df,
                targets=targets,
                uncertainty_specs=uncertainty_specs,
                used_metrics=used_metrics,
                reductions=reductions,
                improb_threshold=args.improb_threshold,
                domain=args.sa_domain,
            )
            sa_out_path = Path(args.sa_out) if args.sa_out else outdir / "sensitivity_uncertainty.csv"
            sa_out_path.parent.mkdir(parents=True, exist_ok=True)
            sa_df.to_csv(sa_out_path, index=False)
            print(f"Saved sensitivity table to {sa_out_path}")

            sa_report_path = Path(args.sa_report) if args.sa_report else outdir / "sensitivity_report.md"
            generate_sensitivity_report(
                out_path=sa_report_path,
                data_path=data_path,
                hm_df=hm_df,
                unc_df=uncertainty_df,
                sa_df=sa_df,
                targets=targets,
                used_metrics=used_metrics,
                reductions=reductions,
                improb_threshold=args.improb_threshold,
                domain=args.sa_domain,
            )
            print(f"Saved sensitivity report to {sa_report_path}")
    else:
        print("Targets not provided; skipping history matching.")


if __name__ == "__main__":
    main()
