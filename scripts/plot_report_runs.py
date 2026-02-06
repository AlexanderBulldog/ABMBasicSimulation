from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _parse_rep_seed(path: Path) -> Tuple[int, int]:
    m = re.search(r"rep_run_(\d+)_seed(\d+)", path.stem)
    if not m:
        raise ValueError(f"Unrecognized filename (expected rep_run_XX_seedY.csv): {path.name}")
    return int(m.group(1)), int(m.group(2))


def _load_run_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Some outputs may include an unnamed index column.
    if df.columns.size and str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def _x_values(df: pd.DataFrame):
    for c in ("step", "Step", "t", "time"):
        if c in df.columns:
            return df[c].to_numpy()
    return list(range(len(df)))


def _plot_rep(rep_id: int, runs: List[Tuple[int, pd.DataFrame]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [
        ("Employment", "Employment"),
        ("Output", "Output"),
        ("Consumption", "Consumption"),
        ("AvgPrice", "AvgPrice"),
        ("HH_Deposit", "HH_Deposit"),
        ("HH_Debt", "HH_Debt"),
        ("Bank_Equity", "Bank_Equity"),
        ("BankResolutionHaircut", "BankResolutionHaircut"),
    ]

    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), dpi=160)
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, series):
        plotted = False
        for seed, df in runs:
            if col not in df.columns:
                continue
            x = _x_values(df)
            ax.plot(x, df[col], linewidth=1.2, alpha=0.9, label=f"seed {seed}")
            plotted = True

        ax.set_title(title)
        ax.grid(True, linewidth=0.4, alpha=0.35)
        if plotted:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"{col} not found", ha="center", va="center", transform=ax.transAxes)

    # Hide any leftover axes (in case series list changes)
    for ax in axes[len(series) :]:
        ax.axis("off")

    fig.suptitle(f"Representative run rep_id={rep_id} (all seeds)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot rep_run_XX_seedY.csv panels for all representative points.")
    p.add_argument("--runs-dir", type=str, required=True, help="Directory with rep_run_XX_seedY.csv")
    p.add_argument("--out", type=str, required=True, help="Output directory for PNGs")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(runs_dir.glob("rep_run_*_seed*.csv"))
    if not files:
        raise SystemExit(f"No rep_run CSVs found in {runs_dir}")

    grouped: Dict[int, List[Tuple[int, pd.DataFrame]]] = {}
    for path in files:
        rep_id, seed = _parse_rep_seed(path)
        grouped.setdefault(rep_id, []).append((seed, _load_run_csv(path)))

    for rep_id in sorted(grouped.keys()):
        runs = sorted(grouped[rep_id], key=lambda x: x[0])
        out_path = out_dir / f"rep_{rep_id:02d}.png"
        _plot_rep(rep_id, runs, out_path)
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
