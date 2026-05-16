"""End-to-end analysis runner.

Walks a folder of CSVs, applies Riedl's four tests to each, then aggregates
across the folder using Wilcoxon signed-rank (BC values > 0) and Fisher's method.

Usage:
  python -m analysis.run_all data/synthetic/plain --label "fake-Plain"
  python -m analysis.run_all data/synthetic/persona --label "fake-Persona"
  python -m analysis.run_all data/synthetic/tom --label "fake-ToM"

  # Or compare multiple folders side-by-side:
  python -m analysis.run_all data/synthetic/plain data/synthetic/persona data/synthetic/tom

Outputs a Markdown-formatted table on stdout, plus optionally writes a CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from analysis.data_io import load_run, quantile_bin
from analysis.tests import (
    practical_criterion,
    emergence_capacity,
    coalition_test,
    differentiation_test,
)
from analysis.nulls import null_distribution, bias_corrected
from analysis.significance import wilcoxon_above_zero, fisher_combine


# ---------- Per-experiment analysis ----------

def analyze_one(
    csv_path: Path,
    K_bins: int = 2,
    lag: int = 1,
    n_surrogates: int = 100,
    block_len: int = 2,
    seed: int = 0,
) -> dict:
    """Run all four tests on one CSV. Returns dict of observed values, BC values, and p-values."""
    run = load_run(csv_path)
    raw = run["raw"]
    devs = run["devs"]
    V = run["macro"]

    # Quantile-bin devs and V.
    X_codes = np.zeros_like(devs, dtype=int)
    for j in range(devs.shape[1]):
        X_codes[:, j] = quantile_bin(devs[:, j], K=K_bins)
    V_codes = quantile_bin(V, K=K_bins)

    # ---- Compute observed test statistics ----
    obs_practical = practical_criterion(X_codes, V_codes, lag=lag)
    obs_emergence = emergence_capacity(X_codes, lag=lag)
    obs_coal = coalition_test(X_codes, V_codes, lag=lag)
    obs_I3, obs_G3 = obs_coal["I3"], obs_coal["G3"]

    # ---- Bias-correct against block-shuffle null (all 4 tests in one pass) ----
    # Computing 4 statistics on the same shuffled X is ~4× faster than running
    # null_distribution() four separate times (each with its own surrogates).
    from analysis.nulls import block_shuffle as _block_shuffle
    rng_seed = seed + abs(hash(str(csv_path))) % (10**8)
    rng = np.random.default_rng(rng_seed)
    null_practical = np.empty(n_surrogates)
    null_emergence = np.empty(n_surrogates)
    null_I3 = np.empty(n_surrogates)
    null_G3 = np.empty(n_surrogates)
    for k in range(n_surrogates):
        Xs = _block_shuffle(X_codes, rng, block_len=block_len)
        null_practical[k] = practical_criterion(Xs, V_codes, lag=lag)
        null_emergence[k] = emergence_capacity(Xs, lag=lag)
        coal = coalition_test(Xs, V_codes, lag=lag)
        null_I3[k] = coal["I3"]
        null_G3[k] = coal["G3"]

    # ---- Differentiation test (mixed model) ----
    diff = differentiation_test(devs)

    return {
        "path": str(csv_path),
        "n_rounds": run["n_rounds"],
        "n_agents": run["n_agents"],
        "target": run["target"],
        # Observed
        "practical_obs": obs_practical,
        "emergence_obs": obs_emergence,
        "I3_obs": obs_I3,
        "G3_obs": obs_G3,
        # Bias-corrected
        "practical_bc": bias_corrected(obs_practical, null_practical),
        "emergence_bc": bias_corrected(obs_emergence, null_emergence),
        "I3_bc": bias_corrected(obs_I3, null_I3),
        "G3_bc": bias_corrected(obs_G3, null_G3),
        # Differentiation p-values
        "p_m0_vs_m1": diff.get("p_m0_vs_m1", float("nan")),
        "p_m1_vs_m2": diff.get("p_m1_vs_m2", float("nan")),
    }


# ---------- Per-folder aggregation ----------

def analyze_folder(
    folder: Path,
    label: Optional[str] = None,
    n_surrogates: int = 100,
    K_bins: int = 2,
    lag: int = 1,
    block_len: int = 2,
    progress: bool = True,
) -> dict:
    """Run analyze_one on every CSV in the folder. Aggregate across runs."""
    label = label or folder.name
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        return {"label": label, "n": 0, "error": "no CSVs found"}

    per_run = []
    for i, p in enumerate(csvs):
        if progress:
            print(f"  [{label}] {i+1}/{len(csvs)}: {p.name}", file=sys.stderr)
        try:
            per_run.append(analyze_one(p, K_bins=K_bins, lag=lag,
                                       n_surrogates=n_surrogates, block_len=block_len))
        except Exception as e:
            print(f"    ! error: {e}", file=sys.stderr)

    df = pd.DataFrame(per_run)

    # Aggregate: medians + Wilcoxon-above-zero on BC values
    summary = {
        "label": label,
        "n_runs": len(per_run),
        "median_practical_bc": float(np.nanmedian(df["practical_bc"])) if len(df) else float("nan"),
        "median_emergence_bc": float(np.nanmedian(df["emergence_bc"])) if len(df) else float("nan"),
        "median_I3_bc": float(np.nanmedian(df["I3_bc"])) if len(df) else float("nan"),
        "median_G3_bc": float(np.nanmedian(df["G3_bc"])) if len(df) else float("nan"),
        "wilcoxon_practical": wilcoxon_above_zero(df["practical_bc"]) if len(df) else None,
        "wilcoxon_emergence": wilcoxon_above_zero(df["emergence_bc"]) if len(df) else None,
        "wilcoxon_I3": wilcoxon_above_zero(df["I3_bc"]) if len(df) else None,
        "wilcoxon_G3": wilcoxon_above_zero(df["G3_bc"]) if len(df) else None,
        # Differentiation: fraction of runs with p < 0.05
        "frac_m1_significant": (
            float(np.mean(df["p_m0_vs_m1"] < 0.05)) if len(df) else float("nan")
        ),
        "frac_m2_significant": (
            float(np.mean(df["p_m1_vs_m2"] < 0.05)) if len(df) else float("nan")
        ),
    }
    summary["_per_run"] = per_run  # keep full data
    return summary


# ---------- Output formatting ----------

def format_summary_table(summaries: list) -> str:
    """Print a Markdown summary table across conditions."""
    rows = []
    rows.append("| Test | " + " | ".join(s["label"] for s in summaries) + " |")
    rows.append("|---" + "|---" * len(summaries) + "|")

    def fmt_w(s, key):
        w = s.get(f"wilcoxon_{key}")
        if w is None:
            return "—"
        p = w.get("pvalue", float("nan"))
        return f"p={p:.3g}" if not np.isnan(p) else "—"

    rows.append("| Practical criterion (BC median) | " +
                " | ".join(f"{s['median_practical_bc']:.4f}" for s in summaries) + " |")
    rows.append("| Practical criterion (Wilcoxon) | " +
                " | ".join(fmt_w(s, "practical") for s in summaries) + " |")
    rows.append("| Emergence capacity (BC median) | " +
                " | ".join(f"{s['median_emergence_bc']:.4f}" for s in summaries) + " |")
    rows.append("| Emergence capacity (Wilcoxon) | " +
                " | ".join(fmt_w(s, "emergence") for s in summaries) + " |")
    rows.append("| I₃ (BC median) | " +
                " | ".join(f"{s['median_I3_bc']:.4f}" for s in summaries) + " |")
    rows.append("| I₃ (Wilcoxon) | " +
                " | ".join(fmt_w(s, "I3") for s in summaries) + " |")
    rows.append("| G₃ (BC median) | " +
                " | ".join(f"{s['median_G3_bc']:.4f}" for s in summaries) + " |")
    rows.append("| G₃ (Wilcoxon) | " +
                " | ".join(fmt_w(s, "G3") for s in summaries) + " |")
    rows.append("| Frac groups w/ agent-intercept p<.05 | " +
                " | ".join(f"{s['frac_m1_significant']:.2f}" for s in summaries) + " |")
    rows.append("| Frac groups w/ agent-slope p<.05 | " +
                " | ".join(f"{s['frac_m2_significant']:.2f}" for s in summaries) + " |")
    rows.append("| Runs analyzed | " +
                " | ".join(str(s["n_runs"]) for s in summaries) + " |")
    return "\n".join(rows)


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="Run Riedl's four tests over CSV folder(s).")
    p.add_argument("folders", nargs="+", type=Path, help="One or more folders of CSVs.")
    p.add_argument("--n_surrogates", type=int, default=100,
                   help="Permutation null size per test (default 100; Riedl uses 200).")
    p.add_argument("--K_bins", type=int, default=2, help="Quantile bins (default 2).")
    p.add_argument("--lag", type=int, default=1, help="Integration timescale ℓ (default 1).")
    p.add_argument("--block_len", type=int, default=2, help="Block size for shuffle (default 2).")
    p.add_argument("--save_json", type=Path, default=None,
                   help="Optional: save full per-run results as JSON.")
    args = p.parse_args()

    summaries = []
    for f in args.folders:
        if not f.is_dir():
            print(f"!! {f} is not a directory", file=sys.stderr)
            continue
        s = analyze_folder(
            f,
            label=f.name,
            n_surrogates=args.n_surrogates,
            K_bins=args.K_bins,
            lag=args.lag,
            block_len=args.block_len,
        )
        summaries.append(s)

    print()
    print(format_summary_table(summaries))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            # Strip _per_run for compactness; use a separate file if you need the full data.
            slim = [{k: v for k, v in s.items() if k != "_per_run"} for s in summaries]
            json.dump(slim, f, indent=2, default=lambda x: None)
        print(f"\nSaved summary to {args.save_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
