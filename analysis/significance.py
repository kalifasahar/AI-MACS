"""Significance combination and one-sided Wilcoxon tests (Riedl §2)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import wilcoxon, combine_pvalues


def wilcoxon_above_zero(values: Sequence[float]) -> dict:
    """One-sided Wilcoxon signed-rank: H1 = median > 0.

    Drops NaNs. Returns dict with statistic, p-value, n.
    """
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    n = arr.size
    if n < 3 or np.all(arr == 0):
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": n}
    try:
        # alternative='greater' tests H1: median > 0
        res = wilcoxon(arr, alternative="greater", zero_method="wilcox")
        return {"statistic": float(res.statistic), "pvalue": float(res.pvalue), "n": n}
    except ValueError:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": n}


def fisher_combine(pvalues: Sequence[float]) -> dict:
    """Fisher's method to combine independent p-values."""
    arr = np.asarray([p for p in pvalues if not np.isnan(p)], dtype=float)
    arr = np.clip(arr, 1e-300, 1.0)  # avoid log(0)
    if arr.size < 2:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": int(arr.size)}
    stat, pval = combine_pvalues(arr, method="fisher")
    return {"statistic": float(stat), "pvalue": float(pval), "n": int(arr.size)}
