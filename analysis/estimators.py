"""Discrete entropy / mutual-information estimators with bias correction.

Used by tests.py for the practical criterion (Eq. 2) and coalition test (Eq. 3).

We support three estimators, matching Riedl §2 Entropy Estimation:
  - plug-in           : empirical frequencies (biased high for small N)
  - jeffreys          : add 1/2 to every count cell (Riedl's main choice)
  - miller_madow      : plug-in + (K-1)/(2N) correction (robustness check)

All functions work on integer-coded discrete data of any dimensionality.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _joint_counts(*arrays: np.ndarray) -> np.ndarray:
    """Stack 1D arrays of integer codes into joint count tensor.

    arrays: each is shape (N,) of non-negative integers.
    Returns an array of shape (K1, K2, ...) of integer counts.
    """
    if not arrays:
        raise ValueError("Need at least one array")
    arrays = [np.asarray(a, dtype=int) for a in arrays]
    n = len(arrays[0])
    shape = tuple(int(a.max()) + 1 for a in arrays)
    counts = np.zeros(shape, dtype=np.int64)
    idx = tuple(arrays)
    np.add.at(counts, idx, 1)
    assert counts.sum() == n
    return counts


def _smooth(counts: np.ndarray, method: str) -> np.ndarray:
    """Convert integer counts into probabilities with optional smoothing.

    method: 'plugin' | 'jeffreys'
    """
    counts = counts.astype(float)
    if method == "jeffreys":
        counts = counts + 0.5
    return counts / counts.sum()


def entropy(probs: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy of a discrete distribution (any shape). Zeros are handled."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p) / np.log(base)))


def mi(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "jeffreys",
    base: float = 2.0,
) -> float:
    """I(X; Y) for integer-coded X, Y.

    method:
      'jeffreys'    - Jeffreys-prior plug-in (Riedl's main).
      'plugin'      - raw plug-in.
      'miller_madow'- plug-in + Miller-Madow correction.
    """
    x = np.asarray(x, dtype=int).ravel()
    y = np.asarray(y, dtype=int).ravel()
    assert x.size == y.size, "X and Y must have same length"

    joint_c = _joint_counts(x, y)
    if method == "miller_madow":
        p_xy = _smooth(joint_c, "plugin")
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        h_xy = entropy(p_xy, base) + (np.count_nonzero(p_xy) - 1) / (2 * x.size * np.log(base))
        h_x = entropy(p_x, base) + (np.count_nonzero(p_x) - 1) / (2 * x.size * np.log(base))
        h_y = entropy(p_y, base) + (np.count_nonzero(p_y) - 1) / (2 * x.size * np.log(base))
        return h_x + h_y - h_xy

    p_xy = _smooth(joint_c, "jeffreys" if method == "jeffreys" else "plugin")
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    return entropy(p_x, base) + entropy(p_y, base) - entropy(p_xy, base)


def mi_joint(
    sources: Iterable[np.ndarray],
    target: np.ndarray,
    method: str = "jeffreys",
    base: float = 2.0,
) -> float:
    """I({X1, X2, ...}; Y) where the sources are treated as a joint variable.

    Encodes the joint of the sources as a single integer code, then computes
    standard MI to the target.
    """
    src_list = [np.asarray(s, dtype=int).ravel() for s in sources]
    n = src_list[0].size
    # Encode joint sources via flat index.
    src_arr = np.stack(src_list, axis=1)  # (n, k)
    multipliers = np.array([1] + [int(s.max()) + 1 for s in src_list[:-1]])
    multipliers = np.cumprod(multipliers)
    joint_code = (src_arr * multipliers[None, :]).sum(axis=1)
    return mi(joint_code, np.asarray(target, dtype=int).ravel(), method=method, base=base)
