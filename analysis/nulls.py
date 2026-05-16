"""Permutation null distributions for Riedl's tests.

Two surrogates per Riedl §2 Falsification Tests:
  - block_shuffle (column-wise time shuffle): preserves per-agent dynamics,
    breaks cross-agent temporal alignment. Use to test identity-linked
    coordination vs. spurious dynamic alignment.
  - row_shuffle: completely permutes guesses within each round, breaking
    agent identities. Use to test identity-locked specialization.

Each function returns a generator yielding shuffled (rounds × agents) arrays.
"""

from __future__ import annotations

import numpy as np


def block_shuffle(X: np.ndarray, rng: np.random.Generator, block_len: int = 2) -> np.ndarray:
    """For each agent (column), apply a circular shift by a random integer multiple
    of block_len. Preserves within-block dynamics, breaks cross-agent time alignment.
    """
    n_rounds, n_agents = X.shape
    out = np.empty_like(X)
    n_blocks = max(1, n_rounds // block_len)
    for j in range(n_agents):
        shift_blocks = int(rng.integers(0, n_blocks))
        shift = (shift_blocks * block_len) % n_rounds
        out[:, j] = np.roll(X[:, j], shift)
    return out


def row_shuffle(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Within each round, permute the agent ordering. Breaks agent identities;
    preserves the within-round distribution of guesses.
    """
    n_rounds, n_agents = X.shape
    out = np.empty_like(X)
    for t in range(n_rounds):
        perm = rng.permutation(n_agents)
        out[t, :] = X[t, perm]
    return out


def null_distribution(
    X: np.ndarray,
    statistic_fn,
    surrogate: str = "block",
    n_surrogates: int = 200,
    block_len: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Compute statistic_fn on n_surrogates shuffled copies of X.

    statistic_fn: takes (X_shuffled,) -> float
    surrogate: 'block' or 'row'
    Returns numpy array of n_surrogates values.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_surrogates)
    for k in range(n_surrogates):
        if surrogate == "block":
            Xs = block_shuffle(X, rng, block_len=block_len)
        elif surrogate == "row":
            Xs = row_shuffle(X, rng)
        else:
            raise ValueError(f"Unknown surrogate {surrogate!r}")
        out[k] = statistic_fn(Xs)
    return out


def bias_corrected(observed: float, null: np.ndarray) -> float:
    """Observed value minus median of null distribution (Riedl §2)."""
    return observed - float(np.median(null))


def p_value_above(observed: float, null: np.ndarray) -> float:
    """Right-tailed empirical p-value: fraction of null ≥ observed.
    Uses (k+1)/(n+1) for finite-sample correction.
    """
    n = null.size
    k = int(np.sum(null >= observed))
    return (k + 1) / (n + 1)
