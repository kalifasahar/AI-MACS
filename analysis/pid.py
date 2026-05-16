"""Williams-Beer two-source PID — fast numpy implementation.

Implements I_min redundancy from Williams & Beer (2010), the same redundancy
measure Riedl uses (paper §2 Entropy Estimation). Direct implementation
avoids `dit`'s symbolic overhead (~100x speedup for small alphabets).

Definitions (Williams-Beer Imin for 2-source PID):
  Specific info:   I(X_i; T=t) = Σ_x p(x|t) * log(p(x|t) / p(x))
  Redundancy:      I_∩({X1},{X2}; T) = Σ_t p(t) * min_i I(X_i; T=t)
  Unique_i:        I(X_i; T) - redundancy
  Synergy:         I(X1,X2; T) - I(X1; T) - I(X2; T) + redundancy
  Check:           red + unique_i + unique_j + syn = I(X1,X2; T)

All quantities in bits (log base 2).
"""

from __future__ import annotations

import numpy as np


def _joint_3way(x_i: np.ndarray, x_j: np.ndarray, t: np.ndarray,
                jeffreys: bool = True) -> tuple:
    """Return (p_xij_t, p_xi, p_xj, p_t, p_xij) joint/marginal probabilities."""
    x_i = np.asarray(x_i, dtype=int).ravel()
    x_j = np.asarray(x_j, dtype=int).ravel()
    t = np.asarray(t, dtype=int).ravel()
    assert x_i.size == x_j.size == t.size

    K_i = int(x_i.max()) + 1
    K_j = int(x_j.max()) + 1
    K_t = int(t.max()) + 1

    # Build joint count tensor (K_i × K_j × K_t)
    counts = np.zeros((K_i, K_j, K_t), dtype=float)
    np.add.at(counts, (x_i, x_j, t), 1.0)
    if jeffreys:
        counts = counts + 0.5
    p_xij_t = counts / counts.sum()

    p_xi = p_xij_t.sum(axis=(1, 2))    # P(X_i)
    p_xj = p_xij_t.sum(axis=(0, 2))    # P(X_j)
    p_t = p_xij_t.sum(axis=(0, 1))     # P(T)
    p_xij = p_xij_t.sum(axis=2)        # P(X_i, X_j) -- for I(X1,X2;T) joint MI

    return p_xij_t, p_xi, p_xj, p_t, p_xij


def _mi(p_joint: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> float:
    """Mutual information I(A;B) given the joint and marginals. All in bits."""
    # I(A;B) = Σ p(a,b) log[p(a,b) / (p(a)p(b))]
    p_outer = np.outer(p_a, p_b)
    mask = p_joint > 0
    if not mask.any():
        return 0.0
    ratio = p_joint[mask] / p_outer[mask]
    return float(np.sum(p_joint[mask] * np.log2(ratio)))


def _specific_info(p_xt: np.ndarray, p_x: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """I(X; T=t) for each t. Returns array of length K_t.

    I(X; T=t) = Σ_x p(x|t) * log2(p(x|t) / p(x))
    Where p(x|t) = p(x,t) / p(t)
    """
    K_t = p_t.size
    out = np.zeros(K_t)
    for t in range(K_t):
        if p_t[t] <= 0:
            continue
        p_x_given_t = p_xt[:, t] / p_t[t]
        mask = (p_x_given_t > 0) & (p_x > 0)
        if not mask.any():
            continue
        out[t] = float(np.sum(p_x_given_t[mask] * np.log2(p_x_given_t[mask] / p_x[mask])))
    return out


def pid_two_source(
    x_i: np.ndarray,
    x_j: np.ndarray,
    target_codes: np.ndarray,
    jeffreys: bool = True,
) -> dict:
    """Williams-Beer Imin PID of I({X_i, X_j}; T).

    Returns dict with keys:
      'red'      - redundant info from both sources (I_min)
      'unique_i' - unique info from source i only
      'unique_j' - unique info from source j only
      'syn'      - synergistic info requiring both sources
      'mi_total' - I({X_i, X_j}; T) = sum of the four atoms
    """
    p_xij_t, p_xi, p_xj, p_t, p_xij = _joint_3way(x_i, x_j, target_codes, jeffreys)

    # Marginal joint of (X_i, T) and (X_j, T)
    p_xi_t = p_xij_t.sum(axis=1)  # (K_i, K_t)
    p_xj_t = p_xij_t.sum(axis=0)  # (K_j, K_t)

    # Specific information of each source about each target value
    spec_i = _specific_info(p_xi_t, p_xi, p_t)
    spec_j = _specific_info(p_xj_t, p_xj, p_t)

    # I_min redundancy: Σ_t p(t) * min(spec_i(t), spec_j(t))
    red = float(np.sum(p_t * np.minimum(spec_i, spec_j)))

    # Individual mutual informations
    mi_i = _mi(p_xi_t, p_xi, p_t)
    mi_j = _mi(p_xj_t, p_xj, p_t)

    # Joint mutual information I({X_i, X_j}; T)
    # Treat (X_i, X_j) as a single variable -> reshape joint
    K_i, K_j, K_t = p_xij_t.shape
    p_xij_flat = p_xij.reshape(-1)          # length K_i*K_j
    p_xij_t_flat = p_xij_t.reshape(-1, K_t) # (K_i*K_j, K_t)
    mi_total = _mi(p_xij_t_flat, p_xij_flat, p_t)

    unique_i = mi_i - red
    unique_j = mi_j - red
    syn = mi_total - mi_i - mi_j + red

    return {
        "red": red,
        "unique_i": unique_i,
        "unique_j": unique_j,
        "syn": syn,
        "mi_total": mi_total,
    }
