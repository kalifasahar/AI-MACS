"""Load Riedl-format CSVs into numpy arrays and apply Riedl's data transforms.

A Riedl-format CSV has columns: round, agent_1, agent_2, ..., agent_N

Functions:
  load_csv(path)              -> numpy array (rounds × agents) of raw guesses
  devs(raw, target)           -> deviation-from-equal-share microstate
  macro(devs)                 -> macro signal V_t per round
  quantile_bin(x, K=2)        -> discretize into K equal-frequency bins
  load_run(path, target=None) -> dict with raw, devs, macro, target

We accept either:
  - target known (e.g., synthetic data we generated): pass target as int
  - target unknown (e.g., real Riedl CSVs): derive from the filename
    pattern game_data_target_<N>.csv, or fall back to using sum-mean as proxy.

Riedl's transforms (paper §2 Estimation Details):
  devs_{i,t} = raw_{i,t} - target/N
  V_t = sum_i devs_{i,t} = group_sum_t - target
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_csv(path: Path | str) -> np.ndarray:
    """Load a Riedl-format CSV into an (rounds × agents) array of raw guesses."""
    df = pd.read_csv(path)
    if "round" in df.columns:
        df = df.drop(columns=["round"])
    return df.to_numpy(dtype=int)


def infer_target(path: Path | str) -> Optional[int]:
    """Extract target from filename pattern like 'game_data_target_85.csv'.
    Returns None if no target is encoded in the filename.
    """
    name = Path(path).name
    m = re.search(r"target[_\s]*(\d+)", name)
    return int(m.group(1)) if m else None


def devs(raw: np.ndarray, target: int) -> np.ndarray:
    """devs_{i,t} = raw_{i,t} - target/N  (Riedl §2)"""
    n_agents = raw.shape[1]
    return raw - (target / n_agents)


def macro(devs_arr: np.ndarray) -> np.ndarray:
    """V_t = sum_i devs_{i,t}  (group sum minus target)"""
    return devs_arr.sum(axis=1)


def quantile_bin(x: np.ndarray, K: int = 2) -> np.ndarray:
    """Discretize x into K equal-frequency bins, returning integer codes 0..K-1.

    Uses `pd.qcut` with duplicate-edge handling to avoid errors on constant data.
    Falls back to zero-array if everything is constant.
    """
    flat = np.asarray(x).ravel()
    if np.allclose(flat, flat[0]):
        return np.zeros_like(flat, dtype=int).reshape(x.shape)
    codes = pd.qcut(flat, q=K, labels=False, duplicates="drop")
    # qcut may return fewer bins if duplicates collapse; that's OK.
    return np.asarray(codes, dtype=int).reshape(x.shape)


def load_run(path: Path | str, target: Optional[int] = None) -> dict:
    """High-level loader: returns {raw, target, devs, macro, n_agents, n_rounds}."""
    raw = load_csv(path)
    if target is None:
        target = infer_target(path)
    if target is None:
        # Last-resort: estimate target as the modal group sum across rounds
        # (Riedl's experiments end when the sum equals target). We use median
        # of the final 3 rounds' group sums.
        last_sums = raw.sum(axis=1)[-3:]
        target = int(np.median(last_sums))
    d = devs(raw, target)
    return {
        "path": str(path),
        "raw": raw,
        "target": target,
        "devs": d,
        "macro": macro(d),
        "n_agents": raw.shape[1],
        "n_rounds": raw.shape[0],
    }
