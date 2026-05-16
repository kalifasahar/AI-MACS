"""The four tests from Riedl (2026) §2.

All tests operate on a single experiment's data (rounds × agents matrix of
quantile-binned codes), plus optional macro signal codes.

Functions:
  practical_criterion(X_codes, V_codes, lag=1)  -> float (Eq. 2)
  emergence_capacity(X_codes, lag=1)            -> float (Eq. 1, median pairwise syn)
  coalition_test(X_codes, V_codes, lag=1)       -> dict with I3, G3
  differentiation_test(raw)                     -> dict with LR p-values for m0->m1, m1->m2

All info-theoretic outputs are in bits.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

from analysis.estimators import mi, mi_joint
from analysis.pid import pid_two_source


# ---------- Test 2: Practical criterion (Eq. 2) ----------

def practical_criterion(
    X_codes: np.ndarray,
    V_codes: np.ndarray,
    lag: int = 1,
    method: str = "jeffreys",
) -> float:
    """S_macro(ℓ) = I(V_t; V_{t+ℓ}) - Σ_k I(X_k,t; V_{t+ℓ})

    X_codes: (rounds × agents) integer-coded matrix
    V_codes: (rounds,) integer-coded macro signal
    """
    n_rounds, n_agents = X_codes.shape
    if n_rounds <= lag:
        return float("nan")
    V_now = V_codes[:-lag]
    V_future = V_codes[lag:]
    macro_self = mi(V_now, V_future, method=method)

    parts_sum = 0.0
    for k in range(n_agents):
        Xk_now = X_codes[:-lag, k]
        parts_sum += mi(Xk_now, V_future, method=method)
    return macro_self - parts_sum


# ---------- Test 1: Emergence capacity (Eq. 1, pairwise PID synergy) ----------

def emergence_capacity(
    X_codes: np.ndarray,
    lag: int = 1,
    method: str = "jeffreys",
) -> float:
    """Median pairwise PID synergy of I({X_i,t, X_j,t}; (X_i,t+ℓ, X_j,t+ℓ)).

    Target is the JOINT future of the two agents — encoded as a single integer.
    """
    n_rounds, n_agents = X_codes.shape
    if n_rounds <= lag:
        return float("nan")

    syns = []
    for i, j in combinations(range(n_agents), 2):
        Xi_now = X_codes[:-lag, i]
        Xj_now = X_codes[:-lag, j]
        Xi_fut = X_codes[lag:, i]
        Xj_fut = X_codes[lag:, j]
        # Encode joint (Xi_fut, Xj_fut) → single integer code
        K_j = int(Xj_fut.max()) + 1
        joint_future = Xi_fut * K_j + Xj_fut
        pid = pid_two_source(Xi_now, Xj_now, joint_future,
                             jeffreys=(method == "jeffreys"))
        syns.append(pid["syn"])
    return float(np.median(syns))


# ---------- Test 3: Coalition test (Eq. 3) ----------

def coalition_test(
    X_codes: np.ndarray,
    V_codes: np.ndarray,
    lag: int = 1,
    method: str = "jeffreys",
) -> dict:
    """I_3 and G_3 for triplets vs future macro.

    I_3 = I({X_i,t, X_j,t, X_k,t}; V_{t+ℓ})
    G_3 = I_3 - max over pairs of I_2 = I({X_i,t, X_j,t}; V_{t+ℓ})
    Returns dict with median I_3 and median G_3 across triplets.
    """
    n_rounds, n_agents = X_codes.shape
    if n_rounds <= lag or n_agents < 3:
        return {"I3": float("nan"), "G3": float("nan")}

    V_future = V_codes[lag:]
    I3_list = []
    G3_list = []
    for i, j, k in combinations(range(n_agents), 3):
        Xi = X_codes[:-lag, i]
        Xj = X_codes[:-lag, j]
        Xk = X_codes[:-lag, k]
        I3 = mi_joint([Xi, Xj, Xk], V_future, method=method)
        I2_ij = mi_joint([Xi, Xj], V_future, method=method)
        I2_ik = mi_joint([Xi, Xk], V_future, method=method)
        I2_jk = mi_joint([Xj, Xk], V_future, method=method)
        G3 = I3 - max(I2_ij, I2_ik, I2_jk)
        I3_list.append(I3)
        G3_list.append(G3)
    return {
        "I3": float(np.median(I3_list)),
        "G3": float(np.median(G3_list)),
    }


# ---------- Test 4: Agent differentiation (mixed models m0/m1/m2) ----------

def differentiation_test(devs_arr: np.ndarray) -> dict:
    """Fit Riedl's three nested mixed models and report likelihood-ratio p-values.

    m0: y = β0 + u_time + ε
    m1: y = β0 + u_time + u_agent + ε
    m2: y = β0 + u_time + u_agent + u_agent:time + ε

    Returns p-values:
      p_m0_vs_m1: does adding agent random intercepts improve fit?
                  (agents differ in baseline)
      p_m1_vs_m2: does adding agent random slopes (by time) improve fit?
                  (agents differ in learning rate)
    """
    n_rounds, n_agents = devs_arr.shape

    # Long-form dataframe.
    rows = []
    for t in range(n_rounds):
        for a in range(n_agents):
            rows.append({"y": float(devs_arr[t, a]),
                         "time": t + 1, "agent": f"a{a}"})
    df = pd.DataFrame(rows)

    # statsmodels MixedLM uses formula-style fitting. We treat time as a
    # categorical random effect via a `groups`/`re_formula` trick where needed.
    # The simplest practical comparison:
    #   m0: only time variance (groups='time')
    #   m1: m0 + agent intercept (groups='agent')  -- different grouping
    #   m2: m1 + slope in time per agent
    # Riedl's design has crossed random effects (time × agent). statsmodels
    # supports this via VarianceComponents; we use the simpler approach of
    # fitting agent-grouped m1/m2 and time-grouped m0 since the LRTs we care
    # about are m0 vs m1 (does between-agent variance exist) and m1 vs m2
    # (do agent-specific slopes add explanatory power).
    p_m0_m1 = float("nan")
    p_m1_m2 = float("nan")

    try:
        # m0: agent label collapsed — only time fixed effect
        m0 = smf.mixedlm("y ~ time", df, groups=df["agent"]).fit(
            reml=False, method="powell", maxiter=200)
        # m1: add agent-level random intercept implicitly via groups='agent'
        # (m0 is *also* grouped by agent here but with a degenerate variance).
        # For a more honest m0, fit OLS with time-only fixed effect.
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        Xmat = add_constant(df[["time"]])
        m0_ols = OLS(df["y"], Xmat).fit()
        ll0 = m0_ols.llf
        ll1 = m0.llf
        df_diff = 1  # one extra variance component
        lr = 2 * (ll1 - ll0)
        if lr > 0:
            p_m0_m1 = 1.0 - chi2.cdf(lr, df_diff)
        else:
            p_m0_m1 = 1.0

        # m2: add random slope in time by agent
        m2 = smf.mixedlm(
            "y ~ time", df, groups=df["agent"], re_formula="~time"
        ).fit(reml=False, method="powell", maxiter=200)
        ll2 = m2.llf
        df_diff = 2  # slope variance + covariance
        lr = 2 * (ll2 - ll1)
        if lr > 0:
            p_m1_m2 = 1.0 - chi2.cdf(lr, df_diff)
        else:
            p_m1_m2 = 1.0
    except Exception as e:
        # If fitting fails (singular covariance, etc.), return NaNs and the error.
        return {
            "p_m0_vs_m1": float("nan"),
            "p_m1_vs_m2": float("nan"),
            "error": str(e),
        }

    return {
        "p_m0_vs_m1": float(p_m0_m1),
        "p_m1_vs_m2": float(p_m1_m2),
    }
