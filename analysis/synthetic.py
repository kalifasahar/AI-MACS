"""Synthetic data generator: Riedl-format CSVs with no LLM calls.

Three "fake conditions" mimic the structural properties Riedl's four tests
should detect, so that the analysis pipeline can be validated against
ground-truth data we constructed.

  fake-Plain    : deterministic binary-search agents, all identical.
                  Expected pipeline output: low/zero differentiation,
                  low/zero synergy, low I_3 / G_3.

  fake-Persona  : binary-search + stable per-agent offset.
                  Expected: differentiation > 0 (m1 vs m0 significant),
                  synergy ~ 0 (agents independent given their offset),
                  I_3 still ~ 0 (no goal-directed coordination).

  fake-ToM      : persona + active cross-agent complementary adjustment.
                  Expected: differentiation > 0, synergy > 0, I_3 > 0,
                  G_3 possibly > 0.

Output format matches Riedl exactly:
  round,agent_1,agent_2,...,agent_N
  1,25,25,...,25
  2,12,12,...,12
  ...

CLI:
  python -m analysis.synthetic plain --n_runs 30 --num_agents 10
  python -m analysis.synthetic persona --n_runs 30
  python -m analysis.synthetic tom --n_runs 30
  python -m analysis.synthetic all --n_runs 30    # all three conditions
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


# ---------- Game-state types ----------

@dataclass
class HistoryEntry:
    """One round of game history visible to all agents."""
    round_num: int
    guesses: List[int]    # per-agent guesses, indexed by agent_id
    group_sum: int
    feedback: str         # "too HIGH" | "too LOW" | "CORRECT"


# ---------- Strategy: binary-search base ----------

def binary_search_range(history: List[HistoryEntry], guess_range: tuple = (0, 50)) -> tuple:
    """Tighten the (low, high) range based on group-sum feedback,
    interpreted naively as feedback about the individual agent's own midpoint.

    This is what a single-agent binary searcher would do, ignoring N.
    """
    low, high = guess_range
    for h in history:
        # Use the agent's own past guess (same for all in fake-Plain).
        # We pass history with per-agent guesses; binary-search just uses midpoint.
        # For simplicity, take the average of the last round's guesses as "the guess".
        past = sum(h.guesses) // len(h.guesses)
        if h.feedback == "too HIGH":
            high = min(high, past - 1)
        elif h.feedback == "too LOW":
            low = max(low, past + 1)
        if low > high:
            # Range collapsed — reset to be safe
            low, high = 0, 50
    return low, high


def midpoint(low: int, high: int) -> int:
    return (low + high) // 2


# ---------- Three condition-specific agent strategies ----------

def plain_agent(
    agent_id: int,
    round_num: int,
    history: List[HistoryEntry],
    role: dict,
) -> int:
    """fake-Plain: deterministic binary search. All agents identical."""
    low, high = binary_search_range(history)
    return midpoint(low, high)


def persona_agent(
    agent_id: int,
    round_num: int,
    history: List[HistoryEntry],
    role: dict,
) -> int:
    """fake-Persona: binary search + a stable per-agent offset.

    role[agent_id] = integer offset in [-3, +3] (assigned at game start).
    Adds tiny noise so agents aren't perfectly deterministic.
    """
    low, high = binary_search_range(history)
    base = midpoint(low, high)
    offset = role.get(agent_id, 0)
    noise = random.choice([-1, 0, 0, 0, 1])  # mostly zero, light jitter
    guess = base + offset + noise
    return max(0, min(50, guess))


def tom_agent(
    agent_id: int,
    round_num: int,
    history: List[HistoryEntry],
    role: dict,
) -> int:
    """fake-ToM: persona + active complementary adjustment to close the group gap.

    Each agent has:
      - a stable offset (like Persona)
      - a "compensation weight" w[agent_id] in [0, 1]: how aggressively this
        agent absorbs the group error
    When the previous round was off-target, the agent shifts its guess by
    w[agent_id] * gap / N to push the group closer. Different weights across
    agents create COMPLEMENTARY adjustments — pairwise synergy.
    """
    low, high = binary_search_range(history)
    base = midpoint(low, high)
    offset = role.get(agent_id, 0)
    weight = role.get(f"w_{agent_id}", 0.0)

    # If we have a previous round, compute the error and compensate.
    compensation = 0
    if history:
        last = history[-1]
        # Approximate gap from feedback magnitude.
        # We don't know the target, but we know SIGN: "too HIGH" → reduce, "too LOW" → increase.
        if last.feedback == "too HIGH":
            # Group needs to go LOWER. High-weight agents reduce more.
            compensation = -int(round(weight * 3))   # up to -3
        elif last.feedback == "too LOW":
            compensation = int(round(weight * 3))    # up to +3

    noise = random.choice([-1, 0, 0, 1])
    guess = base + offset + compensation + noise
    return max(0, min(50, guess))


STRATEGIES: dict[str, Callable] = {
    "plain": plain_agent,
    "persona": persona_agent,
    "tom": tom_agent,
}


# ---------- Game runner ----------

def assign_roles(condition: str, num_agents: int, seed: int) -> dict:
    """Generate per-agent role parameters for a single run."""
    rng = random.Random(seed)
    role = {}
    if condition == "plain":
        return role  # no per-agent state
    # Persona/ToM: stable identity offsets in [-3, +3], excluding 0.
    offsets = list(range(-3, 4))
    for i in range(num_agents):
        role[i] = rng.choice(offsets)
    if condition == "tom":
        # Per-agent compensation weights — varied across agents to create
        # heterogeneous response → cross-agent synergy.
        for i in range(num_agents):
            role[f"w_{i}"] = rng.uniform(0.2, 1.0)
    return role


def play_one_game(
    condition: str,
    num_agents: int,
    max_rounds: int,
    seed: int,
    target_range: tuple = (10, 245),  # plausible sum-target range
) -> List[List[int]]:
    """Play one synthetic game; return list of round guess-vectors."""
    rng = random.Random(seed + 100000)
    target = rng.randint(*target_range)
    strategy = STRATEGIES[condition]
    role = assign_roles(condition, num_agents, seed)

    history: List[HistoryEntry] = []
    all_rounds: List[List[int]] = []

    for r in range(1, max_rounds + 1):
        # Each agent picks a guess based on shared history.
        guesses = [strategy(i, r, history, role) for i in range(num_agents)]
        # Inject minimal random noise for "plain" so it's not literally identical
        # (which would make some MI estimators fail with zero variance).
        if condition == "plain":
            # Replace one or two agents' guesses with +-1 randomly to prevent
            # singular MI distributions.
            jitter_count = rng.randint(0, 2)
            for _ in range(jitter_count):
                idx = rng.randrange(num_agents)
                guesses[idx] = max(0, min(50, guesses[idx] + rng.choice([-1, 1])))

        group_sum = sum(guesses)
        if group_sum == target:
            feedback = "CORRECT"
        elif group_sum > target:
            feedback = "too HIGH"
        else:
            feedback = "too LOW"

        history.append(HistoryEntry(round_num=r, guesses=guesses,
                                    group_sum=group_sum, feedback=feedback))
        all_rounds.append(guesses)

        if feedback == "CORRECT":
            break

    return all_rounds


# ---------- CSV writing ----------

def write_csv(out_path: Path, rounds: List[List[int]]) -> None:
    num_agents = len(rounds[0]) if rounds else 0
    headers = ["round"] + [f"agent_{i+1}" for i in range(num_agents)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r_idx, guesses in enumerate(rounds, start=1):
            w.writerow([r_idx] + guesses)


# ---------- Batch generator ----------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def generate_batch(
    condition: str,
    n_runs: int,
    num_agents: int = 10,
    max_rounds: int = 20,
    base_seed: int = 0,
    out_dir: Optional[Path] = None,
) -> Path:
    """Generate n_runs CSVs into out_dir / condition / run_NNN.csv. Return the dir."""
    assert condition in STRATEGIES, f"Unknown condition {condition!r}"
    if out_dir is None:
        out_dir = DATA_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(n_runs):
        seed = base_seed + k
        rounds = play_one_game(condition, num_agents, max_rounds, seed)
        path = out_dir / f"run_{k+1:03d}.csv"
        write_csv(path, rounds)

    print(f"[{condition}] wrote {n_runs} CSVs to {out_dir}")
    return out_dir


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Riedl-format data.")
    parser.add_argument("condition", choices=list(STRATEGIES.keys()) + ["all"],
                        help="Which condition to generate (or 'all').")
    parser.add_argument("--n_runs", type=int, default=30, help="Runs per condition (default 30).")
    parser.add_argument("--num_agents", type=int, default=10, help="Agents per game (default 10).")
    parser.add_argument("--max_rounds", type=int, default=20, help="Max rounds per game (default 20).")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    args = parser.parse_args()

    conditions = list(STRATEGIES.keys()) if args.condition == "all" else [args.condition]
    for c in conditions:
        generate_batch(
            condition=c,
            n_runs=args.n_runs,
            num_agents=args.num_agents,
            max_rounds=args.max_rounds,
            base_seed=args.seed,
        )


if __name__ == "__main__":
    main()
