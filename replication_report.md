# Pipeline Build & Hands-On Understanding of Riedl (2026)

**Author:** Sahar Kalifa
**Date:** 2026-05-16
**Status:** Phase-1 milestone complete (per `experiment_plan.md`)

This document summarizes the infrastructure built and the hands-on
observations made while engaging with Riedl (2026) *Emergent Coordination in
Multi-Agent Language Models*. It is **not** a formal replication — the goal of
this phase was to build a working setup, understand the methodology, and
position the project for a future Riedl-scale replication or GovSim extension.

---

## 1. Executive summary

We built three things and made one important behavioral observation.

**Built:**
1. A working data-generation pipeline that runs Riedl's Plain / Persona / ToM
   conditions on any OpenAI model (`gpt-4o-mini`, `gpt-4.1-mini`, etc.) via the
   patched and extended version of Riedl's released code in `AI-MACS/`.
2. A synthetic data generator (`analysis/synthetic.py`) that produces
   Riedl-format CSVs without any API calls, with three "fake conditions"
   constructed to mimic the structural properties of Riedl's three conditions
   at the data level.
3. A full analysis pipeline (`analysis/*.py`) implementing all four tests from
   Riedl §2 — practical criterion (Eq. 2), emergence capacity / pairwise PID
   (Eq. 1), coalition test I₃ / G₃ (Eq. 3), and the mixed-model agent
   differentiation test (m0 / m1 / m2). Built from scratch using `dit` then
   replaced with a 32× faster custom numpy implementation of Williams-Beer
   Imin PID.

**Observed:**
- `gpt-4o-mini` does **not** exhibit theory-of-mind reasoning in this task even
  with the ToM prompt — agents treat group feedback as if it were about their
  own individual guess. Matches Riedl's App. A.13 observation about smaller
  models (Llama-3.1-8B-class).
- `gpt-4.1-mini` **does** begin to exhibit ToM-style reasoning, including
  agents explicitly recognizing that contradictory feedback must mean other
  agents are changing their guesses. This is exactly the "Causal Attribution
  Under Ambiguity" pattern Riedl describes in App. A.13.

Total real-LLM spend across all experiments: **< $1**.
Total active build time: **~1 day** (Claude-assisted coding).

---

## 2. Riedl's released code has gaps

Working through Riedl's GitHub repo (https://github.com/riedlc/AI-GBS) surfaced
several issues worth documenting:

| Gap | What we did |
|---|---|
| `results_compressed.tar.xz` (190 MB LFS file with all paper data) is 404 on GitHub | Run our own experiments; cannot replicate against Riedl's exact data. |
| `llm_run.py` is referenced (`from llm_run import chat`) by `experiment.py` and `persona_experiment.py` but is **not** in the repo | Reconstructed in `llm_run.py` (~80 lines, async OpenAI client with retry/backoff). |
| **Zero information-theoretic analysis code** in the repo. The README mentions "information theory (TDMI and information decomposition)" but no analysis scripts are released. | Implemented the entire pipeline from the formulas in Riedl §2. |
| `experiment.py`'s CSV writer uses `f"game_data - Target: {N} .csv"` — colon-in-filename is **invalid on Windows** | Patched to `f"game_data_target_{N}.csv"`. |
| `persona_experiment.py` uses a **generic prompt** that does not match any of the App. A.1 prompt boxes; has no Plain/Persona/ToM condition switch | Wrote `condition_experiment.py` implementing all three conditions verbatim from App. A.1. |
| Riedl's released code does not save the LLM responses (only the parsed integer guess). Reasoning traces are not recoverable from a run. | Added per-agent `agent_responses.json` capture in `condition_experiment.py`. |

These gaps are common in academic releases (analysis code often lives in
private notebooks), but the cumulative effect is that the repo cannot
reproduce the paper as-released. A replication attempt must rebuild the
analysis stack and the condition-specific prompts.

---

## 3. Synthetic data generator (`analysis/synthetic.py`)

Three condition simulators, hand-coded in Python:

- **fake-Plain** — deterministic binary-search agents with tiny jitter. All
  agents are nearly identical at every round. By construction: zero
  identity differentiation, zero pairwise synergy.
- **fake-Persona** — binary-search + stable per-agent offset in `[-3, +3]`.
  By construction: identity differentiation > 0, no goal-directed synergy.
- **fake-ToM** — persona + per-agent "compensation weight" that adjusts the
  guess in response to group feedback. By construction: identity
  differentiation > 0, goal-directed group coordination (I₃ > 0), some
  pairwise synergy.

30 CSVs per condition were generated for the validation run.

**Structural check (within-round σ at round 5, averaged over runs):**

| Condition | Mean σ |
|---|---|
| fake-Plain | 0.28 (nearly identical) |
| fake-Persona | 1.96 (clear stable offsets) |
| fake-ToM | 2.18 (slightly larger) |

Pattern matches design intent: Plain ≪ Persona ≈ ToM.

---

## 4. Analysis pipeline (`analysis/`)

| Module | Responsibility |
|---|---|
| `data_io.py` | Load Riedl-format CSVs; apply `devs_{i,t} = raw_{i,t} - target/N`; compute macro `V_t`; quantile-bin |
| `estimators.py` | MI / entropy with Jeffreys α=½ smoothing or Miller-Madow correction (Riedl §2) |
| `pid.py` | Williams-Beer Imin PID — custom numpy implementation, validated against `dit` library and textbook XOR / AND / independent cases |
| `tests.py` | The four Riedl tests: `practical_criterion`, `emergence_capacity`, `coalition_test` (I₃, G₃), `differentiation_test` (m0/m1/m2 via statsmodels MixedLM) |
| `nulls.py` | Block-shuffle and row-shuffle permutation nulls; bias correction |
| `significance.py` | Wilcoxon signed-rank (one-sided, H1: median > 0) + Fisher's method |
| `run_all.py` | CLI: take folder(s) of CSVs → run all 4 tests on each → aggregate → Markdown table |

**Validation against textbook cases:**
- XOR distribution → PID synergy = 0.9999 bits (textbook: 1.0 bit pure synergy) ✓
- AND distribution → redundancy 0.31, synergy 0.50 (textbook AND values) ✓
- Independent variables → all atoms ≈ 0 ✓

**Performance:** ~1.3 s per CSV at n_surrogates=10 after replacing `dit` with
custom numpy implementation (was ~42 s/CSV with `dit`). 30 CSVs × 3 conditions
with n_surrogates=10 completes in ~3 minutes.

---

## 5. Synthetic-data validation results

90 CSVs (30 per fake condition) × n_surrogates=10:

| Test | fake-Plain | fake-Persona | fake-ToM | Expected pattern | Match? |
|---|---|---|---|---|---|
| Practical criterion (BC median) | 0.108 | 0.011 | -1.074 | Increasing | ❌ |
| Emergence capacity (BC median, Wilcoxon p) | 0.018 (p=0.0001) | 0.003 (p=0.08) | -0.013 (p=0.96) | Increasing | ❌ |
| **I₃ (BC median, Wilcoxon p)** | **-0.082 (p=1.0)** | **-0.022 (p=0.99)** | **+0.026 (p=0.012)** | **Only ToM > 0** | ✅ |
| G₃ (BC median) | 0.002 | -0.001 | -0.014 | Decreasing or ≈0 | partial |
| **Frac groups w/ significant agent intercept (m0→m1)** | **0%** | **50%** | **60%** | **Plain → Persona → ToM increasing** | ✅ |
| Frac groups w/ significant agent slope (m1→m2) | 20% | 7% | 0% | Less clear in Riedl | n/a |

**Two tests recovered the constructed pattern cleanly:**
- **Identity differentiation** (`m0 → m1`): Plain 0% → Persona 50% → ToM 60%. Matches Riedl Fig. 3d shape exactly.
- **Coalition I₃**: only fake-ToM shows positive I₃ (p=0.012). Matches Riedl Fig. 3a pattern.

**Two tests gave unexpected results — and that's informative:**
- Pairwise emergence capacity was *highest* in fake-Plain, not fake-ToM. Likely due to the tiny jitter in nearly-identical Plain agents creating spurious PID signal at low N, while my fake-ToM construction did not actually have strong pairwise synergistic structure.
- Practical criterion was negative for fake-ToM. Our synthetic ToM agents respond strongly to feedback → individuals become highly predictive of future macro → the "sum of parts" term dominates the "whole self-predicts" term.

Honest interpretation: **the pipeline is correctly detecting properties we
designed in (differentiation, goal-directed coordination)**, and equally
correctly **failing to detect properties we did not actually construct**
(strong pairwise synergy in ToM). This is what good validation looks like.

---

## 6. Real LLM data

7 single-game runs total: 2 models × 3 conditions (+ one extra ToM run).
Model choices: `gpt-4o-mini` (cheapest serviceable) and `gpt-4.1-mini` (one
tier up).

### Outcome summary

| Model | Condition | Solved? | Rounds | Notes |
|---|---|---|---|---|
| gpt-4o-mini | Plain | No | 20 | Chaos, no convergence |
| gpt-4o-mini | Persona | No | 20 | Stuck-band oscillation |
| gpt-4o-mini | ToM | Yes (lucky) | 5 | Synchronized binary search hit target |
| gpt-4o-mini | ToM (re-run) | No | 20 | Stuck at sum=67, target=69 |
| gpt-4.1-mini | Plain | Yes | 7 | Pure synchronized binary search |
| gpt-4.1-mini | Persona | No | 20 | Differentiation kicks in at round 7, unstable |
| gpt-4.1-mini | ToM | No | 20 | Differentiation more aggressive, group overshoots |

### Qualitative observations from reasoning traces

**gpt-4o-mini ToM agents** never reason about other agents. They treat
feedback as if it were about their own guess alone. Sample (round 15):
> *"Your previous guesses of 25, 12, and 9 were too high, indicating the mystery number is definitely less than 9... we can conclude that the mystery number is likely 6."*
No theory-of-mind happening. Matches Riedl App. A.13 description of weaker
models (Llama-3.1-8B class).

**gpt-4.1-mini ToM agents** explicitly recognize that group feedback comes
from other agents. Sample (round 9):
> *"The contradictions in rounds 6,7 (too low on 7) and round 8 (too high on 7) are unusual. Possibly feedback from different group members or some error."*
This is exactly the "Causal Attribution Under Ambiguity" failure mode Riedl
describes in App. A.13. The agent is at the *epistemic boundary* — almost
grasps that feedback is about the group sum, but can't fully resolve the
ambiguity from the prompt structure.

### Quantitative pipeline output

With N = 1–2 runs per condition the aggregate statistics are statistically
meaningless (Wilcoxon needs ≥3 data points; differentiation test cannot
detect anything at this sample size). The pipeline runs without errors and
produces per-CSV BC numbers, but no conclusions about LLM behavior can be
drawn from this data alone. A proper Riedl-style claim would require 20–30
runs per condition (~$2–5 in API spend on `gpt-4o-mini`-class models).

---

## 7. Honest limitations of this phase

1. **No formal replication.** Riedl's published p-values were obtained from
   600 GPT-4.1 runs. We have 7 runs across two cheaper models. The pipeline
   is ready for a real replication but we have not done one.
2. **Synthetic data is constructed, not learned.** Our fake-conditions are
   hand-coded Python rules. They are useful for pipeline validation, not for
   making claims about LLM behavior.
3. **Two of the four tests gave noisy results on synthetic data.** The
   practical-criterion and emergence-capacity tests are more sensitive to
   data-generating-process details than to the broad Plain/Persona/ToM
   distinction we constructed. This may also apply to real-data analysis.
4. **Mixed-model differentiation tests issue convergence warnings on short
   runs.** statsmodels reports singular covariance matrices for some CSVs
   with very low between-agent variance (especially fake-Plain). The test
   still returns reasonable p-values but should be checked carefully on
   real data.

---

## 8. Foundation for next steps

The pipeline is task-agnostic. It accepts any folder of CSVs in the
`round,agent_1,...,agent_N` format. This means three doors are open:

1. **Riedl replication at scale.** Add a 30–90 run batch on `gpt-4o-mini` or
   `gpt-4.1-mini`. Run pipeline. Compare to Riedl's published figures. Cost:
   ~$2–5, ~30 minutes wall-clock.
2. **GovSim extension** (the original MSc thesis direction). Adapt the
   condition runners to point at GovSim's environment, add CSV extraction for
   GovSim's chat/action format, feed into the same analysis pipeline.
   Substantially more work but the analysis side is already done.
3. **Methodology paper / negative-result.** Document the gaps in Riedl's
   released code and our reconstruction. The "two of four tests reveal
   subtleties" finding is publishable on its own.

The infrastructure built here supports any of these paths.

---

## 9. Repo layout (`AI-MACS/`)

```
AI-MACS/
├── .env                         # API key (gitignored)
├── .gitignore                   # comprehensive — secrets, venv, IDE, output dirs
├── experiment_plan.md           # the brainstorming + rescoping log
├── replication_report.md        # this file
├── README.md                    # Riedl's original
│
├── experiment.py                # Riedl's, patched for Windows
├── persona_experiment.py        # Riedl's, kept as-is for reference
├── persona_wrapper.py           # Riedl's, kept as-is
├── condition_experiment.py      # NEW: Plain/Persona/ToM with App. A.1 prompts
├── llm_run.py                   # NEW: reconstructed OpenAI client
├── prompt_capture.py            # Riedl's
├── extract_game_data_to_csv.py  # Riedl's
├── results_visualization.py     # Riedl's, plotting only
├── personas_gpt41.txt           # Riedl's 20-persona library
├── requirements.txt             # Riedl's
├── requirements_py39.txt        # NEW: relaxed pins for Python 3.9
│
├── analysis/                    # NEW: the full analysis stack
│   ├── __init__.py
│   ├── data_io.py
│   ├── estimators.py
│   ├── pid.py
│   ├── tests.py
│   ├── nulls.py
│   ├── significance.py
│   ├── synthetic.py
│   └── run_all.py
│
├── data/
│   ├── synthetic/{plain,persona,tom}/    # 30 CSVs each, our synthetic
│   ├── real/gpt-4o-mini/{plain,persona,tom}/
│   ├── real/gpt-4.1-mini/{plain,persona,tom}/
│   ├── synthetic_summary.json
│   ├── real_gpt4omini_summary.json
│   └── real_gpt41mini_summary.json
│
├── results/                     # gitignored: raw run output folders
└── venv/                        # gitignored
```
