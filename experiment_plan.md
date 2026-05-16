# Experiment Plan — Infrastructure & Hands-On Understanding of Riedl (2026)

**Author:** Sahar Kalifa
**Date:** 2026-05-13 (initial), 2026-05-16 (rescoped)

**Revised goal (2026-05-16):** Build a working end-to-end setup — data generation + analysis pipeline — and gain hands-on understanding of Riedl's (2026) methodology. This is *not* a formal replication of Riedl's findings; that is a separate later phase if needed.

Concretely we want four things:
1. A working data-generation pipeline running real LLM experiments (even at tiny scale)
2. A synthetic-data generator that produces Riedl-format CSVs mimicking the three conditions
3. A working analysis pipeline implementing all four of Riedl's tests (`dit` + `scipy` + `statsmodels`)
4. Hands-on understanding: read real agent reasoning traces; run analysis on both real and synthetic data; see what the numbers look like; be able to explain the framework to an advisor

**Not in scope for this phase:** matching Riedl's published p-values, claiming a formal replication, statistical power at the level of Riedl's 200 runs/condition, GovSim integration, the MACS index.

## Four-step execution plan

| Step | Deliverable | Active time | Cost |
|---|---|---|---|
| 1. Small set of real LLM examples | 3 runs `experiment.py` (preliminary prompt, already partially done in Phase 0) + 1 run each of Plain / Persona / ToM via `persona_experiment.py` | ~30 min wall-clock | ~$0.15 |
| 2. Synthetic data generator | `analysis/synthetic.py` producing Riedl-format CSVs for "fake-Plain", "fake-Persona", "fake-ToM" — ~30 CSVs per condition | ~half day my coding | $0 |
| 3. Analysis pipeline | `analysis/` module: PID via `dit`, MI / bias-correction, four tests, permutation nulls, Fisher + Wilcoxon, mixed-model differentiation. CLI entry point. | ~1 day my coding | $0 |
| 4. Run + compare | Pipeline on synthetic data (must show the qualitative Plain→Persona→ToM pattern we constructed in); pipeline on real LLM data (small N, just to see numbers come out) | ~1 hour | $0 |

**Total: ~2 days my active work, < $0.20 API spend.**

---

## 1. What the cloned repo gives us (✅)

Repo: `AI-GBS/` (cloned from https://github.com/riedlc/AI-GBS)

**Experiment runners:**
- `experiment.py` — single experiment runner; group binary search game, async OpenAI calls. (563 lines)
- `persona_experiment.py` — Persona and ToM condition implementations. (455 lines)
- `persona_wrapper.py` — thin wrapper that injects a persona text into the base experiment. (29 lines)
- `run_experiment_multi_model.py` — large-scale orchestration (parallel runs, batches, resume support). (388 lines)
- `launch_experiments.sh` — shell launcher that invokes the orchestrator (Unix-style; we may need a `.bat` equivalent on Windows or just run the Python directly).

**Inputs / configuration:**
- `personas_gpt41.txt` — the 20-persona library used in the paper. (6 KB)
- `requirements.txt` — minimal pip dependencies: `openai`, `numpy`, `matplotlib`, `pydantic`, `httpx`, `tqdm`, `python-dotenv`. **Note: no `scipy`, `statsmodels`, `dit`, or `pandas`** — those we add ourselves for the analysis pipeline.

**Logging / data extraction:**
- `prompt_capture.py` — captures the prompts sent to the LLM for selected rounds (useful for auditing what each agent actually saw). (56 lines)
- `extract_game_data_to_csv.py` — converts raw experiment logs (JSON) into the round × agent CSV format. (106 lines)

**Output handling:**
- `results_visualization.py` — pure matplotlib plots only (`parse_game_log`, `collect_results`, `create_plots`). **No statistics, no information theory.** (350 lines)
- `results_compresser.py` — utility to tar/xz-compress a results folder. (56 lines)
- `results_compressed.tar.xz` — **404'd during clone** (a 190 MB Git LFS file that the repo points to but the server no longer hosts). Would have contained the paper's raw experimental data. Not needed for our work — we run our own experiments.

**Sample outputs (useful for format reference):**
- `sampled_data/*.csv` — 6 small example output CSVs (10–16 lines each) from prior runs on `gpt-4o-mini`, `llama-3.3-70b`, and `gemini-2.5-flash`, split into SUCCESS / FAILURE.

Data format:
```
round, agent_1, agent_2, ..., agent_N
1,     27,      27,       ...  27
2,     0,       4,        ...  0
```

## 2. What is MISSING (❌) — the work we must do

**The repo provides only data generation. Zero information-theoretic analysis code is included.** Confirmed by `grep` across all `.py` files: no PID, no MI, no entropy, no permutation tests, no mixed models, no scipy/statsmodels imports anywhere. `results_visualization.py` is matplotlib only.

This is normal for academic releases but means the entire **emergence-analysis pipeline** must be built. The good news: the math is fully specified in §2 of the paper, and all of it can be implemented via mature Python libraries.

---

## 3. Riedl's Analytical Framework — the FOUR tests

Per the paper §2: *"We implement three tests"* (info-theoretic) plus the *"Test of Agent Differentiation"* (statistical). Four tests total, each providing a non-redundant piece of evidence.

| # | Test | Formula | Unique role |
|---|---|---|---|
| 1 | **Emergence capacity** | Eq. 1: pairwise PID synergy `Syn_ij`, median over pairs | Detects pairwise synergy without needing a macro signal |
| 2 | **Practical criterion** | Eq. 2: `S_macro(ℓ) = I(V_t; V_{t+ℓ}) − Σ_k I(X_k,t; V_{t+ℓ})` | Whole-system emergence: macro is more predictable than sum of parts |
| 3 | **Coalition test** | Eq. 3: `I_3` and `G_3 = I_3 − max(I_2)` | Localizes whether structure is higher-order (triplets) vs. trivially pairwise |
| 4 | **Agent differentiation** | Hierarchical mixed models m0/m1/m2 with likelihood-ratio tests | Distinguishes stable identity-linked roles from transient learning-rate heterogeneity |

Each test rules out a specific alternative explanation. The paper says directly: *"No single measure does all four."*

### Shared machinery (used across tests 1–3)
- Microstate transform: `devs_{i,t} = raw_{i,t} − target/N`
- Macro signal: `V_t = Σ devs_{i,t}`
- Quantile binning (K=2 main, K=3 sensitivity)
- Bias correction: Jeffreys α=½ Dirichlet prior (main), Miller–Madow (robustness)
- Permutation null: column-block-shuffle (block size ℓ=2, B=200), row-shuffle (identity-breaking)
- Significance: Wilcoxon signed-rank one-sided (H1: median > 0), Fisher's method to combine p-values across groups

### Deferred (paper Appendix A.11, optional)
- Causal mediation analysis (treatment → synergy → performance)
- Stabilized IPW for early-synergy censoring correction
- Persona-similarity LMRA analysis (Appendix A.14)

These are not needed for the core qualitative replication and add scope. Build them only if the core replication succeeds and we want stronger claims.

---

## 4. Python library mapping

Every component has a mature library. Pipeline = wiring libraries together, not deriving math.

| Component | Library | Notes |
|---|---|---|
| Williams–Beer PID with I_min redundancy | **`dit`** | First-class object: `dit.pid.PID_WB`. Replaces re-deriving Eq. 1. |
| Mutual information `I(X; Y)` | `dit` or `scipy.stats` | Multiple options; `dit` chosen for consistency with PID. |
| Plug-in entropy estimator | `scipy.stats.entropy` / `dit` | Trivial. |
| Jeffreys α=½ Dirichlet smoothing | `dit` built-in, or 2-line numpy | Adds ½ to every count cell before normalizing. |
| Miller–Madow bias correction | 1-line numpy: `Î_plug-in + (K−1)/(2N)` | Textbook formula. |
| Permutation nulls (block-shuffle, row-shuffle) | `numpy.random` + loop | Custom but ~10 lines each. |
| Wilcoxon signed-rank | `scipy.stats.wilcoxon(alternative='greater')` | Direct match. |
| Fisher's method | `scipy.stats.combine_pvalues(method='fisher')` | One call. |
| Mixed-model differentiation test | `statsmodels.regression.mixed_linear_model.MixedLM` | Python equivalent of R's `lme4` for Riedl's m0/m1/m2. |
| Quantile binning | `pandas.qcut` | One line. |

**Two libraries do 80% of the work:** `dit` (information theory) and `scipy.stats` + `statsmodels` (everything else).

---

## 5. Minimal Replication Design

The paper used 600 runs × GPT-4.1 main + 4 other models × 300 each. We can't reproduce that and don't need to. Goal = **qualitative reproduction**.

| Parameter | Riedl (paper) | Our minimal plan | Rationale |
|---|---|---|---|
| Model | GPT-4.1 | **`gpt-4o-mini` (locked)** | Decision made by Sahar after seeing cost comparison. ~150× cheaper than GPT-4.1 ($2 vs. $30 for our batch). Risk acknowledged: Riedl's §4.4 shows LLAMA-3.1-8B (similar capability tier) largely failed the task — if `gpt-4o-mini` doesn't show the ToM effect at smoke-test time, escalate to `gpt-4.1-mini` (~$6) or `gpt-4.1` (~$30). |
| Group size | N = 10 | N = 10 | Match. |
| Temperature | T = 1.0 | T = 1.0 | Match. |
| Conditions | Plain, Persona, ToM | Plain, Persona, ToM | All three — ToM is the punchline. |
| Replications / condition | 200 | **30** | Sufficient power for Fisher-combined p-values on the large effects in Figs. 2–3. |
| **Total runs** | 600 | **90** | Tractable. |

### Numerical comparison: our run vs. Riedl's run

**Sources for the numbers below:**
- ✅ = stated verbatim in the paper
- 🟡 = derived by us from paper structure (e.g., calls = agents × rounds)
- ⚠️ = our own estimate or guess (e.g., tokens per call, dollar costs) — paper does not report these

**Per-experiment numbers (fixed regardless of plan):**
- ✅ N = 10 agents per group (paper §3.2)
- 🟡 ~15 rounds average to converge (estimated from Riedl Fig. A1 distribution; paper does not state a number explicitly)
- 🟡 ~150 API calls per experiment (10 agents × ~15 rounds)
- ⚠️ ~500 input tokens + ~150 output tokens per call — **our estimate** from reading the prompt text in App. A.1; paper does not report tokens
- 🟡 ~30–120 seconds wall-clock per experiment

**Side-by-side comparison — Riedl (paper) vs. our planned small run vs. minimum-conceivable run:**

| | **Riedl (paper, main only)** | **Our small run (3-cond × 30)** | **Minimum (Plain × 30)** |
|---|---|---|---|
| Model | ✅ GPT-4.1 | **`gpt-4o-mini`** (locked) | `gpt-4o-mini` |
| Conditions | ✅ 3 (Plain/Persona/ToM) | 3 | 1 (Plain only) |
| Replications per condition | ✅ 200 | 30 | 30 |
| **Total experiments** | ✅ **600** | **90** | **30** |
| Agents per experiment (N) | ✅ 10 | 10 | 10 |
| Total agent-instantiations | 🟡 6,000 | 🟡 900 | 🟡 300 |
| Total API calls | 🟡 ~90,000 | 🟡 ~13,500 | 🟡 ~4,500 |
| Input tokens (total) | ⚠️ ~45 M | ⚠️ ~6.75 M | ⚠️ ~2.25 M |
| Output tokens (total) | ⚠️ ~13.5 M | ⚠️ ~2.03 M | ⚠️ ~0.68 M |
| Pricing per 1M tokens (input/output) | ⚠️ ~$2.00 / $8.00 (GPT-4.1) | $0.15 / $0.60 (`gpt-4o-mini`) | $0.15 / $0.60 |
| **Estimated cost** | ⚠️ **~$200** | ⚠️ **~$2.23** | ⚠️ **~$0.75** |
| Wall-clock (concurrent ×8) | 🟡 days | 🟡 ~15–25 min | 🟡 ~5–10 min |

**Cost math, fully shown for our small run (so you can sanity-check):**
- Input: 6.75 M tokens × $0.15 / 1M = **$1.01**
- Output: 2.03 M tokens × $0.60 / 1M = **$1.22**
- Total: **~$2.23**

**Pricing caveat:** OpenAI prices change. The numbers above use `gpt-4o-mini` published pricing as of plan-writing time ($0.15/$0.60 per 1M input/output). **Verify the current price on https://openai.com/api/pricing/ before launching the batch.** A 10× price change to the model would multiply our cost by 10× too.

Note: Riedl's ~$200 is our **estimate** for the main 600 experiments only. The paper also reports a preliminary sweep (✅ 7,150 experiments, paper §3.1) and a 4-model robustness section (✅ 1,200 more experiments, paper App. A.13). Our extrapolated total estimate for the full paper's API spend: ~$1,000–$3,000+ — but Riedl does not report actual dollar cost anywhere in the paper.

**Sub-options within our plan:**

| Plan | Experiments | API calls | Est. cost | Wall-clock |
|---|---|---|---|---|
| Plain + Persona + ToM × 30 (current plan) | 90 | 13.5k | **~$2.20** | ~15–25 min |
| Plain + Persona + ToM × 20 | 60 | 9.0k | ~$1.45 | ~10–15 min |
| Plain + ToM × 30 (drop Persona) | 60 | 9.0k | ~$1.45 | ~10–15 min |
| Plain + Persona + ToM × 15 | 45 | 6.75k | ~$1.10 | ~7–12 min |
| Plain only × 30 | 30 | 4.5k | ~$0.75 | ~5–10 min |
| Smoke test (1 experiment, N=5) | 1 | ~75 | **~$0.02** | ~1 min |

**Recommended first step:** Run the smoke test (~$0.02) before committing to any batch. It validates the entire setup chain (API key, dependencies, output format, model behavior) before spending the batch budget.

### Success criteria (qualitative replication)

Three target findings from the paper:
1. **Practical criterion (BC) > 0** in all three conditions (Wilcoxon p < 0.05 in at least 2/3).
2. **I₃ (BC) significantly > 0 *only* in ToM** (Plain & Persona ≈ 0, ToM p < 0.05).
3. **Agent differentiation rises** Plain → Persona → ToM (fraction of groups with significant random-intercept or random-slope effect increases monotonically).

3/3 → strong replication; ready to extend.
2/3 → partial; investigate the missing one, may need bigger model or more runs.
0–1/3 → re-evaluate; possibly `gpt-4o-mini` too weak; a negative result itself becomes thesis content.

---

## 6. Phased Workflow

Active coding is done by Claude. "Time" below = active work, not wall-clock waiting for API calls.

### Phase 0 — Setup (~30 min)
- [ ] Create Python venv inside `Msc_Sahar/`.
- [ ] Install Riedl's `requirements.txt` + our analysis deps: `dit`, `scipy`, `statsmodels`, `pandas`.
- [ ] Set up `.env` with `OPENAI_API_KEY` (Sahar provides).
- [ ] Smoke test: run a single experiment via `experiment.py` (5 agents, small to keep it cheap). Verify output CSV format matches what we expect.

### Phase 1 — Build the analysis pipeline (~1 day active)
Build `analysis/` folder with our own scripts, independent of Riedl's repo:

- [ ] `analysis/data_io.py` — load Riedl-format CSVs, apply `devs` transform, compute macro `V_t`.
- [ ] `analysis/estimators.py` — MI with plug-in / Jeffreys / Miller–Madow bias correction.
- [ ] `analysis/pid.py` — wrap `dit` for Williams–Beer two-source PID; expose `synergy(X_i, X_j, T)`.
- [ ] `analysis/tests.py` — the four tests:
  - `emergence_capacity(data)` → median pairwise PID synergy
  - `practical_criterion(data, lag=1)` → Eq. 2
  - `coalition_test(data, lag=1)` → I₃, G₃
  - `differentiation_test(data)` → mixed-model LR tests using `statsmodels`
- [ ] `analysis/nulls.py` — block-shuffle and row-shuffle permutation null distributions.
- [ ] `analysis/significance.py` — Wilcoxon + Fisher combination.
- [ ] `analysis/functional_null.py` — deterministic binary-search agent simulator for residualization (App. A.7).
- [ ] `analysis/run_all.py` — single entry point: takes a folder of CSVs + condition labels → outputs a results table matching Figs. 2–3 of the paper.

### Phase 2 — Pipeline validation (~1 hour active)
- [ ] Synthetic-gate sanity check: AND/OR/XOR truth-table distributions. Verify PID returns known textbook values (e.g., XOR → pure synergy).
- [ ] Functional null sanity check: deterministic-binary-search agents should produce near-zero synergy.
- [ ] Apply pipeline to the 6 `sampled_data/*.csv` examples in Riedl's repo. Outputs should be plausible (low-N so noisy, but no NaNs/crashes).

### Phase 3 — Run the minimal experiment (~few hours wall-clock, ~$3)
- [ ] Configure `run_experiment_multi_model.py` for: `gpt-4o-mini`, N=10, T=1.0, 30 runs × 3 conditions.
- [ ] Launch. Monitor for errors / cost.
- [ ] Collect output CSVs into a structured folder.

### Phase 4 — Apply pipeline and compare (~1 hour active)
- [ ] Run `analysis/run_all.py` on the 90 CSVs.
- [ ] Compare results to Riedl Figs. 2–3 and Table A2 row for `gpt-4o-mini`-class models.
- [ ] Write a short replication report (`replication_report.md`): which findings reproduce, which don't, hypotheses for discrepancies.

### Phase 5 — Decision gate (Sahar reviews)
- ✅ Strong replication → start `thesis_plan.md` for GovSim extension.
- ⚠️ Partial → investigate; consider stepping up to larger model.
- ❌ No replication → re-evaluate; negative result is itself analysis-worthy.

---

## 7. Open questions before Phase 0

1. **API key**: Do you have an OpenAI key with ~$5 of budget for `gpt-4o-mini`?
2. **Python version**: 3.10+ available on your machine?
3. **Advisor**: Worth a quick check-in with your advisor that "spend ~1 week mini-replicating Riedl before extending to GovSim" is endorsed.

---

## 8. Explicitly out of scope for this phase

- GovSim setup, integration, or experiments.
- The MACS multi-vector index.
- Behavioral (NLP) or structural (graph) analyses.
- Predictive-collapse modeling.
- Causal mediation, IPW reweighting, persona-similarity analysis (Appendix A.11/A.14).

These move to a future `thesis_plan.md` drafted after Phase 5.
