# Tier 2 — estimator upgrades, and what each one turned out to be worth

**Status:** examiner-facing. Every number is recomputed from committed Gold artifacts.
**Date:** 2026-09-15

`docs/LITERATURE_IMPROVEMENTS.md` grades the unused literature by whether it changes a claim the
project is allowed to make. Tier 1 — the five inference upgrades — is complete and its results
live in `docs/EVALUATION_LIMITS.md` §5–6 and `docs/REACHABLE_CLAIMS.md`. This note covers the
three Tier 2 items its suggested ordering places next: the mean-CVaR rung (2.5), filtered
posteriors and the jump model (2.1), and GARCH horizon aggregation (2.2).

**All three are implemented, tested and measured. None of them is wired into the DVC pipeline**,
and §5 explains why that is a decision rather than an omission.

---

## 0. The rule every item below was run under

`docs/REACHABLE_CLAIMS.md` §6 adopted a protocol after that note twice explained a null with a
mechanism it had inferred rather than measured:

> A design that cannot detect a tautology cannot be trusted to report a null.

Each item here therefore carries a by-construction control, checked **before** its result is read.
They turned out to be cheap and to catch real errors:

| Item | Control | Result |
|---|---|---|
| mean-CVaR | `min_cvar` must attain the lowest **in-sample** CVaR | 0.01145 vs 0.01178 / 0.01294 / 0.01348 ✓ |
| mean-CVaR | panel must resolve a risk contrast at all | 2.45× its detection threshold ✓ |
| filtered posteriors | filtered **must equal** smoothed at *t* = T | agrees to 2.3×10⁻¹³ ✓ |
| jump model | DP must match brute force over all Kᵀ sequences | 60 instances, 0 mismatches ✓ |
| GARCH horizon | closed form must match the explicit recursion | 300 draws, 0 mismatches ✓ |

---

## 1. Mean-CVaR (item 2.5) — works, and does not improve on minimum variance

Every optimizer in the project targeted variance or the Sharpe ratio, while `metrics` reported
drawdown and Calmar for strategies that were never trying to achieve them. `MinCVaR`
(Rockafellar & Uryasev 2000) closes that gap as a linear program, so it has a global optimum and
needs none of the perturbed-retry and equal-weight degradation machinery SLSQP requires.

On `full_2021`, 455 frozen-test days:

| Strategy | Sharpe | **CVaR₉₅** | max DD | vol | turnover |
|---|---:|---:|---:|---:|---:|
| `min_cvar` | 0.898 | 0.01544 | −0.082 | 0.1046 | 0.099 |
| `min_variance_lw` | 0.903 | **0.01501** | −0.083 | 0.1026 | 0.074 |
| `max_sharpe` | 0.985 | 0.01797 | −0.106 | 0.1230 | 0.144 |
| `equal_weight` | 0.858 | 0.01664 | −0.099 | 0.1124 | 0.053 |

**The CVaR minimiser loses to the variance minimiser on CVaR, out of sample, having beaten it in
sample.** A tail is estimated from the worst 5% of a training window — very few effective
observations — so optimising it directly does not generalise, while the covariance the variance
minimiser uses is estimated from all of them. The gap is not itself detectable (0.32 of its MDE),
so the claim is that CVaR optimisation **does not improve on** minimum variance here, not that it
is worse.

What the rung does establish is that it beats Markowitz: detectably on CVaR (1.28 of the MDE) and
on log volatility at |t| = 5.60, surviving correction across 50 hypotheses. That is the same
verdict every other rung reaches — now reached by an optimizer that was actually trying.

Adding the CVaR estimand raised the family from 30 hypotheses to 50 and the critical |t| to 3.878,
so the two CVaR contrasts at 3.33 and 3.19 do not survive. The correction is paid by the estimand
that motivated it, which is the right direction.

---

## 2. Filtered posteriors (item 2.1, first half) — and a correction to Limit #3

Item 2.1's caveat fixes the order of work: a jump model fitted over a full window carries the same
in-window lookahead as a smoothed HMM, so comparing them would confound the estimator change with
the causality defect. Fix causality first.

Doing so revealed that **`docs/EVALUATION_LIMITS.md` Limit #3 had measured the wrong quantity**.
Its bound — "max drift ~1×10⁻³, fewer than 10 of 141 dates move" — is correct for what it measures,
the *incremental* movement of already-computed rows when the window is lengthened. The train/serve
mismatch is the *total* gap between the smoothed value a row carries in training and the filtered
value it would carry at inference. Measured on one fit per panel, so no refit confound:

| Panel | n | total gap max / mean | rows > 1e-3 | near a switch | away | §3-style incremental |
|---|---:|---:|---:|---:|---:|---:|
| `full_2021` | 1,239 | **0.721** / 0.044 | 47.9% | 0.176 | 0.012 | 1.2×10⁻⁶ |
| `etf_2017` | 5,594 | **0.818** / 0.036 | 45.6% | 0.207 | 0.011 | 5.9×10⁻⁴ |
| `global_2004` | 5,405 | **0.818** / 0.036 | 40.2% | 0.219 | 0.009 | 5.0×10⁻⁴ |

Around 45% of training rows carry a regime probability differing from the servable one by more
than 10⁻³, and near a regime switch the average gap is ~0.18–0.22 — about **18×** the gap away from
one, with a maximum of 0.72–0.82. Those are rows where training says *bull* and live inference
would have said *bear*. Full detail in Limit #3, now corrected.

---

## 3. The jump model (item 2.1, second half) — persistence yes, stability no

Both estimators run causally: refit on expanding prefixes, keep only each prefix's last state.
`full_2021`, 46 monthly evaluation dates:

| Estimator | switches | **rewrites its own history** | agrees with HMM |
|---|---:|---:|---:|
| HMM, filtered | 16 | 21.7% | — |
| jump, λ = 1 | 14 | 17.4% | 93.5% |
| jump, λ = 2.5 | 12 | **15.2%** | 93.5% |
| jump, λ = 5 | 12 | 28.3% | 91.3% |
| jump, λ = 10 | 10 | 28.3% | 87.0% |
| jump, λ = 25 | 8 | 30.4% | 84.8% |

**Persistence holds and is monotone — but it is true by construction**, since λ is defined to
charge switches. That is a knob working, not a finding.

**Stability does not hold.** The rewrite rate is U-shaped: better than the HMM up to λ = 2.5,
clearly *worse* beyond it, at 30.4% against 21.7%. The mechanism is visible in the table — a
heavily penalised sequence has few switches, so each governs a long run of dates and moving one
boundary flips all of them. Penalisation **concentrates** the instability rather than removing it.

The honest reading is that neither estimator is stable: every configuration rewrites 15–30% of its
own history when refitted. The best case is a three-date improvement on 46 dates, well inside
sampling noise, and is not claimed as a result.

λ is swept and reported whole rather than selected. Shu–Yu–Mulvey choose it by cross-validation on
strategy performance; `docs/EVALUATION_LIMITS.md` §6 measures what selection costs on this data —
a degradation slope near −1, unchanged on 3.6× the panel.

---

## 4. GARCH horizon aggregation (item 2.2) — the largest real defect found

Limit #4 diagnosed two defects by inspection and left them unquantified. Both are now measured,
with the aggregation derived from Drost & Nijman (1993) rather than patched:

> per-period variance over an *H*-period hold = σ²∞ + (σ²₁ₛₜₑₚ − σ²∞) · **A(p, H)**,
> where **A(p, H) = (1 − pᴴ)/(H(1 − p))**

| Panel | persistence | A(p, 21) | off-by-one | horizon (mean) | **horizon (high-vol decile)** | max |
|---|---:|---:|---:|---:|---:|---:|
| `full_2021` | 0.939 | 0.672 | ×0.962 | ×1.041 | **×0.912** | 50.9% |
| `etf_2017` | 0.984 | 0.861 | ×0.958 | ×1.041 | **×0.953** | 36.4% |

The correction is **two-sided**, because aggregation pulls toward the unconditional level from both
directions. On an average day it is worth about +4% on volatility; **on the highest-volatility
decile the current one-day input overstates risk by ~9%**, and the worst single date by **51%**.
This is the "systematically over-weights the current volatility state" Limit #4 predicted, now
measured: too low when calm, too high right after a shock.

A methodological note worth more than the number. Evaluated at the **final observation only** —
the obvious way to run this — the combined effect is about 2%, near nothing. That is an artefact
of the chosen day sitting close to average. Sizing a state-dependent defect at a single state
understates it, and the distribution is the honest summary.

---

## 5. Why none of this is in the pipeline

`src/strategies.py` is a declared dependency of **16** DVC stages, `src/regime.py` of 12,
`src/ml_signals.py` of 8, `src/dcc_garch.py` of 4. DVC hashes each file whole, so adding code that
no stage calls still invalidates every dependent stage — three to five hours of recompute for
nothing. Commit `b9b45a3` set the rule when it moved four instruments out of `src/metrics.py`:

> the test is whether a DVC stage calls the code, not whether it is the same kind of thing.

So the four new modules — `strategies_research`, `regime_research`, `jump_model`, `garch_horizon`
— live outside the graph, and `grep -c <name> dvc.yaml` is **0** for each. That is an invariant to
preserve, not an accident.

**Two promotions are genuinely worth their rebuild, and both are decisions to take explicitly:**

1. **Filtered posteriors into `ml_signals.attach_regime_feature`.** The measured gap — 45% of
   training rows, 0.18–0.22 near switches — justifies it. It changes every F7 result.
2. **Both GARCH corrections into `dcc_garch`.** A 9% risk overstatement on the worst decile is
   material for a risk model. It changes every `dcc_garch` result.

Neither is taken here, because invalidating published numbers is not a side effect an instrument
should have.

---

## Reproduce

```bash
.venv/Scripts/python experiments/cvar_rung_evaluation.py          # §1
.venv/Scripts/python experiments/regime_filtered_posteriors.py    # §2
.venv/Scripts/python experiments/jump_model_vs_hmm.py             # §3
.venv/Scripts/python experiments/garch_horizon_quantified.py      # §4
.venv/Scripts/python -m pytest tests/test_min_cvar.py tests/test_regime_filtered.py \
    tests/test_jump_model.py tests/test_garch_horizon.py
```

---

## What Tier 2 leaves open

- **2.3 nonlinear shrinkage** and **2.4 Hayashi–Yoshida** are not attempted. The survey predicts
  2.3 gains little at *p/n* ≈ 0.03, and `docs/NONSYNC_COVARIANCE.md` already covers 2.4's problem.
- The **correlation channel** of the horizon correction. DCC's R_t mean-reverts over the same 21
  days; only the variance channel is implemented.
- **Tier 3** is next-project scope. If exactly one structural item were taken, the survey names
  Gârleanu–Pedersen (3.2), which targets the turnover failure this project has already measured.
