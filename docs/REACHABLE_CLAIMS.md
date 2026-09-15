# What this sample can establish, and what it cannot

**Status:** examiner-facing. Every number below is recomputed from committed Gold artifacts.
**Date:** 2026-09-15

`docs/EVALUATION_LIMITS.md` §5 establishes that the Sharpe difference is unreachable on this
data: on the frozen-length `full_2021` window the observed gap sits at 0.01–0.24 of what the
design could detect. That is a statement about **one estimand**, and it was being read as a
statement about the data.

This note asks the prior question instead. Before running a comparison, which estimands can this
sample resolve at all? Then, having asked that of several estimands across several strategy
pairs, it corrects for the fact that asking repeatedly is itself a search.

---

## 1. Which estimands are reachable

`inference.paired_estimand_mde` reports, for any paired statistic, the observed difference, its
block-bootstrap standard error, and the minimum detectable effect at 80% power (α = 0.10).
`regime_conditional` against `max_sharpe`, last 455 days of `full_2021`:

| Estimand | Observed | MDE (80%) | Observed / MDE | Reachable |
|---|---:|---:|---:|:--|
| Sharpe difference | −0.006 | 0.572 | 0.01 | no |
| CEQ difference | −0.016 | 0.066 | 0.25 | no |
| max drawdown difference | 0.017 | 0.044 | 0.39 | no |
| mean turnover difference | 0.267 | 0.223 | 1.20 | marginal |
| **log volatility ratio** | **−0.156** | **0.076** | **2.05** | **yes** |

Variance is estimated far more precisely than mean return, so a risk claim is reachable on the
sample where a return claim is not. Nothing about the models changes between those rows — only
the question being asked of them.

The turnover row carries a caveat: 21 monthly rebalances is 7 bootstrap blocks, so treat its
standard error as indicative. The same row computed with the default 21-day block returned a
detectability ratio of 2.6e13, because at `block_len >= n` every circular resample is a rotation
and the standard error collapses to zero. `paired_estimand_mde` now refuses that call.

---

## 2. Correcting for the choice of estimand

Selecting which estimand to report is a search, and the maximum of a search beats a benchmark by
chance more often than any single test does — the mechanism `docs/MULTIPLE_TESTING.md` corrects
for across configurations. `inference.family_maxt_correction` applies Westfall-Young maxT across
a family of estimands resampled on **shared** block draws, so the correction inherits the
family's real dependence instead of assuming independence.

The family is every pair of the four `full_2021` strategies × five estimands = **30 hypotheses**,
n = 455. Critical |t| rises from the uncorrected 1.96 to **3.214** (α = 0.10) and **3.733**
(α = 0.05).

**Two of thirty survive at α = 0.05. Both are volatility ratios.**

| Hypothesis | Observed | \|t\| |
|---|---:|---:|
| log vol ratio: `max_sharpe` vs `min_variance_lw` | +0.1813 | 6.10 |
| log vol ratio: `max_sharpe` vs `regime_conditional` | +0.1564 | 5.05 |

At α = 0.10 five cost-drag differences join them (|t| 3.31–3.70). They are real but **marginal**,
and they do not survive at 0.05. No Sharpe, CEQ or drawdown comparison survives at either level.

---

## 3. What this actually licences — read this before quoting section 2

The strongest survivor is a **positive control, not a finding**. That minimum-variance optimisation
produces lower variance than maximum-Sharpe optimisation is true by construction; recovering it at
|t| = 6.10 is evidence the instrument works, not evidence about the models.

The second survivor must be read against a **non**-survivor:

> log vol ratio: `min_variance_lw` vs `regime_conditional` — observed −0.0249, |t| = **1.43**, does
> not survive.

So `regime_conditional` runs materially lower volatility than Markowitz — and is **not
distinguishable from plain `min_variance_lw`** on the same measure. The defensible claim is that
the regime layer attains a minimum-variance risk profile, which is the profile of its own bear
sub-strategy. The evidence does **not** show it improves on running `min_variance_lw` directly.

Stated plainly, so it cannot be quoted more strongly than it deserves:

- **Established:** `regime_conditional` is lower-volatility than `max_sharpe`, after correcting
  across 30 hypotheses.
- **Not established:** that it beats `min_variance_lw` on volatility, on Sharpe, on CEQ, or on
  drawdown.
- **Not established:** any Sharpe outperformance whatsoever — consistent with every earlier phase,
  and now with a measured reason rather than an inference from overlapping intervals.

The honest summary is that the risk reduction is real and attributable to the minimum-variance
branch, not to the regime switch. Whether the switch earns its cost is section 4's question, and
the answer is not yet in.

---

## 4. Open

The cost-drag family sits at α = 0.10 and not 0.05, which is exactly the resolution this sample
affords. `regime_conditional` gives back 0.0733 Sharpe to transaction costs against `max_sharpe`'s
0.0125 — about 60.6 bps/yr — while its gross advantage (+0.0552, |t| = 0.10 against the MDE) is
not establishable at all. The one thing the evidence can say about the regime switch versus
Markowitz is that it costs more to run.

---

## 5. The same family on the 20-year panel — nothing survives

Section 4 named this the obvious next measurement. It has been run:
`data/gold/deep_morocco_equity.parquet`, frozen test 2017-09-01 → 2024-05-31, **n = 1,638**
(3.6x section 2's window). Five strategies, ten pairs, four estimands = **40 hypotheses**. Cost
drag is absent because the artifact stores net equity only.

**Zero of forty survive**, at α = 0.05 (critical |t| = 4.296) and at α = 0.10 (3.845). The
strongest is `log vol ratio: xgb_tuned vs equal_weight` at |t| = 2.75. Five would pass
uncorrected — which is what the correction is for.

### Why more data produced fewer findings

This looks backwards and is not. The effect shrank faster than the error bar:

| log vol ratio, `regime_conditional` vs `max_sharpe` | observed | SE | MDE | obs/MDE |
|---|---:|---:|---:|---:|
| `full_2021`, n = 455 | −0.1564 | 0.0312 | 0.0775 | **2.02** |
| `deep_morocco`, n = 1,638 | −0.0405 | 0.0246 | 0.0613 | **0.66** |

The extra data did what extra data does — the standard error fell from 0.0312 to 0.0246. But the
effect itself is **3.9x smaller** on the deep panel, so the ratio falls anyway. This is not a
power failure; it is a smaller thing to find.

Two natural explanations were tested and **both are wrong**, so neither should be repeated:

- *"Moroccan equities are too correlated for a variance minimiser to act."* Mean pairwise
  correlation is **0.176** on the deep panel against **0.161** on `full_2021` — essentially
  identical.
- *"There is no low-volatility asset to hide in."* The opposite. Asset volatility dispersion is
  **wider** on the deep panel (max/min = 2.00, lowest asset 35% below the mean) than on
  `full_2021` (1.61, 18%).

What does differ is the **strategy set**. `full_2021`'s family contains `min_variance_lw`, and
both of section 2's survivors are volatility contrasts against it — one directly, one through
`regime_conditional`, which is bit-identical to `min_variance_lw` on 63.4% of `full_2021` days.
The deep panel's family contains no variance minimiser at all: `rf_tuned`, `xgb_tuned`,
`regime_conditional`, `equal_weight`, `max_sharpe`. None of them optimises variance, and they duly
land within 0.1307–0.1421 annualised of each other.

This is stated as the surviving explanation, **not** a demonstrated one. Testing it means re-running
`deep_morocco_starvation.py` with `min_variance_lw` added to the comparison set, which is cheap and
has not been done.

### What it does to section 3

It strengthens the reading there rather than contradicting it. Section 3 concluded the risk result
is inherited from the minimum-variance branch rather than earned by the regime switch. Section 5 is
what that predicts: remove the variance minimiser from the family and put the regime switch on a
panel where it is not shadowing one, and the volatility advantage drops by a factor of four and
stops being detectable.

Across both panels, at every correction level, **no Sharpe, CEQ or drawdown comparison has ever
survived** — 70 hypotheses now, on 455 and 1,638 days, on two universes.

## Reproduce

```bash
.venv/Scripts/python -c "import inference"   # paired_estimand_mde, family_maxt_correction
```

Both read `data/gold/dashboard_equity.parquet` only; neither refits a model. `src/inference.py` is
a dependency of no DVC stage, so neither costs recompute.
