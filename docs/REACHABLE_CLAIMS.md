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

That control is **panel-specific**, which was not appreciated when this section was written. On the
deep Moroccan panel the identical contrast reaches only |t| = 2.14, at 0.86 of its own detection
threshold. The instrument works *here*; section 6 draws the protocol consequence.

The second survivor must be read against a **non**-survivor:

> log vol ratio: `min_variance_lw` vs `regime_conditional` — observed −0.0249, |t| = **1.43**, does
> not survive.

So `regime_conditional` runs materially lower volatility than Markowitz — and is **not
distinguishable from plain `min_variance_lw`** on the same measure. The defensible claim is that
the regime layer attains a minimum-variance risk profile, which is the profile of its own bear
sub-strategy. The evidence does **not** show it improves on running `min_variance_lw` directly.

Stated plainly, so it cannot be quoted more strongly than it deserves:

- **Established, on `full_2021`:** `regime_conditional` is lower-volatility than `max_sharpe`,
  after correcting across 30 hypotheses. The scope restriction is load-bearing — section 5 shows
  this does not replicate on the deep panel, where the by-construction positive control below is
  itself undetectable. That panel cannot corroborate the claim; neither can it refute it.
- **Not established:** that it beats `min_variance_lw` on volatility, on Sharpe, on CEQ, or on
  drawdown.
- **Not established:** any Sharpe outperformance whatsoever — consistent with every earlier phase,
  and now with a measured reason rather than an inference from overlapping intervals.

The honest summary is that the risk reduction is real and attributable to the minimum-variance
branch, not to the regime switch. Whether the switch earns its cost is section 4's question, and
the answer is not yet in.

---

## 4. Cost, and the one decision this evidence settles

The cost-drag family sits at α = 0.10 and not 0.05, which is exactly the resolution this sample
affords. `regime_conditional` gives back 0.0733 Sharpe to transaction costs against `max_sharpe`'s
0.0125 — about 60.6 bps/yr — while its gross advantage (+0.0552, |t| = 0.10 against the MDE) is
not establishable at all. The one thing the evidence can say about the regime switch versus
Markowitz is that it costs more to run.

**Put beside section 3, that is a decision.** `regime_conditional` attains a minimum-variance risk
profile; is not distinguishable from `min_variance_lw` on either panel (−0.0249 on `full_2021`,
−0.0274 on `deep_morocco`, neither detectable); has no establishable gross advantage; and costs
about 60.6 bps/yr more to run. It is a regime-switching layer being paid to reproduce its own bear
branch.

> **Recommended:** run `min_variance_lw`. It is simpler, cheaper to trade, and carries the same
> measured risk profile.

The caveat that keeps this honest: "not distinguishable" is not "identical", and three undetectable
contrasts are not proof of equivalence. But they point the same way on two universes, and the
burden sits with the more complex model.

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

What differs is the **strategy set** — or so this note argued until it was tested. `full_2021`'s
family contains `min_variance_lw`, and both of section 2's survivors are volatility contrasts
against it. The deep panel's family contained no variance minimiser at all.

**That explanation has now been run, and it is wrong.** `min_variance_lw` was added to
`deep_morocco_starvation.py`; the other five strategies reproduce bit-identically. The family grows
to 15 pairs × 4 estimands = **60 hypotheses**, which *raises* the critical value from 4.296 to
**4.664**. **Zero of sixty survive.**

The decisive row is the positive control itself:

| log vol ratio, `max_sharpe` vs `min_variance_lw` | observed | \|t\| | rank | obs/MDE |
|---|---:|---:|---:|---:|
| `full_2021`, n = 455 | +0.1813 | **6.10** | 1st of 30 | — |
| `deep_morocco`, n = 1,638 | +0.0678 | **2.14** | 12th of 60 | **0.86** |

A contrast that is true **by construction** — section 3 leans on it as proof the instrument works —
is not detectable on the deep panel even as a single pre-specified test, before any correction. So
the missing-minimiser story cannot be what was happening: the minimiser is now in the family, and
nothing changed.

### What actually differs, and two further explanations that also failed

The obvious next inference was that the constraint set collapses the strategies together: a
long-only book of 12 names capped at 20% has a floor of five positions, so the cap rather than the
objective would be picking the portfolio. **Measured, and also wrong.**
`experiments/deep_morocco_weight_concentration.py` reads the weights the backtest already produces:

| strategy | effective N | names at cap | weight at cap | realised vol |
|---|---:|---:|---:|---:|
| `max_sharpe` | 6.67 | 1.59 | 31.9% | 0.1365 |
| `regime_conditional` | 7.95 | 1.37 | 27.4% | 0.1309 |
| `min_variance_lw` | 8.81 | 1.00 | 20.0% | 0.1274 |
| `equal_weight` | 12.00 | 0.00 | 0.0% | 0.1323 |

Effective N spans a **1.80x** range. These are genuinely different portfolios and the cap is not
collapsing them. The real finding is the one that table makes unavoidable: **weight dispersion does
not become risk dispersion here.** `min_variance_lw` holds 1.32x the effective names of
`max_sharpe`, and that buys a volatility ratio of 1.07.

A fourth attempt tried to make that predictive — estimate the achievable separation from the
correlation structure and the weights, ahead of any test. `experiments/risk_separation_bound.py`
records its failure: an equicorrelation model predicts +0.0854 on `full_2021` against +0.1565
observed, and −0.0008 on `etf_2017` against +0.0996. Given per-asset volatilities so it can see the
low-volatility tilt, it moves to +0.1177 and +0.0268 — still wrong by a factor of four on the
second. The failure is structural, not a missing term: equicorrelation imposes one correlation on
every pair, and a variance minimiser's edge lives in the deviations from that.

`etf_2017` is the row that should be read twice. There `min_variance_lw` and `max_sharpe` hold the
**same** effective N — 4.00 against 4.02 — while separating by +0.0996. Separation with no
diversification difference whatsoever, coming entirely from *which* assets are held. Whatever
governs how much risk separation a universe affords, it is not captured by any simple structural
summary computed here, and four explanations have now been tried and discarded.

### What it does to section 3

It does **not** overturn section 3, and the reason matters. The deep panel cannot detect a
by-construction effect, so it has no power to refute anything — this is absence of evidence with a
measured explanation, not evidence of absence. The correct edit is a scope restriction, made in
section 3 above, not a retraction.

What does replicate is section 3's other reading. `min_variance_lw` vs `regime_conditional` is
−0.0274 on the deep panel against −0.0249 on `full_2021`, undetectable on both: the regime layer
attains a minimum-variance profile on either panel without being distinguishable from running the
minimiser directly.

Across both panels, at every correction level, **no Sharpe, CEQ or drawdown comparison has ever
survived** — 30 hypotheses on one panel and 60 on the other, on 455 and 1,638 days, on two
universes.

---

## 6. The protocol this forced — run the positive control first

Sections 3 and 5 record the same mistake made twice. Section 3 read a by-construction contrast as
proof the instrument works, without noticing that claim was panel-specific. Section 5 then
explained a null with a mechanism it had inferred rather than measured, and the explanation was
wrong; so were the two that replaced it. In each case the error was invisible until something
already known to be true was checked against the design.

**The rule, stated so it can be applied without rereading this note:**

> A design that cannot detect a tautology cannot be trusted to report a null. Every family run
> includes a contrast that is true by construction, and if that contrast does not clear detection,
> the run reports "this panel cannot resolve this class of question" — not a negative finding.

The natural control for portfolio comparisons is `max_sharpe` vs `min_variance_lw` on log
volatility: a variance minimiser produces less variance than a Sharpe maximiser by construction, on
any panel, with no assumption about skill. Its cost is one extra hypothesis in the family, against
which it would have caught section 5's misreading before that section was written.

The measured values, which are the reason this is a rule and not a precaution:

| Panel | control observed | control \|t\| | obs/MDE | verdict |
|---|---:|---:|---:|:--|
| `full_2021`, n = 455 | +0.1813 | 6.10 | — | instrument works |
| `deep_morocco`, n = 1,638 | +0.0678 | 2.14 | 0.86 | **cannot resolve** |

A null from the second row is uninformative about the models, and 3.6x the data does not fix it —
the deep panel's usefulness is for the information-coefficient question and for PBO, not for
portfolio comparison. Match the panel to the question.

The corollary for anything added later, **including the Tier 2 items in
`docs/LITERATURE_IMPROVEMENTS.md`**: a new strategy rung evaluated on a panel whose control fails
will produce an undetectable contrast whatever its merits, and that is a fact about the panel, not
about the rung. Run the control before building the rung.

---

## Reproduce

```bash
.venv/Scripts/python experiments/deep_morocco_estimand_family.py          # sections 5, 6
.venv/Scripts/python experiments/deep_morocco_weight_concentration.py     # section 5 weights
.venv/Scripts/python experiments/risk_separation_bound.py                 # section 5 refutations
```

The family driver takes optional input and output paths, and was validated against this note's own
published section 5 numbers before being trusted on the extension: pointed at the pre-`min_variance_lw`
equity curve it returns the same n, the same 40 hypotheses, zero survivors, critical |t| 4.296 and
strongest |t| 2.75. Only α = 0.10's critical value differs, 3.847 against the 3.845 recorded above,
which is bootstrap-quantile noise and changes no verdict.

All three read committed Gold artifacts; `risk_separation_bound.py` refits nothing at all and the
other two re-run only the cheap non-ML strategies. `src/inference.py` is a dependency of no DVC
stage, so none of this costs recompute.
