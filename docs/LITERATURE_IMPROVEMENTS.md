# Literature review — where the published research says this project can still improve

**Status:** research-facing, forward-looking. Nothing here is a result; everything is a
*candidate* with a citation and a predicted cost.
**Date:** 2026-09-14

This document exists because the repository has reached a defensible negative result and the
natural next question is "what would a stronger version of this study look like?". It answers
that from the literature rather than from intuition.

## How to read this

The repository already builds on a specific set of papers. Chapter 2's état de l'art cites
Markowitz (1952), DeMiguel et al. (2009), Ledoit–Wolf (2004), Engle (2002), Jagannathan–Ma
(2003) and López de Prado (2018) + Bailey–López de Prado (2014); the source additionally
leans on Chopra–Ziemba (1993) (`ml_signals.apply_mu_transform`), White (2000) and Hansen
(2005) (`metrics.reality_check`), Politis–Romano (`metrics.paired_block_bootstrap`),
Hamilton (1989) (`regime`) and Dimson (1979) / Scholes–Williams (1977)
(`experiments/nonsync_covariance.py`).

**None of those are repeated below.** Every entry is something the repository does *not*
currently use. Each is graded on the axis that matters here — not "is it fashionable" but
**"does it change a claim this project is allowed to make, and at what cost"**.

Entries are ordered by value-per-unit-effort, not by prestige. Tier 1 changes what the
*existing* numbers are allowed to say without re-running a single backtest. Tier 4 is a list
of things the literature suggests **not** to bother with, which is as useful as the rest.

---

# Tier 1 — Inference upgrades: new claims from data already on disk

These add no strategy, no model and no re-run of `run_backtest`. They re-analyse stored
return series. They are the highest-leverage items in this document because the project's
central bottleneck is not modelling — it is **statistical power and what the tests permit
saying** (documented as Limit #5 in `docs/EVALUATION_LIMITS.md`).

## 1.1 Ledoit & Wolf (2008) — studentized bootstrap test for a *difference* of Sharpe ratios

**Reference.** Ledoit, O. and Wolf, M. (2008), "Robust performance hypothesis testing with the
Sharpe ratio", *Journal of Empirical Finance* 15(5), 850–859.

**What the repository does now.** `metrics.paired_block_bootstrap` draws a moving-block
bootstrap (block length 21) of the ΔSharpe between two strategies and reports a percentile
interval. It is paired and calendar-aligned, which is correct and non-trivial.

**What the paper adds.** A plain percentile bootstrap of ΔSharpe is not pivotal: its coverage
degrades exactly under the two conditions this data has — heavy tails and serial dependence.
Ledoit–Wolf derive the HAC standard error of the Sharpe difference via the delta method over
the four moments (μ₁, μ₂, γ₁, γ₂), then **studentize** the time-series (circular block)
bootstrap with it. The studentized statistic is asymptotically pivotal, so the bootstrap
achieves a higher order of accuracy. In their simulations this is the difference between a
test with nominal size and one that is visibly mis-sized at the sample lengths used here.

**Why it matters specifically here.** The frozen test on `full_2021` is ~460 days, and the
project's headline intervals — e.g. `global_2004` ΔSharpe −0.0862, 90% CI [−0.2133, +0.0414] —
are wide enough that *calibration is the whole ballgame*. A better-calibrated, higher-power
test on identical data is the cheapest possible improvement to the evidence chain.

**Second, under-appreciated payoff.** The same delta-method variance formula gives a
closed-form **minimum detectable ΔSharpe** for a given n, σ and target power. Limit #5 states
that a pre-registered MDE "would have framed the negative result as *designed* rather than as
a disappointment". This paper supplies exactly that calculation, and it can be computed
retrospectively for every comparison already published.

**Cost.** Low. A new function beside `paired_block_bootstrap`, reusing its existing circular
block-index helper (`metrics._circular_block_indices`). The HAC standard error needs a kernel
estimator, which the repository does not currently have in `src/` — `statsmodels` (already a
direct dependency) supplies one, or the prewhitened QS kernel with automatic bandwidth that
the paper recommends is a contained addition. No pipeline re-run.

## 1.2 Hansen, Lunde & Nason (2011) — the Model Confidence Set

**Reference.** Hansen, P. R., Lunde, A. and Nason, J. M. (2011), "The Model Confidence Set",
*Econometrica* 79(2), 453–497.

**The problem it solves.** Every headline in the README is of the form "**not established**".
White's Reality Check and Hansen's SPA are both *one-sided tests against a single
pre-specified benchmark*: they can reject "nothing beats the benchmark", and when they fail to
reject they return silence. After 240 trials the project can report p = 0.9045 (White RC) and
p = 0.8656 (SPA) and nothing more.

**What MCS gives instead.** A set — analogous to a confidence interval, but over *models* —
constructed to contain the best model with a stated confidence level. It is the natural
positive statement to make from a negative result: rather than "no challenger is established
as superior", the project could report *"at 90% confidence the set of strategies
indistinguishable from the best is {…}"*. Two outcomes are both publishable and both
interesting:

- the MCS is large and contains `equal_weight` → a **quantified** restatement of DeMiguel et
  al. (2009) on this project's own data, which is a much stronger finding than a failed test;
- the MCS *excludes* the ML challengers → a genuine positive result in the negative
  direction, i.e. evidence that the ML layer is inferior rather than merely unproven.

Note the property the authors emphasise and that suits this project's ethos: the MCS is honest
about power by construction — uninformative data yield a large set, informative data a small
one. It cannot manufacture a winner.

**Cost.** Low–moderate. The elimination procedure reuses the same bootstrap distribution
`metrics.reality_check` already constructs. Reference implementations exist (e.g. the R
`MCS` package, arXiv:1410.8504) to validate a Python port against.

**Recommendation: this is the single highest-value item in the document.** It converts the
project's central non-result into a statement with content, using data already committed.

## 1.3 Romano & Wolf (2005) — StepM, a strictly more powerful Reality Check

**Reference.** Romano, J. P. and Wolf, M. (2005), "Stepwise Multiple Testing as Formalized
Data Snooping", *Econometrica* 73(4), 1237–1282.

**What it changes.** White's RC is a *single-step* procedure returning one global p-value for
"is the best of the 240 better than the benchmark". StepM controls the same familywise error
rate but proceeds in stages: reject the clearly significant strategies, remove them, re-derive
the critical value on the remainder, repeat. Because it captures the joint dependence
structure of the test statistics and re-tightens after each round, it is **uniformly more
powerful than the RC at the same FWER** and, critically, it returns a *per-strategy* decision
instead of one number.

**Why it matters here.** The 240-trial search currently collapses to a single p-value. StepM
would name which individual challengers, if any, survive correction — a finer-grained and more
defensible statement, at no cost in rigour. Hsu, Hsu & Kuan's Step-SPA combines the stepwise
idea with Hansen's SPA if the project prefers to stay in the SPA family.

**Cost.** Low. Same bootstrap resamples as the existing RC; the change is the iteration.

## 1.4 Bailey, Borwein, López de Prado & Zhu — PBO via combinatorially symmetric CV

**Reference.** Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2017), "The
Probability of Backtest Overfitting", *Journal of Computational Finance* 20(4), 39–69.

**Relation to what exists.** `metrics.deflated_sharpe_ratio` implements the *other* half of
this research programme. DSR deflates an observed Sharpe given a trial count and a variance of
trial Sharpes — it is parametric and needs both inputs to be right. CSCV is model-free,
non-parametric and symmetric: it splits the performance matrix into complementary
in-sample/out-of-sample combinations and estimates the probability that the configuration
selected in-sample underperforms the median out-of-sample.

**Why both.** They fail differently, which is the point of having two. CSCV also produces
diagnostics DSR cannot: performance degradation (the OOS-vs-IS regression slope) and
first/second-order stochastic dominance of the selected configuration. For a project whose
thesis is "tell a seductive backtest maximum apart from a real edge", PBO is close to a
purpose-built instrument, and it attaches naturally to
`model_selection.select_portfolio_levers`.

One caution worth recording if adopted: PBO → 1 as the number of configurations grows
*regardless of whether any configuration has genuine skill*. It is a statement about the
selection procedure, not solely about the strategies, and must be reported as such.

**Cost.** Moderate. Needs the full trial × time performance matrix retained, not just the
per-trial summary. Worth checking whether the 240-trial artifacts already store enough.

## 1.5 DeMiguel et al. (2009) — use the *rest* of the paper

**Not a new reference — an under-used one.** Chapter 2 cites this work as the project's
"haie d'honnêteté", but takes only the Sharpe comparison from it. The paper evaluates every
strategy on **three** criteria: Sharpe ratio, **certainty-equivalent return**, and **turnover**.

**Why this is pointed.** Limit #2 documents that the Sharpe ranking on `full_2021` is *not
invariant* to the risk-free rate — at rf = 3.00%, `equal_weight` overtakes
`regime_conditional`. A ranking that flips under a nuisance parameter is fragile, and the
cited paper's own answer to that fragility is to report a criterion triple rather than one
number. CEQ under a stated risk aversion is not rf-rank-invariant either, but it is a
*different* fragility, and turnover is invariant. Reporting all three is both more robust and
more faithful to the benchmark paper the project has chosen to be judged against.

**Cost.** Very low. Both quantities are already computable from stored weights and returns;
`metrics.summarize` is the natural home.

---

# Tier 2 — Estimator upgrades that close a *documented* defect

Each of these maps onto a specific numbered limitation in `docs/EVALUATION_LIMITS.md`.

## 2.1 Statistical jump models — for the HMM's known instability (Limit #3, and the warm-up fallbacks)

**References.** Nystrup, P., Kolm, P. N. and Lindström, E. (2020/2021) on jump models for
regime identification; Shu, Y., Yu, C. and Mulvey, J. M. (2024), "Downside Risk Reduction
Using Regime-Switching Signals: A Statistical Jump Model Approach", arXiv:2402.05272.

**The diagnosis in the literature matches this project's symptoms exactly.** The
regime-switching literature reports that HMMs, under *high regime persistence, low
signal-to-noise ratio and limited data* — a precise description of a 2-state HMM on a
~5-year Moroccan/ETF panel — produce state sequences that lack persistence and stability.
The repository's own artifacts show the consequence: six fallbacks in `regime_conditional`'s
warm-up (6.2% of that comparator's out-of-sample days, `docs/MODEL_INTEGRITY.md`), and a
refit on more data moving posteriors by ~0.7 in probability units with some states merely
relabelled (`docs/EVALUATION_LIMITS.md` §3).

**What a jump model changes.** A statistical jump model fits states by minimising a clustering
loss **plus an explicit jump penalty λ charged at every state transition**. Persistence stops
being an emergent property of an estimated transition matrix and becomes a tunable
regularisation parameter. Shu–Yu–Mulvey select λ by time-series cross-validation *on strategy
performance* — which slots into the project's existing purged-CV lever selection rather than
requiring new machinery — and report improvements over HMM-guided and buy-and-hold strategies
on volatility, maximum drawdown and Sharpe across US, German and Japanese indices 1990–2023,
with transaction costs and trading delays included.

**Fit with this codebase.** Good. `RegimeConditionalStrategy` consumes a state/posterior
series; the estimator behind it is swappable. The jump penalty also attacks turnover directly,
which is this project's demonstrated weak point.

**Honest caveat.** Switching estimators does **not** by itself resolve Limit #3. The
train/serve mismatch there is about *smoothed vs filtered* inference, and a jump model fitted
over a full window has the same in-window lookahead unless it is run online over expanding
prefixes. Fix the causality first, then compare estimators — otherwise the comparison is
confounded by exactly the defect the project already documented.

## 2.2 Temporal aggregation of GARCH forecasts — for Limit #4

**Reference.** Drost, F. C. and Nijman, T. E. (1993), "Temporal Aggregation of GARCH
Processes", *Econometrica* 61(4), 909–927.

**Why cite it.** Limit #4 already diagnoses the horizon mismatch correctly — a 1-day
conditional covariance scaled by 252 is fed to an optimizer holding for ~21 days, and GARCH
mean-reverts toward its unconditional level over that horizon. The proposed fix (average the
1…21-step forecast variances) is right. This reference matters because it turns that fix from
an ad-hoc repair into the **standard, citable treatment**: the temporal-aggregation properties
of GARCH are a known result, and the correct multi-period object is derived rather than
guessed. If the DCC rung is revisited for the report, the fix should be presented with this
citation, not as a bug patch.

The same correction applies, more mildly, to `MinVarianceEWMA`, as the limits document notes.

## 2.3 Nonlinear shrinkage — Ledoit & Wolf (2020, 2022) and DCC-NL (Engle, Ledoit & Wolf, 2019)

**References.** Ledoit, O. and Wolf, M. (2020), "Analytical nonlinear shrinkage of
large-dimensional covariance matrices", *Annals of Statistics* 48(5), 3043–3065; Ledoit, O. and
Wolf, M. (2022), quadratic-inverse shrinkage (QIS); Engle, R. F., Ledoit, O. and Wolf, M.
(2019), "Large Dynamic Covariance Matrices", *Journal of Business & Economic Statistics*
37(2), 363–375.

**What they are.** Linear Ledoit–Wolf (2004) — what `MinVarianceLW` uses — pulls *all* sample
eigenvalues toward the grand mean with a single intensity. Nonlinear shrinkage shrinks **each
eigenvalue individually** toward its population counterpart, which matters disproportionately
for the *inverse* covariance, and the inverse is precisely what a mean-variance optimizer
consumes. The 2020 paper replaces the numerical QuEST inversion with an analytical formula via
the Hilbert transform of the sample spectral density — roughly 1000× faster at essentially
equal accuracy. DCC-NL marries DCC dynamics to nonlinear shrinkage of the correlation
targeting matrix, estimated by composite likelihood.

**Honest assessment for *this* project — read before implementing.** These methods are
motivated by the high-dimensional regime where the concentration ratio p/n is non-negligible.
This project's universes are N = 4–10 assets against hundreds of training days, so
p/n ≈ 0.02–0.04. **The literature's own theory predicts the gain here will be small**, because
the sample eigenvalue dispersion that nonlinear shrinkage corrects is itself small at this
concentration. DCC-NL's composite-likelihood machinery is designed for large N and is simply
unnecessary at N = 9.

This is included not as a recommendation to implement, but because it is the obvious next rung
on the covariance ladder and someone will propose it. The defensible position is: **it is a
cheap rung and a legitimate pre-registered null** — adding it and reporting "no improvement, as
the concentration ratio predicts" is a better contribution than adding it and hoping. The
project's own Jagannathan–Ma result already suggests the 25% cap is doing the regularisation
work that a fancier estimator would compete for.

## 2.4 Hayashi & Yoshida (2005) — asynchronous covariance without alignment

**Reference.** Hayashi, T. and Yoshida, N. (2005), "On covariance estimation of
non-synchronously observed diffusion processes", *Bernoulli* 11(2), 359–379.

**Relation to Limit #1.** `experiments/nonsync_covariance.py` attacks the
Casablanca/New-York calendar distortion with Dimson (1979) lead-lag aggregation and weekly
re-sampling — both reasonable, neither established as better (weekly_lw p = 0.304,
dimson_lw p = 0.442). The Hayashi–Yoshida estimator is the third branch of that literature:
rather than aligning observations onto a common grid (which is what forward-filling BVC prices
across Moroccan holidays does, and what creates the distortion), it sums products of
*overlapping* return intervals directly and is consistent without any synchronisation.

**Whether it applies.** Partially, and this should be checked before investing. HY was designed
for tick-level asynchronicity within a day; the project's problem is a *closing-time offset*
plus genuine non-trading days (BVC zero-return days run 13–21%). The zero-return days are
closer to an illiquidity/stale-price problem than an asynchronicity problem, and HY does not
fix stale prices. It is the right thing to read before concluding Limit #1 has been exhausted;
it is not obviously the right thing to implement.

## 2.5 Rockafellar & Uryasev (2000) — mean-CVaR as a genuinely different objective

**Reference.** Rockafellar, R. T. and Uryasev, S. (2000), "Optimization of conditional
value-at-risk", *Journal of Risk* 2(3), 21–41.

**The gap.** Every optimizer in `strategies.py` targets variance or the Sharpe ratio.
`metrics` *reports* maximum drawdown and the Calmar ratio but nothing *optimizes* for downside.
CVaR is the canonical fix and its key property is practical: with a scenario-based
formulation, mean-CVaR minimisation is a **linear program**, so it is cheaper and far more
reliable than the SLSQP solves currently used — no convergence fallbacks, no equal-weight
degradations of the kind `_optimize_weights` has to instrument.

**Why it suits this project.** It is a new rung that is cheap, convex, well-understood, and
changes the *question* rather than re-estimating the same inputs — which is more informative
than another covariance estimator. It also gives the drawdown numbers already in the report an
optimizer that was actually trying to achieve them.

**Cost.** Low–moderate. `scipy.optimize.linprog` is enough and `scipy` is already a direct
dependency, so this needs no new package (`cvxpy` would be more expressive but is not
currently a dependency). The long-only / 25%-cap constraints carry over unchanged.

---

# Tier 3 — Structural changes: where the genuine upside is

These are research-grade and would not be retrofitted into the current release. They are the
answer to "what would the next version of this study do differently".

## 3.1 Decision-focused learning — the project's Phase 4B result is this literature's canonical motivating example

**References.** Elmachtoub, A. N. and Grigas, P. (2022), "Smart 'Predict, then Optimize'",
*Management Science* 68(1), 9–26; Amos, B. and Kolter, J. Z. (2017), "OptNet", ICML; Agrawal,
A. et al. (2019), "Differentiable Convex Optimization Layers", NeurIPS; Butler, A. and Kwon,
R. H. (2023), integrating prediction in mean-variance optimization via implicit differentiation
of the KKT system; Costa, G. and Iyengar, G. (2023), "Distributionally robust end-to-end
portfolio construction", *Quantitative Finance*; Lee, J., Jeon, H., Bae, H. and Lee, Y.
(2024/2025), "Return Prediction for Mean-Variance Portfolio Selection: How Decision-Focused
Learning Shapes Forecasting Models", arXiv:2409.09684.

**The argument, stated against this project's own numbers.** `RandomForestSignalStrategy` is
trained to minimise squared error on per-asset returns. The optimizer downstream does not care
about squared error; it cares about the *ranking of risk-adjusted marginal scores*, because
that is what determines weights. The two objectives are not merely different — the docstring of
`apply_mu_transform` already records how badly they diverge in practice: `rf_signal` posted the
**best gross Sharpe of any strategy on `full_2021` (1.240)** and lost 0.178 of it to 0.885
average turnover. A model that predicts well and allocates badly is the exact pathology
decision-focused learning exists to address.

Lee et al. make this precise and the result is directly useful: DFL **tilts the MSE-based
prediction errors by the inverse covariance matrix**, so that a forecast error is penalised in
proportion to how much it actually moves the portfolio, and inter-asset correlation enters the
loss rather than being ignored. They also document that DFL systematically overestimates
returns for included assets and underestimates excluded ones, and argue these biases are
features rather than flaws — worse point forecasts, better decisions.

**Framing that should go in the report regardless of whether this is implemented.** The
existing `mu_transform` modes are a *hand-designed approximation* of what DFL learns. `"rank"`
("trust the ordering, not the magnitudes") is a crude, fixed version of the ranking structure
Wang & Hasuike derive from the KKT conditions; `"shrink"` is a fixed-intensity version of a
tilt DFL estimates. Saying so situates the project's own engineering inside a named research
programme and costs nothing.

**The caveat, which is as important as the recommendation.** Wang, Y. and Hasuike, T. (2026),
"Decision-Induced Ranking Explains Prediction Inflation and Excessive Turnover in SPO-Based
Portfolio Optimization" (arXiv:2605.01176), show that SPO-style DFL **produces inflated return
signals and unstable reallocations** — i.e. it can *worsen* the very turnover pathology it is
being recruited to fix. Their KKT analysis explains why (decisions are a ranking over risk- and
cost-adjusted marginal scores, and the surrogate loss is free to inflate the scores without
changing the ranking), and they evaluate three stabilisers: clipping, min-max rescaling, and
partial portfolio adjustment. **Any DFL work here must be paired with turnover control from the
start**, or it will reproduce the Phase 4B failure in a more complicated form.

**Cost.** High. `cvxpylayers` would replace the SLSQP call inside a differentiable training
loop, which is a redesign of the fit path, not an added strategy class. This is a next-project
item.

## 3.2 Gârleanu & Pedersen (2013) — the turnover penalty is a myopic approximation of a solved problem

**References.** Gârleanu, N. and Pedersen, L. H. (2013), "Dynamic Trading with Predictable
Returns and Transaction Costs", *Journal of Finance* 68(6), 2309–2340; Boyd, S. et al. (2017),
"Multi-Period Trading via Convex Optimization", *Foundations and Trends in Optimization*
(implementation: `cvxportfolio`).

**What the repository does.** `_optimize_weights` adds `λ·Σ|w − w_prev|` to the objective —
a single-period penalty that prices the cost of *getting to* a portfolio. The docstring is
candid that λ is a judgement call rather than an estimated quantity.

**What the theory says.** Gârleanu–Pedersen solve this problem in closed form when returns are
predictable by signals with **different mean-reversion speeds**, and the solution is not the
myopic penalty. It is: *aim in front of the target, and trade partially toward the aim.* The
optimal portfolio is a weighted average of the existing portfolio and an **aim portfolio**,
where the aim is itself a weighted average of the current Markowitz portfolio and all expected
future ones — with weights determined by how fast each signal decays.

**Why this bites here.** The project's features have visibly different decay rates —
short-horizon momentum, EWMA volatility, an HMM regime posterior. A single-period penalty
treats a signal that will persist for six months and one that will have decayed in two weeks
identically, and therefore systematically over-trades the fast signal and under-trades the slow
one. This is a *structural* misallocation of trading, not a tuning error, and no amount of
cross-validating λ fixes it.

**Why it is a good fit for this project specifically.** It targets the demonstrated weak point
(turnover destroying gross edge), it is a closed-form solution rather than another estimator to
fit, it introduces no new overfitting surface, and `cvxportfolio` provides a reference
implementation to validate against. Of the Tier 3 items this has the best
theory-to-implementation-cost ratio.

## 3.3 Kelly, Malamud & Zhou (2024) — the complexity prior may be backwards

**References.** Kelly, B. T., Malamud, S. and Zhou, K. (2024), "The Virtue of Complexity in
Return Prediction", *Journal of Finance* 79(1), 459–503; and "The Virtue of Complexity
Everywhere" (SSRN 4166368).

**Why it is relevant to a project that found ML does not help.** The implicit reasoning
throughout this repository is the standard one: small sample, few assets, therefore keep models
simple, and the negative ML result is unsurprising. KMZ prove the opposite can hold. In the
"ridgeless" high-complexity regime — where the number of parameters **exceeds** the number of
observations — expected out-of-sample predictability can *increase* with complexity, and simple
models can severely understate true predictability. The empirical vehicle is random-feature
ridge regression: project a few predictors into many random nonlinear features, then shrink
heavily.

**How it would enter this project.** As a new challenger alongside RF and XGBoost, and
critically, **as a pre-registered directional prediction**: if the virtue of complexity holds
on this data, out-of-sample performance should *improve* with the number of random features
over a range where classical intuition says it must degrade. That is a falsifiable prediction
with a named source, which is exactly the shape of claim this project is built to evaluate.
Either outcome is reportable, and a negative here is considerably more interesting than another
tree-based null because it contradicts a *Journal of Finance* result rather than confirming
folklore.

**Caveats to state if adopted.** The original result concerns market *timing* / return
prediction rather than cross-sectional allocation; the empirical finding has been contested and
partly attributed to the shrinkage rather than the complexity; and the project's universes are
small. None of these are reasons not to test it — they are reasons to pre-register the claim
narrowly.

**Cost.** Low to implement (random features + ridge is a few dozen lines), moderate to evaluate
honestly, because it adds trials to the multiple-testing budget and the existing RC/SPA/DSR
apparatus must absorb them.

## 3.4 Black & Litterman (1992) — a better injection point for ML views

**Reference.** Black, F. and Litterman, R. (1992), "Global Portfolio Optimization",
*Financial Analysts Journal* 48(5), 28–43.

**The structural point.** `apply_mu_transform` blends a model prediction toward the **sample
mean** — the estimate with no model risk, as the docstring puts it. Black–Litterman blends it
toward a different and arguably better-motivated anchor: the **equilibrium-implied returns**
obtained by reverse-optimising observed market-cap weights. The blend weight is not a
hyperparameter but a stated view-confidence matrix Ω, and the posterior is Bayesian rather than
a convex combination. The well-documented consequence is materially more stable weights, hence
lower turnover — the project's binding constraint.

**Fit.** Clean: a fourth `mu_transform` mode, with the same interface. That is unusually cheap
for a structural change.

**Caveats.** It needs a defensible equilibrium prior. For the BVC equities, market
capitalisations are obtainable. For a 10-ETF multi-asset universe (`global_2004`), "the market
portfolio" is a modelling choice rather than an observable, and choosing it after seeing
results would be precisely the kind of post-hoc benchmark selection the project refuses
elsewhere — it would have to be pre-registered. Note also that Jagannathan–Ma logic applies:
BL is a shrinkage device, and the project has already shown the 25% cap is a powerful implicit
shrinkage, so the two will compete for the same gains.

---

# Tier 4 — What the literature suggests *not* to do

Recording these is deliberate. Both are things a reviewer may ask for, and having a
citation-backed reason to decline is worth more than a hedged attempt.

## 4.1 Hierarchical Risk Parity — likely a dead end at N = 4–10

**Reference.** López de Prado, M. (2016), "Building Diversified Portfolios that Outperform
Out of Sample", *Journal of Portfolio Management* 42(4), 59–69.

HRP is the obvious suggestion — the project already cites López de Prado (2018), so the author
is in scope, and HRP is the best-known ML-flavoured allocator. Two reasons to decline:

1. **Dimension.** HRP's mechanism is hierarchical clustering of the correlation matrix followed
   by recursive bisection. With 4–10 assets the dendrogram is nearly degenerate and recursive
   bisection reduces to something close to inverse-variance weighting with extra steps. The
   method's advantages are asymptotic in N and this project has none of that N.
2. **Evidence.** Recent out-of-sample comparisons find **1/N outperforming HRP across
   experimental setups**, and the broader empirical picture is mixed. Given that
   `equal_weight` is already this project's honesty hurdle and already competitive, HRP is
   unlikely to clear it, and a cleared trial still costs multiple-testing budget.

If added at all, it should be added as a **pre-registered expected null** with these two
reasons stated in advance — not as a hopeful candidate.

## 4.2 Deep reinforcement learning — incompatible with this evaluation design

Deep RL portfolio agents (Jiang et al. 2017 and successors; Moody & Saffell 2001 for the direct
performance-criterion ancestor already referenced in the codebase) are the most-published
recent direction and the least suitable here. They are sample-hungry in a setting with a
~460-day frozen test; they add a large hyperparameter surface to a multiple-testing budget that
already spans 240 trials; and their results are notoriously sensitive to seed and environment
specification, which is in direct tension with this repository's reproducibility guarantees. A
DRL result on this data could not survive the project's own gates, and building something that
cannot pass one's own gate is not a contribution.

---

# Suggested ordering

If the goal is the strongest possible *evidence chain* on existing data — which is what this
project is actually about — the order is:

| # | Item | Cost | What it buys |
|---|---|---|---|
| 1 | **MCS** (1.2) | Low–moderate | Converts "not established" into a positive, quantified claim |
| 2 | **Ledoit–Wolf Sharpe test + MDE** (1.1) | Low | Correctly-sized test; the pre-registered MDE Limit #5 asks for |
| 3 | **CEQ + turnover reporting** (1.5) | Very low | Fixes the rf-rank-fragility of Limit #2 the way the cited paper does |
| 4 | **Romano–Wolf StepM** (1.3) | Low | Per-strategy decisions, strictly more power than RC |
| 5 | **PBO / CSCV** (1.4) | Moderate | A second, non-parametric overfitting check beside DSR |
| 6 | **Mean-CVaR rung** (2.5) | Low–moderate | A different question, not another estimator; LP-stable |
| 7 | **Filtered posteriors, then jump model** (2.1) | Moderate | Closes Limit #3, then attacks HMM instability at its root |
| 8 | **GARCH horizon aggregation** (2.2) | Moderate | Closes Limit #4 with a citation rather than a patch |

Items 1–5 require **no new strategy and no backtest re-run**. They are re-analysis of committed
artifacts, and between them they would change several headline sentences in the README from
"not established" to something with content.

Tier 3 is next-project scope. If exactly one structural item were chosen, **Gârleanu–Pedersen
(3.2)** has the best ratio of theoretical grounding to implementation cost, and it targets the
failure mode this project has already measured.

---

## References

Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S. and Kolter, J. Z. (2019).
Differentiable convex optimization layers. *NeurIPS*.

Amos, B. and Kolter, J. Z. (2017). OptNet: Differentiable optimization as a layer in neural
networks. *ICML*.

Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2017). The probability of
backtest overfitting. *Journal of Computational Finance* 20(4), 39–69.

Black, F. and Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*
48(5), 28–43.

Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P. and Speth, J. (2017).
Multi-period trading via convex optimization. *Foundations and Trends in Optimization* 3(1).

Butler, A. and Kwon, R. H. (2023). Integrating prediction in mean-variance portfolio
optimization. *Quantitative Finance*.

Costa, G. and Iyengar, G. (2023). Distributionally robust end-to-end portfolio construction.
*Quantitative Finance*.

Drost, F. C. and Nijman, T. E. (1993). Temporal aggregation of GARCH processes. *Econometrica*
61(4), 909–927.

Elmachtoub, A. N. and Grigas, P. (2022). Smart "predict, then optimize". *Management Science*
68(1), 9–26.

Engle, R. F., Ledoit, O. and Wolf, M. (2019). Large dynamic covariance matrices. *Journal of
Business & Economic Statistics* 37(2), 363–375.

Gârleanu, N. and Pedersen, L. H. (2013). Dynamic trading with predictable returns and
transaction costs. *Journal of Finance* 68(6), 2309–2340.

Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set. *Econometrica*
79(2), 453–497.

Hayashi, T. and Yoshida, N. (2005). On covariance estimation of non-synchronously observed
diffusion processes. *Bernoulli* 11(2), 359–379.

Kelly, B. T., Malamud, S. and Zhou, K. (2024). The virtue of complexity in return prediction.
*Journal of Finance* 79(1), 459–503.

Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio.
*Journal of Empirical Finance* 15(5), 850–859.

Ledoit, O. and Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance
matrices. *Annals of Statistics* 48(5), 3043–3065.

Lee, J., Jeon, H., Bae, H. and Lee, Y. (2024). Return prediction for mean-variance portfolio
selection: how decision-focused learning shapes forecasting models. arXiv:2409.09684.

López de Prado, M. (2016). Building diversified portfolios that outperform out of sample.
*Journal of Portfolio Management* 42(4), 59–69.

Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal
of Risk* 2(3), 21–41.

Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping.
*Econometrica* 73(4), 1237–1282.

Shu, Y., Yu, C. and Mulvey, J. M. (2024). Downside risk reduction using regime-switching
signals: a statistical jump model approach. arXiv:2402.05272.

Wang, Y. and Hasuike, T. (2026). Decision-induced ranking explains prediction inflation and
excessive turnover in SPO-based portfolio optimization. arXiv:2605.01176.
