"""
inference.py — Search-aware inference procedures for the evaluation evidence.

These answer questions ABOUT a completed evaluation rather than measuring one
strategy: how large an effect the protocol could have detected, which
strategies the data cannot eliminate, and how much of a search's winning
margin is selection rather than skill. All of them consume return series that
are already frozen; none refits a model or selects a configuration.

WHY THIS IS NOT IN `metrics.py`, WHICH IS WHERE IT STARTED. `src/metrics.py`
is a declared dependency of TWELVE DVC stages, so DVC hashes the whole file
and any edit to it invalidates all twelve — roughly three to five hours of
recompute, dominated by `phase5_compare`. That price is correct for
`certainty_equivalent`, which `metrics.summarize` calls and which therefore
changes what those stages output. It is NOT correct for the procedures here:
no stage calls any of them, so they cannot change a single published number,
yet while they lived in `metrics.py` they invalidated the pipeline exactly as
if they had. Phase 1 paid that toll once. Keeping research instruments in
their own module means the next one is free.

The rule that follows: a function belongs in `metrics.py` when a DVC stage
calls it. A function that only ever reads frozen artifacts belongs here.

Addresses: P4 — overfitting and out-of-sample rigor. `metrics.py` corrects a
single Sharpe for the number of trials (the DSR); these characterise the
search and the sample as a whole.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

# TRADING_DAYS_PER_YEAR for the annualisation convention, and the block-index
# generator because `sharpe_difference_test`'s bootstrap must resample exactly
# the way `reality_check` does — a second implementation that drifted would
# make the two sets of p-values quietly incomparable.
from metrics import TRADING_DAYS_PER_YEAR, _circular_block_indices


def model_confidence_set(
    candidate_returns: Mapping[str, pd.Series],
    *,
    size: float = 0.10,
    block_len: int = 21,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Model Confidence Set of Hansen, Lunde & Nason (2011), Econometrica 79(2).

    Addresses: P4 — `reality_check` can only ever fail to reject. White's RC and
    Hansen's SPA are one-sided tests against a single pre-specified benchmark,
    so when the search turns up nothing they return silence: "no candidate is
    established as superior", and no more. The MCS asks the complementary
    question and returns a SET — the strategies not eliminated at confidence
    1 - `size` — which carries content whichever way it falls. A surviving set
    containing `equal_weight` quantifies DeMiguel et al. (2009) on this
    project's own data; a set that eliminates the ML challengers is evidence
    they are worse rather than merely unproven.

    WORDING (AGENTS.md section 5.2). Survival is not equivalence, and members of
    the returned set must never be called "statistically indistinguishable" —
    that phrase is banned project-wide precisely because failing to eliminate
    is not a demonstration of sameness. Say "retained by the MCS procedure at
    size alpha". `interpretation` below is phrased to comply and should be
    quoted rather than paraphrased.

    STATISTIC. The procedure consumes per-period LOSSES, taken here as negated
    net returns, so this is a mean-return criterion. It is NOT a Sharpe
    criterion: a Sharpe ratio is a ratio of moments with no per-period loss
    representation, so unlike `reality_check` there is no `statistic` argument.
    An MCS reported beside a Sharpe-based RC/SPA compares different quantities,
    and that must be stated wherever the two appear together.

    Implementation note: delegates to `arch.bootstrap.MCS` rather than
    hand-rolling the elimination, but only after verifying it. Its neighbour
    `arch.bootstrap.SPA` accepts a `studentize` flag it never applies — the
    flag only writes a metadata string — so the library is not taken on trust.
    MCS was checked to retain a genuinely best model, eliminate hopeless ones,
    retain everything under exchangeability, and shrink monotonically in `size`.

    Args:
        candidate_returns: Per-strategy simple net return series sharing one
            index. At least two — a confidence set over one model is vacuous.
        size: Test size alpha; the set has confidence 1 - alpha. Default 0.10,
            matching `phase5.bootstrap.alpha`.
        block_len: Circular block length, matching the project's other
            bootstraps (~one trading month).
        n_boot: Bootstrap replications.
        seed: Seeded for reproducibility, like every estimator here.

    Returns:
        dict with `included` / `excluded` (sorted name lists), `n_candidates`,
        `n_included`, `size`, `block_len`, `n_boot`, `seed`, `statistic` and
        `interpretation`.

    Raises:
        ValueError: on fewer than two candidates, misaligned indexes, NaN, or
            too little data for one block — the same input discipline as
            `reality_check`, and for the same reason: silent alignment would
            change which periods are being compared.
    """
    from arch.bootstrap import MCS as _ArchMCS

    names = sorted(candidate_returns)
    if len(names) < 2:
        raise ValueError(
            f"model_confidence_set needs at least two candidates; got {len(names)}."
        )

    reference = candidate_returns[names[0]].index
    for name in names:
        series = candidate_returns[name]
        if not series.index.equals(reference):
            raise ValueError(
                f"model_confidence_set requires identical date indexes; '{name}' "
                f"differs from '{names[0]}' — refusing to align silently."
            )
        if series.isna().any():
            raise ValueError(
                f"model_confidence_set requires NaN-free returns; '{name}' has NaN."
            )

    n = len(reference)
    if n < block_len:
        raise ValueError(
            f"Not enough observations for a block bootstrap: n={n}, block_len={block_len}."
        )

    # Losses, not returns: the MCS eliminates models with HIGHER loss.
    losses = pd.DataFrame(
        {name: -candidate_returns[name].to_numpy(dtype=float) for name in names},
        index=reference,
    )

    mcs = _ArchMCS(
        losses,
        size=size,
        reps=n_boot,
        block_size=block_len,
        bootstrap="circular",
        seed=seed,
    )
    mcs.compute()

    included = sorted(str(x) for x in mcs.included)
    excluded = sorted(str(x) for x in mcs.excluded)

    return {
        "included": included,
        "excluded": excluded,
        "n_candidates": len(names),
        "n_included": len(included),
        "size": float(size),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "statistic": "mean_return",
        "interpretation": (
            f"{len(included)} of {len(names)} strategies are retained by the MCS "
            f"procedure at size {size:g} (mean-return losses). Retention is not "
            f"evidence that the retained strategies perform equally; it means the "
            f"data do not support eliminating them."
        ),
    }


def stepm_superior_models(
    candidate_returns: Mapping[str, pd.Series],
    benchmark: pd.Series,
    *,
    size: float = 0.05,
    block_len: int = 21,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Stepwise multiple testing (Romano & Wolf 2005), Econometrica 73(4).

    Addresses: P4 — `reality_check` collapses a 240-configuration search into
    ONE global p-value: "is the best of them better than the benchmark". When
    that fails to reject, it cannot say which candidates were close, and when
    it does reject it names only the single best. StepM controls the same
    familywise error rate but proceeds in stages — reject the clear cases,
    remove them, re-derive the critical value on what remains, repeat — so it
    returns a PER-STRATEGY verdict and is never less powerful than the
    single-step procedure at the same FWER.

    What it adds over `model_confidence_set`: the MCS asks which strategies
    survive mutual comparison, with no privileged benchmark. StepM keeps the
    project's pre-specified benchmark and asks which candidates beat IT. Both
    are one-sided in the project's favour-free direction; they answer different
    questions and are reported side by side, not as substitutes.

    WORDING (AGENTS.md section 5.2). A named model is one for which the null of
    no outperformance is rejected at FWER `size`. That is a rejection, not a
    demonstration of superiority in general — and an empty list is not evidence
    of no difference. `interpretation` is phrased accordingly.

    STATISTIC. Like the MCS, this consumes per-period losses (negated net
    returns), so it is a MEAN-RETURN criterion and takes no `statistic`
    argument. `reality_check` can also run on Sharpe; this cannot, and mixing
    the two in one table compares different quantities.

    Implementation note: delegates to `arch.bootstrap.StepM`, verified first
    rather than trusted — it wraps `arch.bootstrap.SPA`, whose `studentize`
    flag is accepted and never applied. StepM was checked to name exactly the
    genuinely superior models in a positive control, name none under
    exchangeability, be no less informative than the single-step Reality Check
    on the same data, and be monotone in `size`.

    Args:
        candidate_returns: Per-candidate simple net return series.
        benchmark: The pre-specified comparator, same index as every candidate.
        size: Familywise error rate. Default 0.05.
        block_len: Circular block length (~one trading month).
        n_boot: Bootstrap replications.
        seed: Seeded for reproducibility.

    Returns:
        dict with `superior_models` (sorted names whose null was rejected),
        `n_superior`, `n_candidates`, `size`, `block_len`, `n_boot`, `seed`,
        `statistic` and `interpretation`.

    Raises:
        ValueError: on an empty candidate set, misaligned indexes, NaN, or too
            little data for one block — the input discipline of
            `reality_check`, for the same reason.
    """
    from arch.bootstrap import StepM as _ArchStepM

    names = sorted(candidate_returns)
    if not names:
        raise ValueError("stepm_superior_models requires at least one candidate.")

    for name in names:
        series = candidate_returns[name]
        if not series.index.equals(benchmark.index):
            raise ValueError(
                f"stepm_superior_models requires identical date indexes; '{name}' "
                f"differs from the benchmark — refusing to align silently."
            )
        if series.isna().any():
            raise ValueError(
                f"stepm_superior_models requires NaN-free returns; '{name}' has NaN."
            )
    if benchmark.isna().any():
        raise ValueError("stepm_superior_models requires a NaN-free benchmark.")

    n = len(benchmark)
    if n < block_len:
        raise ValueError(
            f"Not enough observations for a block bootstrap: n={n}, block_len={block_len}."
        )

    # Losses, not returns: a model is "superior" when its loss is lower.
    models = pd.DataFrame(
        {name: -candidate_returns[name].to_numpy(dtype=float) for name in names},
        index=benchmark.index,
    )
    bench_loss = pd.Series(-benchmark.to_numpy(dtype=float), index=benchmark.index)

    step = _ArchStepM(
        bench_loss,
        models,
        size=size,
        block_size=block_len,
        reps=n_boot,
        bootstrap="circular",
        seed=seed,
    )
    step.compute()
    superior = sorted(str(x) for x in step.superior_models)

    return {
        "superior_models": superior,
        "n_superior": len(superior),
        "n_candidates": len(names),
        "size": float(size),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "statistic": "mean_return",
        "interpretation": (
            f"StepM rejects the null of no outperformance for {len(superior)} of "
            f"{len(names)} candidates at familywise error rate {size:g} "
            f"(mean-return losses)."
            if superior else
            f"StepM rejects the null of no outperformance for none of the "
            f"{len(names)} candidates at familywise error rate {size:g} "
            f"(mean-return losses). This does not establish that no candidate "
            f"differs from the benchmark."
        ),
    }


def _newey_west_long_run_cov(centered: np.ndarray, bandwidth: int) -> np.ndarray:
    """Bartlett-kernel HAC estimate of the long-run covariance of a k-vector.

    `centered` is (T, k) with column means already removed. Returns (k, k).
    """
    n_obs = centered.shape[0]
    psi = centered.T @ centered / n_obs
    for lag in range(1, bandwidth + 1):
        gamma = centered[lag:].T @ centered[:-lag] / n_obs
        weight = 1.0 - lag / (bandwidth + 1.0)
        psi = psi + weight * (gamma + gamma.T)
    return psi


def _sharpe_difference_point(r_a: np.ndarray, r_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Per-period Sharpe difference and the delta-method gradient at the estimate.

    Parameterised by the four moments (mu_a, mu_b, gamma_a, gamma_b) where
    gamma is the SECOND RAW moment, following Ledoit & Wolf (2008). With
    sigma^2 = gamma - mu^2 and SR = mu / sigma, the partials are
    d SR / d mu = gamma / sigma^3 and d SR / d gamma = -mu / (2 sigma^3).
    """
    mu_a, mu_b = float(r_a.mean()), float(r_b.mean())
    g_a, g_b = float((r_a**2).mean()), float((r_b**2).mean())
    var_a, var_b = g_a - mu_a**2, g_b - mu_b**2
    if var_a <= 0 or var_b <= 0:
        return float("nan"), np.full(4, np.nan)
    sd_a, sd_b = var_a**1.5, var_b**1.5
    diff = mu_a / np.sqrt(var_a) - mu_b / np.sqrt(var_b)
    gradient = np.array(
        [g_a / sd_a, -g_b / sd_b, -mu_a / (2 * sd_a), mu_b / (2 * sd_b)], dtype=float
    )
    return float(diff), gradient


def sharpe_difference_test(
    candidate: pd.Series,
    benchmark: pd.Series,
    *,
    block_len: int = 21,
    n_boot: int = 2000,
    alpha: float = 0.10,
    risk_free_annual: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
    hac_bandwidth: int | None = None,
    seed: int = 0,
) -> dict:
    """
    Studentized test for a DIFFERENCE of Sharpe ratios (Ledoit & Wolf 2008).

    Addresses: P4 — `paired_block_bootstrap` reports a percentile interval for
    the Sharpe difference. A percentile bootstrap of a non-pivotal statistic is
    not asymptotically refined, and its coverage degrades under exactly the two
    conditions this data has: heavy tails and serial dependence. Ledoit and Wolf
    derive the HAC standard error of the difference by the delta method over the
    four moments (mu_a, mu_b, gamma_a, gamma_b) and use it to STUDENTIZE a
    circular block bootstrap. The studentized statistic is asymptotically
    pivotal, so the bootstrap attains a higher order of accuracy.

    Why this matters here rather than in general: the frozen test on
    `full_2021` is ~455 days and the published intervals are wide enough that
    calibration, not point estimation, is the binding constraint. A
    better-calibrated test on identical data is the cheapest possible
    improvement to the evidence chain.

    The same standard error yields the MINIMUM DETECTABLE DIFFERENCE —
    see `sharpe_difference_mde`. `docs/EVALUATION_LIMITS.md` section 5 records
    that a pre-registered MDE "would have framed the negative result as
    DESIGNED rather than as a disappointment"; this supplies it, and it can be
    computed retrospectively for every comparison already published.

    WORDING (AGENTS.md section 5.2): a point estimate is an OBSERVED
    DIFFERENCE, an interval is UNCERTAINTY QUANTIFICATION, and failing to
    reject establishes nothing about equality.

    SHARPE CONVENTION, and it is a real difference worth stating. The delta
    method is derived on the raw-moment parameterisation, where
    sigma^2 = gamma - mu^2 — a population (ddof=0) variance. `annualized_sharpe`
    elsewhere in this module uses pandas' default ddof=1. The two cannot both
    hold exactly, so this function is internally consistent on the Ledoit-Wolf
    basis: `difference` IS `sharpe_candidate - sharpe_benchmark` exactly, and
    all three use ddof=0. The gap against `annualized_sharpe` is O(1/n) —
    around 7e-4 relative at n = 750 — and is a convention, not a disagreement.
    Reporting a difference that did not equal the difference of the two
    reported Sharpes would be the worse trade: a reader can reconcile a stated
    convention, but not an unexplained arithmetic mismatch.

    Args:
        candidate, benchmark: Simple periodic net returns, identical indexes.
        block_len: Circular block length (~one trading month).
        n_boot: Bootstrap replications.
        alpha: Two-sided level; the interval has coverage 1 - alpha.
        risk_free_annual: Converted per-period geometrically and subtracted
            from both series, matching `annualized_sharpe` exactly.
        periods: Periods per year. Sharpe, the difference and the standard
            error are all reported ANNUALIZED; the t-statistic is invariant to
            that rescaling.
        hac_bandwidth: Bartlett-kernel lag truncation. Defaults to the standard
            Newey-West rule floor(4 (T/100)^(2/9)). Ledoit and Wolf prefer a
            prewhitened QS kernel; Bartlett is the documented simplification.
        seed: Seeded, like every stochastic estimator here.

    Returns:
        dict with annualized `sharpe_candidate`, `sharpe_benchmark`,
        `difference`, `standard_error`, plus `t_statistic`, `p_value_hac`
        (normal approximation), `p_value_studentized_bootstrap`,
        `confidence_interval`, `mde_at_power_80`, and the settings used.

    Raises:
        ValueError: on misaligned indexes, NaN, or too little data for one
            block — the input discipline of `paired_block_bootstrap`.
    """
    if not candidate.index.equals(benchmark.index):
        raise ValueError(
            "sharpe_difference_test requires identical date indexes; "
            "refusing to align silently."
        )
    if candidate.isna().any() or benchmark.isna().any():
        raise ValueError("sharpe_difference_test requires NaN-free return series.")

    n = len(candidate)
    if n < 2 or n < block_len:
        raise ValueError(
            f"Not enough observations: n={n}, block_len={block_len}."
        )

    rf_periodic = (1 + risk_free_annual) ** (1 / periods) - 1
    r_a = candidate.to_numpy(dtype=float) - rf_periodic
    r_b = benchmark.to_numpy(dtype=float) - rf_periodic

    if hac_bandwidth is None:
        hac_bandwidth = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    hac_bandwidth = max(0, min(hac_bandwidth, n - 2))

    def point_and_se(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        diff, grad = _sharpe_difference_point(a, b)
        if not np.isfinite(diff):
            return float("nan"), float("nan")
        moments = np.column_stack([a, b, a**2, b**2])
        psi = _newey_west_long_run_cov(moments - moments.mean(axis=0), hac_bandwidth)
        variance = float(grad @ psi @ grad) / len(a)
        return diff, float(np.sqrt(variance)) if variance > 0 else float("nan")

    diff_obs, se_obs = point_and_se(r_a, r_b)
    if not np.isfinite(diff_obs) or not np.isfinite(se_obs) or se_obs == 0.0:
        raise ValueError(
            "sharpe_difference_test: degenerate sample — zero variance in a "
            "series, or a non-finite standard error."
        )
    t_obs = diff_obs / se_obs

    rng = np.random.default_rng(seed)
    studentized = np.empty(n_boot)
    for b in range(n_boot):
        rows = _circular_block_indices(n, block_len, rng)
        diff_b, se_b = point_and_se(r_a[rows], r_b[rows])
        studentized[b] = (diff_b - diff_obs) / se_b if np.isfinite(se_b) and se_b > 0 else np.nan
    studentized = studentized[np.isfinite(studentized)]

    p_boot = float((np.sum(np.abs(studentized) >= abs(t_obs)) + 1) / (len(studentized) + 1))
    lo_q, hi_q = np.quantile(studentized, [alpha / 2, 1 - alpha / 2])

    scale = np.sqrt(periods)
    diff_ann, se_ann = diff_obs * scale, se_obs * scale

    return {
        "n_observations": n,
        # ddof=0, NOT ddof=1 — see the "Sharpe convention" note in the docstring.
        "sharpe_candidate": float(np.mean(r_a) / np.std(r_a, ddof=0) * scale),
        "sharpe_benchmark": float(np.mean(r_b) / np.std(r_b, ddof=0) * scale),
        "difference": diff_ann,
        "standard_error": se_ann,
        "t_statistic": float(t_obs),
        "p_value_hac": float(2 * (1 - norm.cdf(abs(t_obs)))),
        "p_value_studentized_bootstrap": p_boot,
        "confidence_interval": (
            float(diff_ann - hi_q * se_ann),
            float(diff_ann - lo_q * se_ann),
        ),
        "mde_at_power_80": sharpe_difference_mde(se_ann, alpha=alpha, power=0.80),
        "ci_alpha": float(alpha),
        "block_len": int(block_len),
        "n_boot": int(n_boot),
        "hac_bandwidth": int(hac_bandwidth),
        "seed": int(seed),
    }


def sharpe_difference_mde(
    standard_error: float,
    *,
    alpha: float = 0.10,
    power: float = 0.80,
    n_observed: int | None = None,
    n_target: int | None = None,
) -> float:
    """
    Smallest Sharpe difference detectable at `power`, given a standard error.

    Addresses: P4 — `docs/EVALUATION_LIMITS.md` section 5 states that intervals
    containing zero "were the predictable consequence of the design, not a
    discovery about the models", and that a minimum-detectable-effect stated in
    advance "would have framed the negative result as DESIGNED rather than as a
    disappointment". This is that quantity: `(z_{1-alpha/2} + z_{power}) * SE`.

    Report it BEFORE running a comparison, not after. Its value is that it
    distinguishes "the effect is absent" from "this design could never have
    seen it", which a p-value alone cannot.

    Pass `n_observed` and `n_target` to project the standard error onto a
    different sample length under the usual root-n scaling — the honest way to
    answer "how much more data would this question need?".

    Args:
        standard_error: SE of the Sharpe difference, from
            `sharpe_difference_test`. Annualized in, annualized out.
        alpha: Two-sided significance level.
        power: Desired power, conventionally 0.80.
        n_observed, n_target: Optional; supply BOTH to rescale.

    Returns:
        The minimum detectable difference, in the units of `standard_error`.

    Raises:
        ValueError: if only one of `n_observed` / `n_target` is given, or
            either is not positive, or alpha/power are outside (0, 1).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1); got {power}")
    if (n_observed is None) != (n_target is None):
        raise ValueError(
            "supply both n_observed and n_target to rescale, or neither."
        )

    se = float(standard_error)
    if n_observed is not None:
        if n_observed <= 0 or n_target <= 0:
            raise ValueError("n_observed and n_target must be positive.")
        se = se * np.sqrt(n_observed / n_target)

    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def probability_of_backtest_overfitting(
    trial_returns: Mapping[str, pd.Series],
    *,
    n_splits: int = 16,
) -> dict:
    """
    Probability of backtest overfitting via CSCV (Bailey, Borwein, Lopez de
    Prado & Zhu, 2017), Journal of Computational Finance 20(4).

    Addresses: P4 — `deflated_sharpe_ratio` is the other half of this research
    programme and fails differently, which is the point of having both. DSR is
    PARAMETRIC: it needs a trial count and a variance of trial Sharpes, and is
    only as good as those two inputs. CSCV is model-free, non-parametric and
    symmetric. It splits the trial-by-time performance matrix into S blocks,
    forms every balanced in-sample / out-of-sample partition, and asks how
    often the configuration chosen in-sample lands below the OOS median. That
    frequency is the PBO.

    The question it answers is the one this project keeps asking: is the best
    observed candidate a real edge or the luckiest of a wide search? Unlike the
    Reality Check it needs no benchmark, and unlike the DSR it makes no
    distributional assumption about the trial pool.

    CAUTION, and it must be reported with the number: PBO measures the
    SELECTION PROCEDURE, not the strategies alone. It rises toward 1 when
    in-sample advantage is sample-specific — the realistic case for a
    hyperparameter grid fitted repeatedly to one dataset — so a high value over
    a wide grid is not by itself evidence that every candidate is worthless.

    The converse is worth stating too, because it is easy to assume otherwise:
    breadth alone does not inflate PBO. Measured here on INDEPENDENT noise
    series, PBO stays near 0.5 whether the search spans 4 configurations or
    120, because the in-sample winner's out-of-sample rank is then uniform.
    What drives PBO up is dependence between candidates and the selection, not
    the count. Reading a high PBO as "we simply tried too many things" is
    therefore the wrong inference.

    Method. Splits are contiguous in time and disjoint; with S blocks there are
    C(S, S/2) partitions, each using S/2 blocks in-sample and the complement
    out-of-sample. Performance is the per-period Sharpe ratio. For each
    partition the in-sample winner is found, its out-of-sample rank omega among
    all N configurations is taken, and the logit
    lambda = log(omega / (1 - omega)) recorded. PBO is the fraction of
    partitions with lambda <= 0.

    Implementation detail worth knowing: Sharpe ratios for every partition are
    computed from per-block sums of x and x^2 rather than by re-slicing the
    matrix, so the cost is one (C x S) by (S x N) matrix product rather than
    C x N independent passes. At S = 16 that is 12 870 partitions, which is
    seconds rather than hours.

    Args:
        trial_returns: Per-configuration simple return series, identical
            indexes. At least two configurations — ranking one is vacuous.
        n_splits: S, the number of contiguous blocks. Must be even and at
            least 4. The paper uses 16. Observations beyond a multiple of S
            are dropped from the END and the count is reported.

    Returns:
        dict with `pbo`, `n_trials`, `n_splits`, `n_partitions`,
        `n_observations_used`, `n_observations_dropped`, `median_logit`,
        `performance_degradation_slope`, `probability_of_loss`, and
        `interpretation`.

    Raises:
        ValueError: on fewer than two trials, an odd or too-small `n_splits`,
            misaligned indexes, NaN, or too few observations per block.
    """
    from itertools import combinations

    names = sorted(trial_returns)
    if len(names) < 2:
        raise ValueError(
            f"probability_of_backtest_overfitting needs at least two trials; got {len(names)}."
        )
    if n_splits < 4 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even and >= 4; got {n_splits}.")

    reference = trial_returns[names[0]].index
    for name in names:
        series = trial_returns[name]
        if not series.index.equals(reference):
            raise ValueError(
                f"probability_of_backtest_overfitting requires identical date indexes; "
                f"'{name}' differs from '{names[0]}' — refusing to align silently."
            )
        if series.isna().any():
            raise ValueError(
                f"probability_of_backtest_overfitting requires NaN-free returns; "
                f"'{name}' has NaN."
            )

    matrix = np.column_stack([trial_returns[n].to_numpy(dtype=float) for n in names])
    n_obs_total, n_trials = matrix.shape
    block_len = n_obs_total // n_splits
    if block_len < 2:
        raise ValueError(
            f"Too few observations for {n_splits} blocks: {n_obs_total} rows gives "
            f"{block_len} per block; need at least 2."
        )
    used = block_len * n_splits
    trimmed = matrix[:used]

    # Per-block sufficient statistics: a Sharpe over any union of blocks is a
    # function of the summed counts, sums and sums of squares.
    blocks = trimmed.reshape(n_splits, block_len, n_trials)
    block_sum = blocks.sum(axis=1)                 # (S, N)
    block_sumsq = (blocks**2).sum(axis=1)          # (S, N)

    partitions = list(combinations(range(n_splits), n_splits // 2))
    masks = np.zeros((len(partitions), n_splits), dtype=float)
    for i, combo in enumerate(partitions):
        masks[i, list(combo)] = 1.0
    complement = 1.0 - masks

    def sharpes(mask: np.ndarray) -> np.ndarray:
        count = block_len * (n_splits // 2)
        total = mask @ block_sum                   # (C, N)
        total_sq = mask @ block_sumsq              # (C, N)
        mean = total / count
        var = total_sq / count - mean**2
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(var > 0, mean / np.sqrt(np.maximum(var, 1e-300)), 0.0)
        return out

    is_perf = sharpes(masks)
    oos_perf = sharpes(complement)

    winners = np.argmax(is_perf, axis=1)
    rows = np.arange(len(partitions))
    winner_oos = oos_perf[rows, winners]

    # Rank of the in-sample winner among all trials out of sample (1 = worst).
    ranks = (oos_perf < winner_oos[:, None]).sum(axis=1) + 1
    omega = ranks / (n_trials + 1.0)
    omega = np.clip(omega, 1e-12, 1 - 1e-12)
    logits = np.log(omega / (1 - omega))

    pbo = float(np.mean(logits <= 0.0))

    # Performance degradation: OOS performance of the chosen config regressed
    # on its IS performance. A negative slope is the overfitting signature.
    winner_is = is_perf[rows, winners]
    if np.std(winner_is) > 0:
        slope = float(np.polyfit(winner_is, winner_oos, 1)[0])
    else:
        slope = float("nan")

    return {
        "pbo": pbo,
        "n_trials": n_trials,
        "n_splits": int(n_splits),
        "n_partitions": len(partitions),
        "n_observations_used": int(used),
        "n_observations_dropped": int(n_obs_total - used),
        "median_logit": float(np.median(logits)),
        "performance_degradation_slope": slope,
        "probability_of_loss": float(np.mean(winner_oos <= 0.0)),
        "interpretation": (
            f"The configuration selected in-sample falls below the out-of-sample "
            f"median in {pbo:.1%} of {len(partitions)} balanced partitions "
            f"(PBO = {pbo:.4f}, {n_trials} trials). PBO rises toward 1 with the "
            f"size of the search regardless of genuine skill, so it describes the "
            f"selection procedure as much as the strategies."
        ),
    }
