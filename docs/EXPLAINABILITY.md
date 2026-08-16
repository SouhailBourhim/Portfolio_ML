# Explainability

> Companion to [`MODEL_CARD_REGIME_CONDITIONAL.md`](MODEL_CARD_REGIME_CONDITIONAL.md)
> and [`MODEL_GOVERNANCE.md`](MODEL_GOVERNANCE.md).
> Artifact: `data/gold/model_explanations.json` (DVC stage `explainability`).
> Code: [`src/explainability.py`](../src/explainability.py),
> [`src/run_explainability.py`](../src/run_explainability.py).
> Tests: `tests/test_explainability.py`.

---

## 1. Two systems, two different explanations

The reference system and the challengers are explained by different methods,
and the asymmetry is deliberate rather than a shortcut.

| System | Method | Why |
|---|---|---|
| `regime_conditional` | **Deterministic decision trace** | It fits no predictive function over features. It classifies a regime and delegates to a Markowitz optimizer, so attribution has nothing to attribute |
| `rf_signal_tuned` | Exact path decomposition (Saabas) + permutation importance | It does fit a predictive function |
| `xgb_signal_tuned` | TreeSHAP (xgboost native) + permutation importance | Same |

Forcing SHAP onto the reference system would produce a plausible-looking chart
that explains nothing about how the portfolio was actually chosen. The trace
does explain it, because for this system the decision genuinely is a short
chain of deterministic steps.

## 2. The decision trace

Seven steps, each a function of data available at the decision date τ:

1. **HMM inputs at τ** — the three market features the regime model reads
2. **Posterior** — probability of each state, plus `converged`, the
   log-likelihood, and the seed that won the restart competition
3. **Regime label** — with the state→label mapping, recomputed at every refit
   because `hmmlearn` does not guarantee stable state ordering
4. **Selected sub-optimizer** — which of the two Markowitz strategies received
   the decision
5. **Moment inputs** — covariance estimator, annualized mean and volatility per
   asset, training rows and start date
6. **Binding constraints** — which assets rest on the weight cap, which are at
   zero, and whether the cap is close to determining the allocation outright
7. **Weights**, plus `fallback_used` and, when true, the reason in words

### Why step 6 is a first-class field

On `etf_2017` the current trace reports **4 of 5 assets at the cap** and one at
zero — an observed allocation, not a forced one. The artifact states the
arithmetic honestly: 5 assets × 25% = 1.25, so feasibility requires at least
four assets with *positive weight* and caps any asset five percentage points
above equal weight. That does **not** force a corner — equal weight is feasible
with nothing at the cap. What the flag records is the empirical finding that on
this universe the constraint *dominates* the objective: the observed corner is
where every objective landed, measured, not derived.

⚠️ The field is still named `cap_is_near_determining` for artifact-schema
stability. "Dominating" is the accurate word; renaming the key is tracked as a
follow-up because it would change a Gold artifact's schema.

This is the most important honest caveat in the project's results, and burying
it in prose would be a choice. It is computed and reported per decision.

## 3. Challenger attribution

### Global — seeded permutation importance

Mean increase in MSE when one feature is shuffled, averaged over repeats, with
an explicit seed. One procedure for both model families, so **global rankings
are comparable across models**. Emitted as a ranked list rather than a mapping:
the artifact is serialized with `sort_keys=True` for stable diffs, which would
re-alphabetize a dict and silently destroy the ordering.

### Local — exact, additive contributions

| Model | Algorithm | Additive? |
|---|---|---|
| RandomForest | Decision-path decomposition (Saabas), ~30 lines against sklearn's `tree_` API | Yes, exactly |
| XGBoost | TreeSHAP via `Booster.predict(pred_contribs=True)` | Yes, exactly |

Both satisfy `bias + Σ contributions = prediction`. The artifact carries a
`reconstruction_error` field, currently around `1e-19` (RF) and `1e-10` (XGB) —
float noise. Because the property is exact, the test asserting it is a real
check rather than a tolerance to be tuned.

**Trade-off being accepted:** the two families use different exact algorithms,
so local contribution *magnitudes* are not comparable across models. Rankings
within a model are, and global importance is comparable across both.

### Why not the `shap` package

Two reasons, in order of weight.

1. **Exact attribution is already available.** XGBoost ships TreeSHAP; sklearn's
   tree API supports the exact path decomposition. Adding a dependency to get a
   capability already present is not a trade.
2. **`shap` pulls in numba and llvmlite.** This project has already lost a model
   to a native-library conflict — `LSTMSignalStrategy` was fully built and
   tested, then withdrawn after `torch` and `xgboost` segfaulted in one
   process. Adding another native stack for no capability gain is a bad bet.

Recorded in the artifact itself under `explanation_policy.shap_rationale`, so
the decision travels with the output.

## 4. Ranking → weights

The step most often left implicit: a model can rank assets well and still
produce weights that ignore the ranking, because the covariance term and the
weight cap both intervene. The artifact reports, per asset, the predicted
expected return, its rank, the resulting weight, the weight's rank, and whether
that weight is on the cap — plus the Spearman concordance between the two
rankings.

The current run makes the point better than any explanation could:

| Universe | Model | Rank concordance |
|---|---|---:|
| `etf_2017` | `rf_signal_tuned` | **−1.000** |
| `etf_2017` | `xgb_signal_tuned` | +0.894 |
| `full_2021` | `rf_signal_tuned` | +0.466 |
| `full_2021` | `xgb_signal_tuned` | +0.690 |

A concordance of **−1.000** means the allocation is the exact reverse of the
predicted ranking: the top-predicted asset receives **zero weight**. That is not
a bug. The optimizer trades predicted return against covariance under a binding
cap, and on this universe the cap dominates. It is precisely the kind of thing
that stays invisible when a model is judged only by its Sharpe ratio.

Concordance is reported as **`null`, not `0.0`**, when either ranking is
constant — a fully binding cap can give every asset the same weight, and
reporting zero would read as "the weights ignore the ranking" when the truth is
that the constraint left no ordering to express.

## 5. What the tests lock in

| Rule | Test |
|---|---|
| The trace reads no row after τ | `test_trace_reads_no_row_after_the_decision_date` — corrupts the *future* of both frames and requires the decision to be unchanged |
| Explanations are reproducible | `test_trace_is_deterministic_across_repeated_calls`, `test_local_contributions_are_deterministic`, `test_permutation_importance_is_seeded_and_reproducible` |
| Contributions are internally consistent | `test_forest_contributions_reconstruct_the_prediction`, `test_xgboost_contributions_reconstruct_the_prediction` |
| Feature names match the training matrix | `test_contribution_keys_match_the_training_feature_names`, `test_importance_covers_exactly_the_training_features` |
| A binding cap is detected by tolerance, not equality | `test_assets_at_cap_are_reported_with_tolerance_not_equality` |
| An uncertain regime resolves defensively, and says so | `test_a_nonconverged_fit_reports_the_defensive_fallback_and_says_why` |
| Undefined concordance is not reported as zero | `test_concordance_is_undefined_not_zero_when_the_cap_pins_every_weight` |

### One note on the no-lookahead test

Comparing the two traces with exact equality fails, and the reason is worth
recording rather than hiding behind a tolerance. Writing into a frame beyond τ
changes the block layout of the copy, so numpy's pairwise summation walks a
differently-aligned buffer and a mean over the *identical* leading values can
differ in the last bit. Every structural field — regime, sub-optimizer, weights,
constraints, fallback — is still compared **exactly**; only floats carry a
tolerance. Those structural fields are what the guarantee is about.

## 6. Limits

- **Attribution is not causation.** Every method here describes what a fitted
  function does with its inputs. None identifies a cause of a market return.
  The artifact repeats this per challenger.
- **One rebalance date.** The artifact explains the most recent decision per
  universe, derived from the data rather than chosen. It is not a history of
  every decision.
- **Permutation importance is affected by correlated features.** Trailing
  returns over overlapping windows are correlated by construction, so a
  shuffled feature's information may survive in its neighbours and its
  importance be understated. The measured values are small in absolute terms
  and should be read as a ranking, not as a variance decomposition.
- **The challengers being explained did not beat the reference system.** Their
  cards say so, and nothing in this document changes that.
