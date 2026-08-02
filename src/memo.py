"""
memo.py — content-addressed memoization for the deterministic estimators.

WHY THIS EXISTS. The Phase 4C comparison runs 14 strategies over both
universes, and the runner iterates strategies on the OUTER loop and rebalance
dates on the inner one. Several strategies are, by design, the same estimator
under a different *post-estimation* setting:

    rf_signal, rf_signal_cost, rf_signal_shrunk, rf_signal_rank,
    rf_signal_cost_dcc      → identical RandomForest fit; they differ only in
                              the mu transform (applied after prediction) and
                              in optimizer arguments (turnover penalty,
                              covariance rung)
    dcc_garch,
    rf_signal_cost_dcc      → identical DCC-GARCH fit

Measured on `etf_2017` at one rebalance date (scripts/profile_phase4c.py):
7 calls to the panel model with **2** distinct inputs, 8 HMM fits with **1**,
2 DCC fits with **1** — about 80% of the per-date work was recomputation of a
value already computed.

WHAT IS DELIBERATELY *NOT* MEMOIZED, AND WHY. The HMM (`regime.fit_hmm`) shows
the largest duplication of the three — eight identical-input fits per date —
and was cached in the first version of this change. It was then **removed**:
despite fixed restart seeds, hmmlearn's EM likelihood is not bit-for-bit
reproducible on this runtime, so reusing a prior fit would have changed the
calculation rather than merely avoided repeating it. A speedup that alters a
research result is not a speedup. `fit_hmm` therefore keeps its
`_fit_hmm_uncached` split for readability only, with an explicit comment at
the call site so this is not silently re-added later.

The rule that follows: memoize an estimator only after its bit-reproducibility
has been demonstrated on real data, not inferred from a seed argument.

WHY A CACHE IS SOUND HERE, AND WHY IT CANNOT GO STALE. This project has been
bitten by stale caches three times (a Dagster gRPC code server serving a
three-day-old interpreter, Streamlit's module cache, and the API's
`lru_cache(maxsize=1)` pinning artifacts for a process lifetime). Those all
shared one property: the cache key did NOT determine the cached value —
invalidation depended on time, on a path, or on a process restart, and
therefore had to be remembered by a human.

Here the key **is** a digest of every input that can affect the result. A
changed input is a different key, so a stale hit is not a bug that must be
avoided — it is unrepresentable. Two conditions make that argument complete,
and both are enforced by tests rather than asserted:

  1. The wrapped functions are deterministic — `random_state=0` on both
     panel models and fixed DCC initial values. If they were not, caching
     would silently change reported results, so `tests/test_memo.py` proves
     determinism directly rather than trusting it.
  2. The key covers every argument the result depends on. A dropped argument
     would make two genuinely different questions collide, which is the one
     way this could corrupt a number.

THE ONE ASSUMPTION, STATED PLAINLY. "Same inputs ⇒ same output" also assumes
the CODE is fixed for the lifetime of the process. That holds in production
(DVC re-runs a stage when its source changes, and each run is a fresh
process) but it is violated deliberately by any test that `monkeypatch`es an
estimator's internals: such a test changes behaviour without changing the
inputs, so a cached entry would mask the failure path it means to exercise.
That is exactly what happened when this was first wired in — it silently
defeated two Ledoit-Wolf fallback tests in `test_dcc_garch.py`. The fix is
test-side and permanent: `tests/conftest.py::_clear_estimator_caches` resets
every cache around every test.

NO EFFECT ON THE NO-LOOKAHEAD GUARANTEE. Keys are digests of the frames the
engine already sliced to `:τ`. Two different rebalance dates have different
window contents, hence different keys; a hit can only occur when the inputs
are byte-identical, in which case a deterministic function would have
returned the same value anyway. The cache can therefore not move information
backwards in time. `tests/test_memo.py` re-runs the Phase 4 future-corruption
gate with caching live.

Addresses: P4 — the cache is sound only because the estimators are
deterministic, which is the same property the no-lookahead engine and the
reproducible-pipeline guarantee both rest on. Making that dependency explicit
(and tested) is the point of this module.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Must comfortably exceed the number of rebalance dates in one universe
# (248 on `etf_2017` today). The runner's loop order is strategies OUTER,
# dates INNER, so the second strategy only hits entries the first strategy
# wrote 248 dates ago — a cache smaller than the schedule would evict every
# one of them before it could be reused and silently degrade to no cache at
# all. Sized with headroom, and bounded so it cannot grow without limit.
DEFAULT_MAXSIZE = 1024


def content_key(*parts: Any) -> str:
    """Stable digest of everything that can affect an estimator's output.

    Addresses: P4 — this is the cache's correctness argument in one function:
    if two calls produce the same key they were given the same inputs, so a
    deterministic estimator owes them the same answer.

    Handles the argument types the wrapped estimators actually receive.
    DataFrame column names and dtypes are hashed alongside the values because
    `hash_pandas_object` alone would not distinguish two frames that differ
    only by column order — which changes what a positional model fit means.
    """
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, pd.DataFrame):
            h.update(b"DF")
            h.update(pd.util.hash_pandas_object(part, index=True).to_numpy().tobytes())
            h.update("|".join(map(str, part.columns)).encode())
            h.update("|".join(map(str, part.dtypes)).encode())
        elif isinstance(part, pd.Series):
            h.update(b"S")
            h.update(pd.util.hash_pandas_object(part, index=True).to_numpy().tobytes())
            h.update(str(part.name).encode())
        elif isinstance(part, np.ndarray):
            h.update(b"A")
            h.update(np.ascontiguousarray(part).tobytes())
            h.update(str(part.dtype).encode())
        elif isinstance(part, dict):
            h.update(b"D")
            h.update(repr(sorted(part.items(), key=lambda kv: str(kv[0]))).encode())
        elif part is None:
            h.update(b"N")
        else:
            h.update(b"R")
            h.update(repr(part).encode())
        h.update(b"\x1e")
    return h.hexdigest()


class ContentCache:
    """A bounded, content-addressed LRU with hit/miss counters.

    Addresses: P4 — see module docstring. Deliberately NOT `functools.
    lru_cache`: the arguments here are DataFrames, which are unhashable, and
    an implicit decorator would hide the one thing that has to be auditable —
    exactly which inputs went into the key.
    """

    def __init__(self, name: str, maxsize: int = DEFAULT_MAXSIZE) -> None:
        self.name = name
        self.maxsize = maxsize
        self._store: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        """Return the cached value for `key`, computing it once if absent."""
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]

        self.misses += 1
        value = compute()
        self._store[key] = value
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop every entry and reset counters.

        Only needed by tests that assert on miss counts — correctness never
        requires it, since a key that is present was produced by identical
        inputs.
        """
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int]:
        return {
            "name": self.name,
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._store),
            "maxsize": self.maxsize,
        }

    def __len__(self) -> int:
        return len(self._store)
