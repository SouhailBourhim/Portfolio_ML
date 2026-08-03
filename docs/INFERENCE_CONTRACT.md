# Published research allocation contract

The FastAPI service exposes **published research artifacts**. It does not
accept client data, refit models, refresh market data, create an allocation for
a requested date, or execute an order.

## Contract endpoints

- `GET /version` identifies the API and reports whether a snapshot manifest is
  available for the served artifact directory.
- `GET /published-allocation?universe=...&strategy=regime_conditional` returns
  the latest allocation already published in `dashboard_weights.parquet`.
- `GET /explanations/{universe}` returns the DVC-produced final-rebalance
  decision trace in `model_explanations.json`.
- `GET /model-card/{system}` returns a generated model card. Valid systems are
  `regime_conditional`, `rf_challenger`, and `xgb_challenger`.

Every typed contract response contains `provenance`, whose `status` is one of
three states the service can actually emit:

| `status` | Meaning | Safe to cite? |
|---|---|---|
| `manifest_present` | A manifest exists and was written from a clean tree | Only after `snapshot.py verify` |
| `manifest_dirty_tree` | A manifest exists but was written from a **dirty** tree, so `producer_commit` does not identify the code that produced the artifacts | **No** — `snapshot.py verify` rejects it |
| `unavailable` | No manifest for this artifact directory | No — local inspection only |

The API reads the manifest; it never recomputes checksums. There is deliberately
no `verified_snapshot` status, because nothing here could emit one — advertising
it would imply a verification this service does not perform. Full verification
stays an explicit release gate. Before sharing any release, run:

```bash
./scripts/dvc.sh status
./.venv/bin/python src/snapshot.py verify
```

## Explicit non-capabilities

The API is not a trading, advisory, or client-recommendation service. It has no
endpoint for order placement, capital sizing, leverage, shorting, suitability,
or arbitrary-date optimisation. A request never calls `Strategy.fit()` or
changes a published artifact.

## Error behavior

Unknown universes, strategies, systems, and explanation entries return HTTP
404 with the available values. Missing required Gold artifacts return HTTP 503
and name the missing artifact. Clients must treat a 503 or unavailable
provenance as a failed research-release check, not as permission to fall back
to a stale cached allocation.
