# Data governance — provenance, licensing, and audit logging

> Companion to [`MODEL_GOVERNANCE.md`](MODEL_GOVERNANCE.md) (model lifecycle)
> and [`INFERENCE_CONTRACT.md`](INFERENCE_CONTRACT.md) (what the API serves).
>
> **This is a description of the processing this project performs and the
> assumptions it operates under. It is not a legal opinion and not a claim of
> certification or compliance review.** No counsel and no data-protection
> officer has reviewed it.

---

## 1. What data this project processes

| Category | Present? |
|---|---|
| Public market prices and volumes | Yes |
| Public macroeconomic series | Yes |
| Public corporate disclosures (dividends, financial ratios) | Yes |
| **Personal data** (identifiable natural persons) | **No** |
| **Client data** (portfolios, accounts, holdings, orders) | **No** |
| **Special-category data** (health, biometric, political, etc.) | **No** |
| Employee or user behavioural data | No |

The system has no client, no account, no order path, and no user profile. Its
inputs are prices of instruments and published economic statistics.

`tests/test_data_governance.py` asserts no Gold artifact carries a column whose
name suggests a person, an account, or a client — a cheap check that would fire
if the universe were ever extended in that direction without a fresh
assessment.

## 2. Source provenance and licensing

| Source | Used for | Module | Redistribution |
|---|---|---|---|
| Yahoo Finance (via `yfinance`) | ETF prices, MAD FX rates | `ingest.py` | **Not redistributed.** Personal, non-commercial use per Yahoo's terms |
| medias24 (via `BVCscrap`) | BVC equity prices | `ingest.py`, `clean.py` | **Not redistributed.** Free tier, rolling window |
| FRED (St. Louis Fed) | VIX, DGS10, DTWEXBGS, BAA10Y | `ingest.py` | Public domain for US-government-produced series; API key required and kept in `.env` |
| casablanca-bourse.com | BVC dividend history and ex-dates | `dividends.py` | **Not redistributed.** Public issuer disclosures, scraped politely |
| stockanalysis.com | BVC valuation ratios (research experiment) | `fundamentals.py` | **Not redistributed.** S&P Global-sourced; `robots.txt` permits crawling |
| investing.com | 20-year deep BVC history (research experiment) | `experiments/` | **Not redistributed.** Terms prohibit redistribution — the strictest constraint in this table, and the reason the DVC remote is private |

**Consequence, and it is enforced rather than intended:** `data/` is excluded
from Git, the DVC remote is a **private** Cloudflare R2 bucket, and the Docker
image ships with no market data — asserted in CI by a gate that counts
`data/gold` entries in the image and requires zero.

Anyone reproducing this work obtains the data from the sources themselves. The
snapshot manifest lets them verify they got the same bytes without our
republishing any.

## 3. Legal assumptions

### Moroccan Law 09-08 (protection des personnes physiques à l'égard du traitement des données à caractère personnel)

Law 09-08 governs the processing of *données à caractère personnel* — data
relating to an identified or identifiable natural person. This project
processes none: its records are instrument prices and macroeconomic
aggregates, with no data subject. On that basis the obligations attaching to a
*responsable du traitement* (CNDP declaration, data-subject rights, transfer
authorisation) **are not engaged by the model itself**.

### GDPR

The same reasoning applies. With no personal data there is no processing under
Article 4(2) of personal data, and no controller/processor relationship arises
from the model.

### The condition on both statements

Both hold **only while the input set contains no personal data**. They would
stop being true the moment any of these entered the system:

- a client portfolio, holding, or transaction history;
- an account number, client identifier, or advisor name;
- a user profile, risk questionnaire, or suitability assessment;
- API request logs that record who asked for what.

Any such extension requires a fresh assessment **before** it is built, not
after. That is the single most likely way this project would acquire
obligations it currently does not have — and the most likely way to acquire
them accidentally is through logging.

## 4. Audit log format

The API is read-only and serves published artifacts, so an audit trail needs
to answer *which artifact version was served, to which endpoint, with what
outcome* — and nothing about the caller's content.

One JSON object per line:

```json
{
  "ts": "2026-08-03T17:22:41.918Z",
  "event": "api_request",
  "endpoint": "/published-allocation",
  "method": "GET",
  "status": 200,
  "duration_ms": 12,
  "universe": "full_2021",
  "strategy": "regime_conditional",
  "artifact_commit": "0c192d6f1a2b",
  "provenance_status": "manifest_present",
  "api_version": "1.1.0"
}
```

| Field | Why |
|---|---|
| `endpoint`, `method`, `status`, `duration_ms` | Operational: what was asked, what happened |
| `universe`, `strategy` | Enumerated path parameters, not free text — a fixed, closed vocabulary |
| `artifact_commit`, `provenance_status`, `api_version` | **The point of the log.** Reconstructing a past answer requires knowing which artifact set produced it |

### Prohibited fields

Never logged, in any environment:

- request bodies, query strings verbatim, or headers;
- caller IP addresses, session identifiers, or credentials;
- any free-text value supplied by a caller.

Enumerated parameters are logged **after validation** — `_check_universe` and
`_check_strategy` reject anything outside a known set, so what reaches the log
is one of a handful of fixed strings rather than caller-controlled text. An
unvalidated parameter must never be logged.

`tests/test_data_governance.py` inspects the API source and fails if it logs a
request body, a header, or a raw query value. The current API performs no
request-content logging at all; the test exists to keep that true as endpoints
are added.

## 5. Dependency and secret hygiene

| Control | State |
|---|---|
| Secrets in tracked files | None. Verified across the full history, including key-shaped high-entropy strings |
| `.env` | Gitignored, never committed in any branch, excluded from the Docker build context |
| `.dvc/config.local` (R2 credentials) | Gitignored **and** excluded from the Docker build context — gitignoring a secret does not keep it out of an image |
| FRED API key | Read from `FRED_API_KEY`; never a default, never a literal |
| Dependency pinning | `requirements.txt` states intent with ranges; `requirements.lock.txt` pins the exact environment the results came from and is what the image installs and the manifest hashes |
| Image credentials | CI gate asserts no `.env` and no `.dvc/config.local` inside the built image |
| Container privilege | Non-root (uid 10001), asserted in CI |

## 6. What is *not* claimed

- No penetration test, dependency CVE scan, or SAST run has been performed.
- No legal review of the source terms above has been obtained; they are read
  as published and applied conservatively (assume no redistribution).
- No data-protection impact assessment exists, because no personal data is
  processed. One would be required before that changes.
- The API has no authentication. It is intended to run locally or on a trusted
  network, and it serves only already-published research artifacts.
