"""Step-5 verification. Run after `dvc repro` completes, BEFORE anything is published.

Gate 1 is the one that can stop publication: `etf_2017` must be bit-for-bit
unchanged. Those result stages consume both universes, so they recompute the USD
benchmark from unchanged inputs — any difference means the FX correction leaked
into a universe that must not have moved.
"""
import hashlib, json, pathlib, sys
import pandas as pd

GOLD = pathlib.Path("data/gold")
# Baseline hashes of each artifact's etf_2017 block, captured BEFORE the
# rebuild. Path is overridable so the check can be re-run against any recorded
# baseline; --write-baseline regenerates it from the current tree.
DEFAULT_BASELINE = pathlib.Path("data/interim/pre_rebuild_etf2017.json")

def etf_sha(path: pathlib.Path):
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    acc = []
    def find(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "etf_2017":
                    acc.append(json.dumps(v, sort_keys=True))
                else:
                    find(v)
        elif isinstance(o, list):
            for v in o:
                find(v)
    find(d)
    return hashlib.sha256("".join(acc).encode()).hexdigest()[:16] if acc else None


baseline_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BASELINE
if not baseline_path.is_file():
    sys.exit(f"baseline not found at {baseline_path}; capture it before the rebuild")

print("=" * 78)
print("GATE 1 — etf_2017 must be UNCHANGED (non-negotiable)")
print("=" * 78)
before = json.loads(baseline_path.read_text())
failures = []
for name, rec in before.items():
    old = rec.get("etf_2017_sha")
    if not old:
        continue
    new = etf_sha(GOLD / name)
    ok = (old == new)
    print(f"  {'OK ' if ok else 'DIFF'}  {name:34s} {old} -> {new}")
    if not ok:
        failures.append(name)

print("\n" + "=" * 78)
print("GATE 2 — full_2021 is MAD, etf_2017 is USD")
print("=" * 78)
man = json.loads((GOLD / "currency_manifest.json").read_text())
for u, v in man["universes"].items():
    print(f"  {u:10s} converted={v.get('converted')} base={v.get('base_currency')}")
full = pd.read_parquet(GOLD / "log_returns.parquet")
etf = pd.read_parquet(GOLD / "log_returns_etf.parquet")
print(f"  full_2021 {full.shape} {full.index.min().date()} -> {full.index.max().date()}")
print(f"  etf_2017  {etf.shape} {etf.index.min().date()} -> {etf.index.max().date()}")

print("\n" + "=" * 78)
print("GATE 3 — OOS window dates unchanged (2022-07-01+)")
print("=" * 78)
oos = full.index[full.index >= pd.Timestamp("2022-07-01")]
print(f"  {len(oos)} dates, {oos.min().date()} -> {oos.max().date()}  (expected 1061)")

print("\n" + "=" * 78)
print("NEW full_2021 HEADLINE (MAD) vs OLD (mixed-currency)")
print("=" * 78)
print("  OLD (AGENTS.md §5.1): max_sharpe 1.1644 | regime_conditional 1.2363 | +6.2%")
try:
    show = json.loads((GOLD / "dashboard_showcase.json").read_text())
    print("  NEW: " + json.dumps(show.get("full_2021", show), indent=2)[:900])
except Exception as e:
    print(f"  dashboard_showcase.json not yet rebuilt ({e})")

print("\n" + "=" * 78)
print("VERDICT: " + ("PASS" if not failures else f"STOP — etf_2017 moved in {failures}"))
print("=" * 78)
sys.exit(1 if failures else 0)
