# Portfolio ML — reproducible runtime for the research API, pipeline and tests.
#
# Data is NOT baked in. `data/` is DVC-managed and licence-restricted, so it is
# mounted from the host or fetched with `dvc pull`; an image carrying frozen
# market data would both go stale and redistribute data we may not redistribute.
#
# Quick start (see also the Docker section in README.md):
#   docker compose up api               # research API on http://localhost:8000
#   docker compose run --rm test        # offline test suite, zero config
#   docker compose run --rm pipeline    # full Bronze→Silver→Gold (needs .env)
#   docker compose up notebook          # Jupyter on http://localhost:8888

# Pinned to 3.11 to match requirements.lock.txt and the `python` field of
# data/gold/snapshot_manifest.json. The image previously used 3.13 while the
# lock recorded 3.11.14, so the "reproducible environment" was reproducing a
# different interpreter than the one the committed results came from.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependencies first, in their own layer, so code edits don't re-install them.
# Installing the LOCK, not the range file: requirements.txt states intent with
# permissive `>=` bounds, which names a family of environments rather than the
# one these numbers came from. xgboost in particular is version-sensitive here
# (the single-worker policy was validated against one specific build).
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# Non-root. The API reads committed artifacts and writes nothing, so it has no
# reason to hold root inside the container. `data/` is mounted read-write for
# the pipeline service, hence the ownership fix rather than a read-only mount.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# One image serves all three surfaces; the compose service's `command` picks
# which. 8000 API (the default CMD), 8501 Streamlit dashboard, 8888 Jupyter.
EXPOSE 8000 8501 8888

# Default to serving the API, and refuse to start against an incomplete
# artifact bundle: every response would otherwise be a 503, or worse a subset
# a caller could mistake for the whole.
CMD ["sh", "-c", "python scripts/check_artifacts.py && exec uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000"]
