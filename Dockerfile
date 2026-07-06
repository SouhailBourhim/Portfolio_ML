# Portfolio ML — reproducible environment for the Phase 1 pipeline and tests.
#
# The image pins OS + Python + every dependency, so "does it run on your
# machine" stops being a question. Data is NOT baked in: data/ is generated
# at runtime (or mounted from the host) because it is DVC-managed and
# refreshed daily — an image with frozen market data would go stale.
#
# Quick start (see also the Docker section in README.md):
#   docker compose run --rm test        # run the offline test suite
#   docker compose run --rm pipeline    # full Bronze→Silver→Gold (needs .env)
#   docker compose up notebook          # Jupyter on http://localhost:8888

FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependencies first, in their own layer, so code edits don't re-install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Zero-config default: prove the code works without any key or data
CMD ["python", "-m", "pytest", "tests/", "-q"]
