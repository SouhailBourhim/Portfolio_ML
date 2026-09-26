#!/usr/bin/env bash
# build_rapport_final.sh — compile the final PFA report to output/pdf/.
#
# WHY THIS EXISTS. MONTAGE.md documents `tectonic -X compile main.tex`, which
# assumes a TeX toolchain on the machine. Several of us do not have one, and the
# delivered PDF went a month out of date behind corrected sources because
# rebuilding it was not a one-liner anybody could run. This is that one-liner.
#
# It compiles inside a container, so nothing is installed on the host. Two
# passes are required: the first writes .aux/.toc, the second resolves the table
# of contents, the figure and table lists, and every \ref.
#
# Usage:
#     bash scripts/build_rapport_final.sh            # Docker (default)
#     ENGINE=tectonic bash scripts/build_rapport_final.sh   # local tectonic
#
# After building, run the guards that read the rendered PDF:
#     python -m pytest tests/test_final_report.py tests/test_rendered_pdf.py

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/docs/rapport_final"
OUT="$ROOT/output/pdf"
PDF_NAME="Rapport_PFA_Final_2026.pdf"
IMAGE="${IMAGE:-texlive/texlive:latest}"
ENGINE="${ENGINE:-docker}"

mkdir -p "$OUT"

case "$ENGINE" in
  tectonic)
    command -v tectonic >/dev/null || {
      echo "tectonic absent du PATH. Voir MONTAGE.md, ou utilisez ENGINE=docker." >&2
      exit 1
    }
    ( cd "$SRC" && tectonic -X compile main.tex )
    ;;
  docker)
    command -v docker >/dev/null || { echo "docker absent du PATH." >&2; exit 1; }
    docker image inspect "$IMAGE" >/dev/null 2>&1 || {
      echo "Image $IMAGE absente. Téléchargement (~4,5 Go)…" >&2
      docker pull "$IMAGE"
    }
    # --user keeps the generated files owned by the caller rather than root.
    docker run --rm \
      -v "$SRC":/doc \
      -w /doc \
      --user "$(id -u):$(id -g)" \
      "$IMAGE" \
      sh -c 'xelatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null &&
             xelatex -interaction=nonstopmode -halt-on-error main.tex'
    ;;
  *)
    echo "ENGINE inconnu : $ENGINE (attendu: docker | tectonic)" >&2
    exit 1
    ;;
esac

[ -f "$SRC/main.pdf" ] || { echo "La compilation n'a produit aucun main.pdf." >&2; exit 1; }
cp "$SRC/main.pdf" "$OUT/$PDF_NAME"
echo "écrit : output/pdf/$PDF_NAME ($(du -h "$OUT/$PDF_NAME" | cut -f1))"
