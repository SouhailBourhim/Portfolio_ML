"""Figures for Chapter 3 (architecture) of the internship report.

The pipeline diagram is READ FROM `dvc.yaml`, not drawn by hand: stages, their
dependency edges and the layer each one writes to are all derived. Adding a
stage to the pipeline updates the figure; a figure that disagrees with the
pipeline is therefore not possible.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "rapport" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREY = "#1F3864", "#595959"
LAYER_COLOR = {                      # medallion palette, kept close to the text
    "bronze": ("#F3E3D3", "#8C5A2B"),
    "silver": ("#E9ECEF", "#5A6570"),
    "gold":   ("#FBEFC8", "#9A7B10"),
    "model":  ("#E7EDF6", BLUE),
}
LAYER_LABEL = {"bronze": "Bronze — brut", "silver": "Silver — validé",
               "gold": "Gold — prêt pour la modélisation",
               "model": "Modélisation et publication"}

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.bbox": "tight",
                     "figure.dpi": 150})

# French one-liners: what each stage is FOR, for a reader who will never open
# dvc.yaml. Keyed by stage name — an unknown stage falls back to its own name,
# so a new stage shows up in the figure even before this table is updated.
ROLE = {
    "ingest": "téléchargement\ndes sources",
    "clean": "alignement,\nrendements log.",
    "features": "stationnarité,\nmacro décalée",
    "ml_features": "variables ML\ncausales",
    "phase2_hurdle": "référence\nclassique",
    "phase4_compare": "régimes +\ncovariance",
    "phase4b_compare": "signaux\nRF / XGBoost",
    "phase4c_compare": "coûts +\nrégularisation",
    "phase5_compare": "évaluation\nhors échantillon",
    "dashboard_data": "chiffres du\ntableau de bord",
}


def load_graph():
    """Return (stages, edges, layer_of) derived from dvc.yaml."""
    spec = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]

    def outs(v):
        for o in v.get("outs", []):
            yield list(o)[0] if isinstance(o, dict) else o

    producer = {o: name for name, v in spec.items() for o in outs(v)}
    edges = {(producer[d], name) for name, v in spec.items()
             for d in v.get("deps", []) if d in producer}

    def layer(name):
        paths = list(outs(spec[name]))
        for tag in ("bronze", "silver"):
            if any(f"data/{tag}/" in p for p in paths):
                return tag
        # A Gold *dataset* is a data-layer stage; a Gold *result* (json) is a
        # modelling stage that merely happens to land in the same directory.
        if all(p.endswith(".parquet") or "manifest" in p for p in paths):
            return "gold"
        return "model"

    return spec, edges, {n: layer(n) for n in spec}


def depths(stages, edges):
    """Longest-path depth of each stage — the row it is drawn on."""
    d = {n: 0 for n in stages}
    for _ in range(len(stages)):
        for a, b in edges:
            d[b] = max(d[b], d[a] + 1)
    return d


def pipeline():
    stages, edges, layer = load_graph()
    d = depths(stages, edges)
    rows = {}
    for n in stages:
        rows.setdefault(d[n], []).append(n)
    for r in rows:
        rows[r].sort()

    W, H, GAP, ROW = 2.75, 1.02, 0.55, 1.80   # box + spacing, in data units
    pos = {}
    for r, names in rows.items():
        span = len(names) * W + (len(names) - 1) * GAP
        for i, n in enumerate(names):
            pos[n] = (-span / 2 + W / 2 + i * (W + GAP), -r * ROW)

    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    for a, b in sorted(edges):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch(
            (x1, y1 - H / 2), (x2, y2 + H / 2),
            connectionstyle="arc3,rad=0.06" if x1 != x2 else "arc3,rad=0",
            arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#9AA3AD", zorder=1))

    for n, (x, y) in pos.items():
        face, edge = LAYER_COLOR[layer[n]]
        ax.add_patch(FancyBboxPatch(
            (x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.07",
            fc=face, ec=edge, lw=1.0, zorder=2))
        ax.text(x, y + 0.28, n, ha="center", va="center", fontsize=7.2,
                family="monospace", color=edge, zorder=3)
        ax.text(x, y - 0.19, ROLE.get(n, ""), ha="center", va="center",
                fontsize=6.4, color=GREY, linespacing=1.3, zorder=3)

    # Layer legend, ordered the way the medallion pattern is read.
    xs = [p[0] for p in pos.values()]
    for i, key in enumerate(("bronze", "silver", "gold", "model")):
        face, edge = LAYER_COLOR[key]
        yl = min(p[1] for p in pos.values()) - 1.25 - i * 0.38
        ax.add_patch(FancyBboxPatch((min(xs) - 1.05, yl - 0.11), 0.34, 0.22,
                                    boxstyle="round,pad=0.01,rounding_size=0.04",
                                    fc=face, ec=edge, lw=1.0))
        ax.text(min(xs) - 0.58, yl, LAYER_LABEL[key], va="center", fontsize=7,
                color=GREY)

    ax.set_title("Le pipeline DVC : dix étapes, un graphe de dépendances explicite",
                 fontsize=10, color=BLUE, pad=14)
    ax.set_xlim(min(xs) - 1.15, max(xs) + W / 2 + 0.35)
    ax.set_ylim(min(p[1] for p in pos.values()) - 2.85,
                max(p[1] for p in pos.values()) + 0.85)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "pipeline_dvc.pdf")
    plt.close(fig)
    print(f"  pipeline_dvc.pdf   ({len(stages)} étapes, {len(edges)} arêtes)")


if __name__ == "__main__":
    print("figures chapitre 3 :")
    pipeline()
