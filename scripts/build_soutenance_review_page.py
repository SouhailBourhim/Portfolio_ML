"""build_soutenance_review_page.py — a reviewable web page of the vulgarised deck.

The deck is reviewed by people who will not open PowerPoint: the supervisor, the
two other team members, and whoever rehearses the timing. This builds a single
HTML page carrying every slide image beside its speaker note and its spoken
budget, so a review is a link rather than a 1.5 MB attachment.

The notes are read back out of the .pptx itself rather than re-typed here, for
the same reason the report chapters are generated: two copies of the same
sentence drift, and the one on the projector is the one that matters.

PREREQUISITE — slide images. python-pptx cannot rasterise slides. Export them
first (PowerPoint, Windows):

    powershell -c "$p=New-Object -ComObject PowerPoint.Application; \
      $d=$p.Presentations.Open('<abs path to .pptx>',$true,$false,$false); \
      $d.Export('<abs out dir>','PNG',1400,788); $d.Close(); $p.Quit()"

or with LibreOffice: `soffice --headless --convert-to pdf` then `pdftoppm -png`.

Usage:
    python scripts/build_soutenance_review_page.py <slide-png-dir>
"""

from __future__ import annotations

import html
import re
import shutil
import sys
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "output" / "presentation" / "Soutenance_PFA_Vulgarisee_FR.pptx"
OUT = ROOT / "output" / "presentation" / "revue_soutenance"

# Where each act of the deck begins. The deck's own eyebrows say the same thing
# slide by slide; this is the grouping a reviewer navigates by.
SECTIONS = [
    (1, "Ouverture", "Le cadrage et la feuille de route"),
    (3, "1 · Le vocabulaire", "Les notions sans lesquelles la suite ne se suit pas"),
    (6, "2 · Le problème", "Pourquoi répartir un capital est difficile"),
    (10, "3 · Notre réponse", "La démarche, la chaîne, les garde-fous"),
    (16, "4 · Les résultats", "Ce que le dispositif établit"),
    (22, "Clôture", "L’explicabilité, le livrable, les suites"),
    (26, "Annexes", "À garder en réserve pour les questions"),
]

DUREE = re.compile(r"\[~(\d+)\s*s\]")


def slide_images(src: Path) -> list[Path]:
    """PowerPoint writes Slide1.PNG … Slide25.PNG — sort numerically, not by name."""
    found = sorted(src.glob("Slide*.PNG"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
    if not found:
        found = sorted(src.glob("*.png"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not found:
        raise SystemExit(f"no slide images in {src} — see the export command in this file's docstring")
    return found


def build(src: Path) -> Path:
    prs = Presentation(str(DECK))
    notes = [s.notes_slide.notes_text_frame.text.strip() for s in prs.slides]
    images = slide_images(src)
    if len(images) != len(notes):
        raise SystemExit(f"{len(images)} images but {len(notes)} slides — re-export the deck")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slides").mkdir(exist_ok=True)
    for i, img in enumerate(images, 1):
        shutil.copyfile(img, OUT / "slides" / f"{i:02d}.png")

    section_at = {n: (titre, sous) for n, titre, sous in SECTIONS}
    total = sum(int(m.group(1)) for note in notes for m in [DUREE.search(note)] if m)

    rows: list[str] = []
    for i, note in enumerate(notes, 1):
        if i in section_at:
            titre, sous = section_at[i]
            rows.append(
                f'<h2 class="act" id="acte-{i}"><span>{html.escape(titre)}</span>'
                f'<small>{html.escape(sous)}</small></h2>'
            )
        m = DUREE.search(note)
        secondes = int(m.group(1)) if m else None
        corps = html.escape(DUREE.sub("", note).strip())
        badge = (f'<span class="budget">{secondes // 60} min {secondes % 60:02d}</span>'
                 if secondes and secondes >= 60 else
                 f'<span class="budget">{secondes} s</span>' if secondes else "")
        rows.append(f"""<article class="slide" id="s{i}">
  <div class="shot"><img src="slides/{i:02d}.png" alt="Diapositive {i}" loading="lazy"></div>
  <div class="note">
    <header><span class="num">{i:02d}</span>{badge}</header>
    <p>{corps}</p>
  </div>
</article>""")

    nav = "".join(
        f'<a href="#acte-{n}">{html.escape(titre)}</a>' for n, titre, _ in SECTIONS
    )

    page = TEMPLATE.format(
        nav=nav,
        rows="\n".join(rows),
        n=len(notes),
        minutes=total // 60,
        secondes=total % 60,
    )
    (OUT / "index.html").write_text(page, encoding="utf-8")
    return OUT / "index.html"


TEMPLATE = """<title>Revue de soutenance Portfolio ML</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@500&display=swap">
<style>
  :root {{
    --ground: #F6F7F9;
    --surface: #FFFFFF;
    --ink: #10151F;
    --muted: #5A6475;
    --line: #DFE3EA;
    --navy: #1B3A6B;
    --accent: #2F6FE0;
    --amber: #9A6310;
    color-scheme: light;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground: #0E1219;
      --surface: #161B25;
      --ink: #E8EBF0;
      --muted: #97A1B2;
      --line: #262D3A;
      --navy: #9EC0F5;
      --accent: #7FA9F0;
      --amber: #D9A24A;
      color-scheme: dark;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground: #0E1219;
    --surface: #161B25;
    --ink: #E8EBF0;
    --muted: #97A1B2;
    --line: #262D3A;
    --navy: #9EC0F5;
    --accent: #7FA9F0;
    --amber: #D9A24A;
    color-scheme: dark;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 16px;
    line-height: 1.6;
  }}
  .wrap {{ max-width: 1120px; margin-inline: auto; padding-inline: 20px; padding-block: 0 72px; }}

  header.top {{ border-bottom: 1px solid var(--line); padding-block: 40px 28px; }}
  .eyebrow {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: var(--accent);
    margin: 0 0 12px;
  }}
  h1 {{
    font-family: Archivo, "Helvetica Neue", Arial, sans-serif;
    font-weight: 700; font-size: clamp(28px, 4.4vw, 44px); line-height: 1.1;
    letter-spacing: -.02em; margin: 0 0 14px; text-wrap: balance;
  }}
  .lede {{ margin: 0; max-width: 62ch; color: var(--muted); font-size: 17px; }}

  .facts {{ display: flex; flex-wrap: wrap; gap: 28px; margin-top: 26px; }}
  .facts div {{ display: flex; flex-direction: column; gap: 2px; }}
  .facts b {{
    font-family: Archivo, Arial, sans-serif; font-size: 26px; font-weight: 600;
    color: var(--navy); font-variant-numeric: tabular-nums; line-height: 1.1;
  }}
  .facts span {{ font-size: 13px; color: var(--muted); }}

  nav.acts {{
    position: sticky; top: 0; z-index: 5;
    display: flex; gap: 4px; overflow-x: auto; scrollbar-width: thin;
    background: color-mix(in srgb, var(--ground) 92%, transparent);
    backdrop-filter: blur(6px);
    border-bottom: 1px solid var(--line);
    padding-block: 10px;
  }}
  nav.acts a {{
    flex: 0 0 auto; text-decoration: none; color: var(--muted);
    font-size: 13.5px; padding: 5px 11px; border-radius: 999px; white-space: nowrap;
  }}
  nav.acts a:hover, nav.acts a:focus-visible {{ color: var(--ink); background: var(--surface); }}

  h2.act {{
    display: flex; flex-direction: column; gap: 2px;
    margin: 56px 0 20px; padding-bottom: 10px;
    border-bottom: 2px solid var(--navy);
    font-family: Archivo, Arial, sans-serif; font-weight: 600; font-size: 21px;
    letter-spacing: -.01em;
  }}
  h2.act small {{ font-family: "Source Sans 3", sans-serif; font-weight: 400; font-size: 14px; color: var(--muted); }}

  .slide {{
    display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr);
    gap: 24px; align-items: start;
    padding-block: 22px;
    border-top: 1px solid var(--line);
  }}
  .slide:first-of-type {{ border-top: none; }}
  .shot {{
    background: var(--surface); border: 1px solid var(--line); border-radius: 3px;
    overflow: hidden; line-height: 0;
  }}
  .shot img {{ width: 100%; max-width: 100%; height: auto; display: block; }}

  .note header {{ display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px; }}
  .num {{
    font-family: "IBM Plex Mono", monospace; font-size: 13px; color: var(--accent);
    font-variant-numeric: tabular-nums;
  }}
  .budget {{
    font-family: "IBM Plex Mono", monospace; font-size: 11.5px; color: var(--muted);
    border: 1px solid var(--line); border-radius: 3px; padding: 1px 6px;
    font-variant-numeric: tabular-nums;
  }}
  .note p {{ margin: 0; font-size: 15.5px; max-width: 46ch; }}

  footer {{ margin-top: 56px; padding-top: 22px; border-top: 1px solid var(--line); color: var(--muted); font-size: 14px; }}
  footer p {{ max-width: 70ch; }}

  @media (max-width: 760px) {{
    .slide {{ grid-template-columns: 1fr; gap: 14px; }}
    .note p {{ max-width: none; }}
  }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
</style>

<div class="wrap">
  <header class="top">
    <p class="eyebrow">Projet de fin d’année · INPT × EURAFRIC Information</p>
    <h1>Revue de la présentation de soutenance</h1>
    <p class="lede">Version vulgarisée, destinée à un jury sans bagage financier : chaque diapositive
      apparaît ici avec le texte à dire et son budget de parole. À lire avant la répétition.</p>
    <div class="facts">
      <div><b>{n}</b><span>diapositives, annexes comprises</span></div>
      <div><b>{minutes} min {secondes:02d}</b><span>de parole cumulée, hors questions</span></div>
      <div><b>4</b><span>actes : vocabulaire, problème, réponse, résultats</span></div>
    </div>
  </header>

  <nav class="acts">{nav}</nav>

  {rows}

  <footer>
    <p>Les diapositives sont exportées depuis
      <code>output/presentation/Soutenance_PFA_Vulgarisee_FR.pptx</code> et les notes sont lues
      depuis ce même fichier : cette page ne contient aucun texte saisi séparément. Tout chiffre
      affiché sur une diapositive provient d’un artefact versionné du dépôt.</p>
    <p>Prototype de recherche académique — ni conseil financier, ni recommandation, ni exécution d’ordres.</p>
  </footer>
</div>
"""


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(f"wrote {build(Path(sys.argv[1])).relative_to(ROOT)}")
