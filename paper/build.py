#!/usr/bin/env python3
"""Build the Cascadia catalog manuscript: Quarto markdown -> Seismica LaTeX -> PDF.

The single editable content source is ``paper/main.qmd`` (front matter + body).
Because ``seismica.cls`` is a heavy journal class that loads its own
hyperref/geometry/natbib/fonts (which clash with Quarto's default LaTeX
preamble), we do NOT let Quarto own the whole document. Instead:

  1. quarto renders ``main.qmd`` to a body-only ``main_body.tex`` (the format uses
     ``template: _partials/body-only.tex`` = just ``$body$``);
  2. this script assembles ``main.tex`` = ``_seismica_head.tex`` (Seismica preamble
     + title block, with title/shorttitle/reporttype/abstract substituted from the
     qmd front matter) + the rendered body + ``_seismica_tail.tex``
     (``\bibliography`` + ``\end{document}``);
  3. it compiles ``main.tex`` -> ``main.pdf`` with ``tectonic`` (XeTeX engine +
     bibtex; the Seismica class needs XeTeX for its ORCID ``\XeTeXLinkBox`` macro).
     If tectonic is absent it falls back to ``latexmk -xelatex`` (NOT ``-pdf``,
     which would fail on the XeTeX-only ORCID macro).

So you edit Markdown (main.qmd) for the body and _seismica_head.tex for
authors/affiliations; you get submission-ready main.tex + main.pdf. main.tex is
the file synced to the Overleaf-linked repo.

Usage:  python paper/build.py         (from repo root, ideally `pixi run -e paper ...`)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "main.qmd"
BODY = HERE / "main_body.tex"          # quarto output (output-file: main_body)
HEAD = HERE / "_seismica_head.tex"
TAIL = HERE / "_seismica_tail.tex"
OUT = HERE / "main.tex"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}  (in {cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def front_matter(qmd: Path) -> dict:
    text = qmd.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        sys.exit(f"error: no YAML front matter found in {qmd.name}")
    return yaml.safe_load(m.group(1)) or {}


def assemble(meta: dict) -> None:
    head = HEAD.read_text(encoding="utf-8")
    for key, ph in (("title", "%%TITLE%%"), ("shorttitle", "%%SHORTTITLE%%"),
                    ("reporttype", "%%REPORTTYPE%%"), ("abstract", "%%ABSTRACT%%")):
        head = head.replace(ph, str(meta.get(key, "")).strip())
    body = BODY.read_text(encoding="utf-8")
    tail = TAIL.read_text(encoding="utf-8")
    OUT.write_text(head + "\n\n" + body + "\n" + tail, encoding="utf-8")
    print(f"assembled {OUT.name} ({OUT.stat().st_size} bytes)")


def main() -> int:
    if shutil.which("quarto") is None:
        sys.exit("error: 'quarto' not found on PATH (use `pixi run -e paper ...` or install quarto).")
    if not SOURCE.exists():
        sys.exit(f"error: {SOURCE} not found.")

    run(["quarto", "render", SOURCE.name, "--to", "latex"], HERE)
    if not BODY.exists():
        sys.exit(f"error: expected body {BODY.name} not produced by quarto "
                 "(check `format.latex.output-file: main_body` in main.qmd).")

    assemble(front_matter(SOURCE))

    # Compile with tectonic (self-contained; runs xelatex + bibtex, auto-fetches
    # packages). latexmk is used instead only if a full TeX Live is on PATH.
    if shutil.which("tectonic"):
        run(["tectonic", "--keep-intermediates", "--keep-logs", OUT.name], HERE)
    elif shutil.which("latexmk"):
        # -xelatex (not -pdf): the Seismica class uses XeTeX-only macros (ORCID).
        run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", OUT.name], HERE)
    else:
        print("warning: no LaTeX engine (tectonic/latexmk) found; wrote main.tex only.")

    print("\nBuild complete:")
    for p in (OUT, HERE / "main.pdf"):
        print(f"  {'ok ' if p.exists() else 'MISSING '}{p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
