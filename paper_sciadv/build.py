#!/usr/bin/env python3
"""Build the Science Advances draft: Quarto markdown -> LaTeX for Overleaf (+ a PDF check).

  1. quarto renders main.qmd and supplementary.qmd to body-only LaTeX (template ../paper/_partials/body-only.tex);
  2. this script writes main.tex and supplementary.tex = a plain preamble (article, natbib, graphicx) with the
     title, teaser and abstract from the qmd front matter + the body + one figure environment per main figure
     (the "Figures (planned)" list gives the captions; the PNG files are in figures/) + the bibliography;
  3. tectonic compiles both as a check (main.pdf, supplementary.pdf are not committed).

main.tex, supplementary.tex, references.bib and figures/ are what the sync workflow copies to the Overleaf
repository (folder sciadv/). The .tex files are copied only when they do not exist there yet: once
colleagues edit on Overleaf, Overleaf is the text source. Figures and the bibliography are always synced.

    pixi run --manifest-path <repo>/pixi.toml -e paper python paper_sciadv/build.py [--no-pdf]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PARTIAL = HERE.parent / "paper" / "_partials" / "body-only.tex"
FIGS = {1: ["fig1_overview"], 2: ["fig2a_mc_three_methods", "fig2b_mc_periods"], 3: ["fig3_fault_surfaces"],
        4: ["fig4_blanco_strands"], 5: ["fig5_clustering"], 6: ["fig6_short_lived"], 7: ["fig7_tomography"]}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage{lmodern}
  \usepackage{textcomp}
  \DeclareUnicodeCharacter{2010}{-}
  \DeclareUnicodeCharacter{2011}{-}
\else
  \usepackage{fontspec}
\fi
\usepackage{graphicx}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{amsmath,amssymb}
\usepackage{booktabs,longtable,array,calc}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{lineno}
\linenumbers
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcommand{\passthrough}[1]{#1}
\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{3em}\sloppy
\title{%%TITLE%%}
\author{[Authors and affiliations to add]}
\date{Draft, generated from \texttt{paper\_sciadv/%%SRC%%} in Denolle-Lab/cascadia\_obs\_ensemble}
\begin{document}
\maketitle
"""


UNICODE = {"°": r"\textdegree{}", "≥": r"\ensuremath{\geq}", "≤": r"\ensuremath{\leq}", "±": r"\ensuremath{\pm}",
           "−": r"\ensuremath{-}", "σ": r"\ensuremath{\sigma}", "≈": r"\ensuremath{\approx}", "–": "--", "—": "---",
           "×": r"\ensuremath{\times}", "µ": r"\ensuremath{\mu}", "∼": r"\ensuremath{\sim}", "→": r"\ensuremath{\rightarrow}"}


def ascii_tex(t):
    """Plain-ASCII LaTeX so the file compiles with pdfLaTeX (Overleaf default) and XeLaTeX alike."""
    for k, v in UNICODE.items():
        t = t.replace(k, v)
    left = sorted({c for c in t if ord(c) > 127})
    if left:
        print("  non-ASCII left in the .tex:", "".join(left))
    return t


def run(cmd, cwd):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def front_matter(qmd):
    m = re.match(r"^---\n(.*?)\n---\n", qmd.read_text(encoding="utf-8"), re.S)
    return yaml.safe_load(m.group(1)) if m else {}


def tex_escape(s):
    return re.sub(r"([&%$#_{}])", r"\\\1", s)


def captions(qmd):
    """Captions from the '- **Fig. N.** ...' list of main.qmd (markdown -> minimal LaTeX)."""
    text = qmd.read_text(encoding="utf-8").split("# Figures (planned)", 1)[-1]
    out = {}
    for m in re.finditer(r"^- \*\*Fig\. (\d+)\.\*\*(.*?)(?=^- \*\*Fig\.|\Z)", text, re.S | re.M):
        c = " ".join(m.group(2).split())
        c = re.sub(r"\(`[^)]*`[^)]*\)\s*", "", c)               # drop "(`figures/...`, from `...`)" source notes
        c = tex_escape(c)
        out[int(m.group(1))] = c
    return out


def figures_tex(qmd):
    cap, parts = captions(qmd), ["\\clearpage\n\\section*{Figures}\n"]
    for n, files in FIGS.items():
        have = [f for f in files if (HERE / "figures" / f"{f}.png").exists()]
        if not have:
            parts.append(f"\\paragraph{{Fig. {n} (not yet drawn).}} {cap.get(n, '')}\n")
            continue
        imgs = "\n".join(f"\\includegraphics[width=\\textwidth]{{figures/{f}.png}}\\par" for f in have)
        parts.append(f"\\begin{{figure}}[p]\n\\centering\n{imgs}\n\\caption*{{\\textbf{{Fig. {n}.}} {cap.get(n, '')}}}\n\\end{{figure}}\n")
    return "\n".join(parts)


def build(name, with_figs, pdf):
    qmd = HERE / f"{name}.qmd"
    body = HERE / f"{name}_body.tex"
    run(["quarto", "render", qmd.name, "--to", "latex", "--template", str(PARTIAL)], HERE)
    meta = front_matter(qmd)
    head = PREAMBLE.replace("%%TITLE%%", tex_escape(str(meta.get("title", "")))).replace("%%SRC%%", qmd.name.replace("_", r"\_"))
    if meta.get("abstract"):
        head += "\\begin{abstract}\n" + tex_escape(" ".join(str(meta["abstract"]).split())) + "\n\\end{abstract}\n"
    if meta.get("teaser"):
        head += "\\noindent\\textbf{Teaser.} " + tex_escape(str(meta["teaser"])) + "\n\n"
    b = body.read_text(encoding="utf-8")
    if with_figs:                                   # the planned-figure list becomes the figure captions
        b = re.split(r"\\section\*?\{Figures \(planned\)\}", b)[0]
    tex = head + b
    if with_figs:
        tex += "\n" + figures_tex(qmd)
    tex += "\n\\bibliographystyle{unsrtnat}\n\\bibliography{references}\n\\end{document}\n"
    out = HERE / f"{name}.tex"
    out.write_text(ascii_tex(tex), encoding="utf-8")
    body.unlink()
    print(f"wrote {out.name} ({out.stat().st_size:,} bytes)")
    if pdf:
        if shutil.which("tectonic"):
            run(["tectonic", "--keep-logs", out.name], HERE)
        else:
            print("tectonic not found: PDF check skipped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    if shutil.which("quarto") is None:
        sys.exit("quarto not on PATH: run through `pixi run -e paper`")
    build("main", True, not a.no_pdf)
    build("supplementary", False, not a.no_pdf)


if __name__ == "__main__":
    main()
