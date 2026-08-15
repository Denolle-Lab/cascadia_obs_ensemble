# Manuscript (Quarto → Seismica LaTeX → Overleaf)

The Cascadia offshore-catalog paper is authored here in Quarto and auto-synced to
the Overleaf-linked repo **[Denolle-Lab/Offshore-Cascadia-Ensemble-Catalog](https://github.com/Denolle-Lab/Offshore-Cascadia-Ensemble-Catalog)**,
which co-authors see and edit in Overleaf. Mirrors the `codameter` → `codameter-paper`
setup, adapted to the **Seismica** class.

## What you edit

| File | Contents |
|------|----------|
| **`main.qmd`** | the paper **body** (Markdown + `[@cite]` + raw-LaTeX figure blocks) — this is what you iterate on |
| `_seismica_head.tex` | Seismica preamble + **title / authors / affiliations / CRediT** (title/shorttitle/reporttype/abstract come from `main.qmd` front matter) |
| `mybibfile.bib` | references (BibTeX; name is load-bearing — `\bibliography{mybibfile}`) |
| `figures/` | manuscript figures (source of truth; synced to Overleaf) |
| `supplementary_section.tex` | supplement, carried verbatim |

`seismica.cls`, `abbrvnat_seismica_upcasetitle.bst`, `banner.png`, `empty.png` are the
vendored Seismica journal kit.

## Build locally

```sh
pixi install --environment paper      # once: Quarto + tectonic (LaTeX)
make paper                            # or: pixi run -e paper python paper/build.py
```
`build.py` renders `main.qmd` → a body-only `main_body.tex`, assembles
`main.tex` = `_seismica_head.tex` (title/abstract substituted from the front matter)
+ body + `_seismica_tail.tex`, and compiles `main.pdf` with tectonic. Only
`main.tex` (and the class/bib/figures) is committed + synced; `main.pdf` is a local
preview — Overleaf compiles the shared PDF.

> **Why body-include?** `seismica.cls` loads its own hyperref/geometry/natbib/fonts,
> which clash with Quarto's default LaTeX preamble. So Quarto renders only the body;
> the Seismica preamble/title lives in `_seismica_head.tex`.

## Sync to Overleaf

On every push to `main` that changes the built paper, the GitHub Action
`.github/workflows/sync-paper-to-overleaf.yml` copies `main.tex`, `mybibfile.bib`,
the class files and `figures/` to the Overleaf-linked repo (as `cascadia-sync[bot]`);
Overleaf's GitHub Sync then shows them. Co-author edits made in Overleaf are surfaced
back here as GitHub issues (label `overleaf-sync`) by a workflow on the paper repo.

**One-time setup (secrets) is documented in the pull request that added this.**
