# Manuscript build entrypoints (see paper/README.md).
#
# The single content source is paper/main.qmd (body) + paper/_seismica_head.tex
# (Seismica preamble/title/authors/abstract). `make paper` renders and assembles
# paper/main.tex + main.pdf locally; CI never renders — it only syncs main.tex,
# mybibfile.bib, the class files and figures to the Overleaf-linked paper repo.
.PHONY: paper figs figures catalog check clean-paper

# Run Python in the default env with its own libstdc++ first: the system one on the
# lab server is too old for pandas (same as the drivers in scale/). Harmless elsewhere.
PY = LD_LIBRARY_PATH=$(CURDIR)/.pixi/envs/default/lib pixi run python

# main.qmd (+ head) -> main.tex + main.pdf, via the lean `paper` pixi env.
paper:
	pixi run -e paper python paper/build.py

# Collect manuscript figures from the figure notebooks' outputs into paper/figures/.
# Needs the input data (see ./download_data.sh). Add --execute to (re)run the
# notebooks first: pixi run python paper/export_figures.py --execute
figs:
	$(PY) paper/export_figures.py

# Preferred-magnitude catalog from the Route A products (CLEANUP_PLAN §4):
# ComCat moment tensors -> preferred M -> event classes. Minutes.
catalog:
	cd workflow/05_magnitude && $(PY) phase18_moment_tensor_match.py
	cd workflow/05_magnitude && $(PY) phase19_preferred_magnitude.py
	cd workflow/06_analysis && $(PY) phase10_event_classification.py

# Redraw every script figure from the data products, then collect all figures into
# paper/figures/ (the notebook figures are collected as they are; see `figs`).
figures:
	$(PY) paper/export_figures.py --execute --scripts-only

# Non-rendering staleness guard.
check:
	@if [ paper/main.qmd -nt paper/main.tex ]; then \
	  echo "stale: paper/main.tex older than main.qmd -> make paper"; exit 1; \
	else echo "paper/main.tex up to date"; fi

clean-paper:
	cd paper && rm -f main_body.tex main.pdf *.aux *.log *.blg *.bbl *.xdv *.out *.fls *.fdb_latexmk main.synctex.gz
