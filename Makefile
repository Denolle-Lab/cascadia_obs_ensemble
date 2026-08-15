# Manuscript build entrypoints (see paper/README.md).
#
# The single content source is paper/main.qmd (body) + paper/_seismica_head.tex
# (Seismica preamble/title/authors/abstract). `make paper` renders and assembles
# paper/main.tex + main.pdf locally; CI never renders — it only syncs main.tex,
# mybibfile.bib, the class files and figures to the Overleaf-linked paper repo.
.PHONY: paper check clean-paper

# main.qmd (+ head) -> main.tex + main.pdf, via the lean `paper` pixi env.
paper:
	pixi run -e paper python paper/build.py

# Non-rendering staleness guard.
check:
	@if [ paper/main.qmd -nt paper/main.tex ]; then \
	  echo "stale: paper/main.tex older than main.qmd -> make paper"; exit 1; \
	else echo "paper/main.tex up to date"; fi

clean-paper:
	cd paper && rm -f main_body.tex main.pdf *.aux *.log *.blg *.bbl *.xdv *.out *.fls *.fdb_latexmk main.synctex.gz
