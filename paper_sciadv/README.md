# Science Advances draft (single paper)

One paper: the catalog (methods, tests, technical development) and what it shows about the faults and
seismic behavior of the Juan de Fuca plate system. Started 2026-11-22.

- `main.qmd`: title, abstract, Introduction, Results, Discussion, Materials and Methods, planned figures.
- `supplementary.qmd`: technical development and tests (Text S1–S6), supplementary tables.
- `references.bib`: every entry resolved from its DOI at doi.org (Crossref or DataCite), with authors,
  title, journal, volume, pages. Rebuild with
  `python utils/build_bib.py --out paper_sciadv/references.bib --sources <files with verified DOIs> paper_sciadv/extra_dois.txt`.
  `references_doi_map.csv` maps DOI to key and source; `references_failed.csv` lists any DOI that did not
  resolve. Add new references to `extra_dois.txt` (one DOI per line), never by hand. One entry
  (`Anon2025`, a GSA abstract) has no author in its DOI record.
- The earlier Seismica draft of the catalog (`paper/main.qmd`) is the source for Text S5.

Markers: `[TO UPDATE]` changes with the 3D relocation (catalog v5); `[TO ADD]` waits on a running
analysis (Mc maps); `[TO WRITE]` is a section still in outline.
