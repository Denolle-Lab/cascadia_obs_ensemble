# 03 Relocation (GraphDD, external) and cross-correlation datasets

Relocation was run outside this repository with GraphDD (graph-based double-difference
relocation), by I. McBrearty and Y. Yu: `Cascadia_relocated_catalog_ver_3.csv` (63,887
events) and its picks (1,004,335). Its velocity and region inputs are `data/vel_*.csv` and
`data/nodes_*.csv`. GraphDD repository, commit and configuration: **TO CONFIRM** (see
[PROVENANCE.md](../../PROVENANCE.md), stages 3 and 5).

The code here builds what GraphDD needs for the cross-correlation refinement:
`create_waveform_datasets_{EH,HH_BH}_on_the_fly_in_bulk.py`: event waveform sets per
station for the CC differential times (checks in `notebooks/diagnostics/verify_*`).
