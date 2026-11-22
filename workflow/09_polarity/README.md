# 09_polarity — P first-motion polarities for catalog v4

Status (2026-11-22): testing AxialPolCap (FM_ML, Zhang et al., in prep.) on v4 Axial picks; nothing run at scale yet.

| file | what |
|---|---|
| `axialpolcap_torch.py` | PyTorch port of `FM_ML/06-models/PolarPicker_h5_20260308_154910.keras` (reads the weights with h5py; matches Keras 3 to 3e-6). `convert` / `selftest` subcommands. |
| `axial_test_extract.py` | Cuts raw Z windows (−3 s to +2 s, 100 Hz, counts) around v4 P picks at the OOI Axial stations, one pnwstore read per station-day. |
| `polarity_eval.py` | Filters (causal / zero-phase Butterworth), 64-sample crop, SNR, model-free first-motion reference. |
| `build_axial_test_notebook.py` | Builds `notebooks/polarity/axialpolcap_axial_v4_test.ipynb` (execute with nbconvert). |
| `PLAN.md`, `PLAN_CRITIQUE.md` | Plan for the full extraction and a critic's notes. Written before the filter-phase finding below. |

**Key finding.** The model's output sign depends on the phase of the 3–20 Hz band-pass: causal (one-pass) input gives
class 1 = up; zero-phase (two-pass) or unfiltered input inverts it. Working preprocessing: causal Butterworth order 4,
3–20 Hz, 100 Hz, 64 samples centred on P, max-normalised. See the notebook for all issues and the questions for the FM_ML authors.

Data paths (local, not in git): model weights and Keras reference in `/wd1/mdenolle_data/polarity/models/`, windows in
`/wd1/mdenolle_data/polarity/axial_test/`, station inventory `workflow/05_magnitude/station_inventory_v2.xml`.

Environment: default pixi env with `LD_LIBRARY_PATH=.pixi/envs/default/lib` (pandas needs the env's libstdc++) and
`PYTHONPATH=/wd1/mdenolle_data/pyext` for pnwstore (extraction only).
