#!/usr/bin/env python3
"""Null truth for the structure contrasts (D-B112; rule pre-registered in data/paper_plan/plate_boundaries/RESUME.md,
2026-11-27 10:13:48 PST, commit 6d45ad98): the calibration truth of the v0.4.2 ensemble with the contrasts removed near
the plate-boundary segments, and its synthetic data, in a NEW ensemble root.

  B2_ENS_CONFIG=v0.4.2 B2_ENS_ROOT=/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null \
      python pb_null_truth.py [--dry-run]

* prior/: copied from the calibration root (read only there); prior.json keeps its provenance records.
* truth model: b2_ens_synth.truth_model() (v0.4.2 tomography x (1 + f), truth_seed, start floor), then for each of the
  15 segments with contrasts on catalog v6.1 every rock node (v0.4.2 medium = 0) within 45 km (3-D distance to the
  segment's fault surfaces and seismicity, segment_node_dist.npz of pb_structure.py on v6.1) is set to the mean of those
  nodes in its depth layer, Vp and Vs separately; nodes within 45 km of several segments go to the nearest; then the
  start floor Vs >= Vp / 1.65 again.
* everything else: b2_ens_synth.truth() unchanged (the same hypocentre, statics, forward-offset and pick-noise draws;
  tomoDD-SE_syn of the frozen binaries per stage; synth/abs_synth_<stage>.parquet, dt_synth_<stage>.parquet).
Writes ROOT/synth/truth/null_truth.json (zones, node counts, changes). Refuses to run with the calibration root.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOMO = HERE.parent / "10_tomography"
CAL_ROOT = Path("/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2")
NULL_ROOT = Path("/wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null")
PB_V61 = Path("/wd1/mdenolle_data/paper_plan/v6.1")
SEGS = ["JR", "GR", "BL", "MFZ", "GP", "MTJ", "GS", "FA", "SIF", "IS", "PU", "MSH", "CAS", "OC", "NAPA"]
ZONE_KM = 45.0
FLOOR = 1.65


def zones(shape_zyx):
    """Zone index per node [z, y, x] (index into SEGS, -1 = outside every zone) and the distance to it."""
    d = np.load(PB_V61 / "segment_node_dist.npz")
    D = np.stack([d[s].reshape(shape_zyx) for s in SEGS])           # [seg, z, y, x], inf beyond 50 km
    k = np.argmin(D, axis=0)
    dm = np.take_along_axis(D, k[None], axis=0)[0]
    k = np.where(dm <= ZONE_KM, k, -1)
    return k, dm


def null_model(tm, medium_zyx):
    """tm: truth dict with vp, vs [lon, lat, z]. Returns the null vp, vs and a summary."""
    vp = np.transpose(tm["vp"], (2, 1, 0)).astype(float)            # [z, y, x]
    vs = np.transpose(tm["vs"], (2, 1, 0)).astype(float)
    zone, _ = zones(vp.shape)
    rock = medium_zyx == 0
    vp0, vs0 = vp.copy(), vs.copy()
    rows = []
    for i, s in enumerate(SEGS):
        n_s = 0
        for k in range(vp.shape[0]):
            sel = (zone[k] == i) & rock[k] & np.isfinite(vp[k]) & np.isfinite(vs[k])
            if not sel.any():
                continue
            vp[k][sel] = vp0[k][sel].mean()
            vs[k][sel] = vs0[k][sel].mean()
            n_s += int(sel.sum())
        rows.append(dict(seg=s, nodes=n_s))
    vs = np.where(vp / vs < FLOOR, vp / FLOOR, vs)                   # the members' start floor, as in the truth
    ch = (zone >= 0) & rock
    summ = dict(zone_km=ZONE_KM, segments=SEGS, nodes_per_segment=rows, nodes_changed=int(ch.sum()),
                dvp_abs_median=float(np.median(np.abs(vp - vp0)[ch])), dvp_abs_p95=float(np.percentile(np.abs(vp - vp0)[ch], 95)),
                dvs_abs_median=float(np.median(np.abs(vs - vs0)[ch])), dvs_abs_p95=float(np.percentile(np.abs(vs - vs0)[ch], 95)),
                floor_nodes=int(((vp / vs) <= FLOOR + 1e-9).sum()))
    return np.transpose(vp, (2, 1, 0)), np.transpose(vs, (2, 1, 0)), summ


def main():
    root = Path(os.environ.get("B2_ENS_ROOT", ""))
    if root.resolve() != NULL_ROOT or os.environ.get("B2_ENS_CONFIG") != "v0.4.2":
        raise SystemExit(f"set B2_ENS_CONFIG=v0.4.2 and B2_ENS_ROOT={NULL_ROOT} (got {root})")
    if root.resolve() == CAL_ROOT.resolve():
        raise SystemExit("refusing to write into the calibration root")
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "prior" / "prior.json").exists():
        shutil.copytree(CAL_ROOT / "prior", root / "prior")
        (root / "prior" / "COPIED_FROM.json").write_text(json.dumps(
            dict(source=str(CAL_ROOT / "prior"), copied=time.strftime("%F %T"), reason="D-B112 null truth: the same "
                 "prior ingredients as the calibration root"), indent=1))
    sys.path.insert(0, str(TOMO))
    import xarray as xr
    import b2_ens_synth as S
    assert S.C.ROOT.resolve() == NULL_ROOT, S.C.ROOT
    medium = xr.open_dataset("/wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo/v0.4.2.nc")["medium"].values
    orig = S.truth_model

    def truth_model_null():
        tm, fp, fs = orig()
        vp, vs, summ = null_model(tm, medium)
        tm = dict(tm, vp=vp, vs=vs)
        S.TD.mkdir(parents=True, exist_ok=True)
        (S.TD / "null_truth.json").write_text(json.dumps(dict(decision="D-B112", rule="RESUME.md 6d45ad98",
                                                             built=time.strftime("%F %T"), **summ), indent=1))
        print(json.dumps(summ, indent=1), flush=True)
        return tm, fp, fs
    if "--dry-run" in sys.argv:
        tm, fp, fs = orig()
        print(json.dumps(null_model(tm, medium)[2], indent=1))
        return
    S.truth_model = truth_model_null
    S.truth()


if __name__ == "__main__":
    main()
