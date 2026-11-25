"""M66 fix, step 5: NonLinLoc locations of the associated picks, 3D (Track A model, iter1 static
station terms where they exist) and 1D (iasp91) cross-check.

Settings follow Track A (workflow/08_relocation/nlloc_template.in on reloc/track-a): EDT_OT_WT,
LOCGAU 0.1 s + 2% of travel time, pick errors by ensemble-probability class (P 0.05/0.10/0.15 s,
S twice that). Search box: ComCat epicentre +-60 km, depth -5..45 km, 1 km cells. Two passes: pass 2
drops picks with |residual| > 2.5 s (P) or 4 s (S) in pass 1.

Output: /wd1/mdenolle_data/m66_2016/nll/<model>/<event>/, and locations.csv + phases.csv there.
"""
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from pyproj import Proj

import m4_associate as A

OUT = Path("/wd1/mdenolle_data/m66_2016")
B = Path("/wd1/mdenolle_data/reloc_v5")
NLL = B / "src/NonLinLoc/src/bin/NLLoc"
PROJ = Proj(proj="lcc", lat_0=45.0, lon_0=-125.8, lat_1=41.0, lat_2=49.0, ellps="WGS84")
ERR_P = {0: 0.15, 1: 0.10, 2: 0.05}
HALF, Z0, Z1 = 60.0, -5.0, 45.0
MODELS = {
    "3d_trackA_iter1": dict(time=OUT / "time3d/cascadia3d", stat=B / "runs/iter1_vpvs176/station_terms.in",
                            topo=f"LOCTOPO_SURFACE {B}/topo/topo_01m.bin 0"),
    "3d_trackA_noterms": dict(time=OUT / "time3d/cascadia3d", stat=None,
                              topo=f"LOCTOPO_SURFACE {B}/topo/topo_01m.bin 0"),
    "1d_iasp91": dict(time=OUT / "time1d/iasp91", stat=None, topo=""),
}
RES_MAX = {"P": 2.5, "S": 4.0}


def obs_lines(a):
    out = []
    for r in a.itertuples():
        t = UTCDateTime(r.pick_time)
        cls = 2 if r.max_prob >= 0.6 else 1 if r.max_prob >= 0.3 else 0
        err = ERR_P[cls] * (2 if r.phase == "S" else 1)
        out.append(f"{r.station}.{r.network:<6s} ?    ?    ? {r.phase:<6s} ? {t.strftime('%Y%m%d %H%M')} "
                   f"{t.second + t.microsecond / 1e6:7.4f} GAU {err:9.2e} -1.00e+00 -1.00e+00 -1.00e+00 1.0")
    return out


def control(m, obs, out, lat, lon, z0=None, z1=None):
    x, y = PROJ(lon, lat)
    x0, y0 = round(x / 1e3 - HALF), round(y / 1e3 - HALF)
    n = int(2 * HALF) + 1
    z0 = Z0 if z0 is None else z0; z1 = Z1 if z1 is None else z1
    nz = int(z1 - z0) + 1
    stat = f"INCLUDE {m['stat']}" if m["stat"] else ""
    return f"""CONTROL 1 54321
TRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0
LOCSIG Cascadia OBS ensemble, m66_2016 fix
LOCFILES {obs} NLLOC_OBS {m['time']} {out}
LOCHYPOUT SAVE_NLLOC_ALL NLL_FORMAT_VER_2
LOCGRID {n} {n} {nz} {x0:.1f} {y0:.1f} {z0:.1f} 1.0 1.0 1.0 PROB_DENSITY SAVE
LOCSEARCH OCT 20 20 5 0.01 50000 5000 0 1
LOCMETH EDT_OT_WT 9999.0 4 -1 -1 -1 -1 -1.0 1
LOCGAU 0.1 0.0
LOCGAU2 0.02 0.05 2.0
LOCPHASEID P P p Pg Pn
LOCPHASEID S S s Sg Sn
LOCQUAL2ERR 0.05 0.1 0.15 0.3 99999.9
LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9
LOCANGLES ANGLES_NO 5
{m['topo']}
{stat}
"""


def kv(tok):
    return {tok[i]: tok[i + 1] for i in range(len(tok) - 1)}


def parse_hyp(path):
    loc, ph, inph = {}, [], False
    for line in open(path):
        tok = line.split()
        if not tok:
            continue
        k = tok[0]
        if k == "NLLOC":
            loc["status"] = tok[2].strip('"')
            loc["message"] = " ".join(tok[3:])
        elif k == "GEOGRAPHIC":
            loc["time"] = f"{tok[2]}-{tok[3]}-{tok[4]}T{tok[5]}:{tok[6]}:{float(tok[7]):09.6f}"
            d = kv(tok[8:])
            loc.update(lat=float(d["Lat"]), lon=float(d["Long"]), depth=float(d["Depth"]))
        elif k == "QUALITY":
            d = kv(tok[1:])
            loc.update(rms=float(d["RMS"]), nphs=int(d["Nphs"]), gap=float(d["Gap"]), dist_min=float(d["Dist"]))
        elif k == "STATISTICS":
            d = kv(tok[1:])
            d = {k.replace("Cov", ""): v for k, v in d.items()}
            loc.update(sig_x=np.sqrt(float(d["XX"])), sig_y=np.sqrt(float(d["YY"])), sig_z=np.sqrt(float(d["ZZ"])),
                       ell_az1=float(d.get("EllAz1", "nan")),
                       ell_len1=float(d["Len1"]), ell_len2=float(d["Len2"]), ell_len3=float(d["Len3"]))
        elif k == "QML_OriginUncertainty":
            d = kv(tok[1:])
            loc.update(h_min=float(d["minHorUnc"]), h_max=float(d["maxHorUnc"]), h_az=float(d["azMaxHorUnc"]))
        elif k == "STAT_GEOG":
            d = kv(tok[1:])
            loc.update(exp_lat=float(d["ExpectLat"]), exp_lon=float(d["Long"]), exp_depth=float(d["Depth"]))
        elif k == "PHASE":
            inph = True
        elif k == "END_PHASE":
            inph = False
        elif inph:
            i = tok.index(">")
            ph.append(dict(code=tok[0], phase=tok[4], tt_pred=float(tok[i + 2]), res=float(tok[i + 3]),
                           weight=float(tok[i + 4]), sdist=float(tok[i + 8]), sazim=float(tok[i + 9]),
                           tcorr=float(tok[i + 13]) if len(tok) > i + 13 else np.nan))
    return loc, pd.DataFrame(ph)


def has_grid(model, code, phase):
    return Path(f"{MODELS[model]['time']}.{phase}.{code}.time.buf").exists()


def run(model, ev, a, lat, lon, tag, zfix=None):
    a = a[[has_grid(model, f"{s_}.{n}", p) for s_, n, p in zip(a.station, a.network, a.phase)]]   # 3D model ends near 38.6 N
    wd = OUT / "nll" / model / ev / tag
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "event.obs").write_text(f"PUBLIC_ID {ev}\n" + "\n".join(obs_lines(a)) + "\n\n")
    z0, z1 = zfix if zfix else (None, None)
    (wd / "nlloc.in").write_text(control(MODELS[model], wd / "event.obs", wd / "loc", lat, lon, z0, z1))
    with open(wd / "nlloc.log", "w") as log:
        subprocess.run([str(NLL), "nlloc.in"], cwd=wd, stdout=log, stderr=subprocess.STDOUT)
    return parse_hyp(wd / "last.hyp")


def main():
    locs, phs = [], []
    for ev, (cid, ot, lat, lon, dep, mag, mt, win, v4) in A.EVENTS.items():
        a0 = pd.read_csv(OUT / "assoc" / f"{ev}_assoc.csv")
        for model, zfix in [(m, None) for m in MODELS] + [(m, (0.0, 20.0)) for m in MODELS]:
            tagz = "" if zfix is None else "_z0-20"
            loc, ph = run(model, ev, a0, lat, lon, "pass1" + tagz, zfix)
            ph["code"] = ph.code.astype(str)
            bad = set(ph[ph.res.abs() > ph.phase.map(RES_MAX)][["code", "phase"]].itertuples(index=False, name=None))
            a = a0[[(f"{s}.{n}", p) not in bad for s, n, p in zip(a0.station, a0.network, a0.phase)]]
            loc, ph = run(model, ev, a, lat, lon, "pass2" + tagz, zfix)
            d, az, _ = gps2dist_azimuth(lat, lon, loc["lat"], loc["lon"])
            loc.update(event=ev, comcat_id=cid, model=model + tagz, depth_range=str(zfix) if zfix else "-5-45", n_assoc=len(a0), n_dropped_pass1=len(bad),
                       n_p=int((a.phase == "P").sum()), n_s=int((a.phase == "S").sum()),
                       comcat_lat=lat, comcat_lon=lon, comcat_depth=dep, dist_to_comcat_km=d / 1e3,
                       az_from_comcat=az, dt_origin_s=UTCDateTime(loc["time"]) - UTCDateTime(ot))
            locs.append(loc)
            ph["event"], ph["model"] = ev, model + tagz
            phs.append(ph)
            print(ev, model + tagz, {k: loc[k] for k in ("status", "lat", "lon", "depth", "rms", "nphs", "gap")},
                  f"d={d / 1e3:.1f} km dt={loc['dt_origin_s']:.2f} s dropped={len(bad)}", flush=True)
    L = pd.DataFrame(locs)
    L.to_csv(OUT / "nll/locations.csv", index=False, float_format="%.4f")
    pd.concat(phs).to_csv(OUT / "nll/phases.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
