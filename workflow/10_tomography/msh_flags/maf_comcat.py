"""Check (c) of the rule: are the candidate events in ComCat/ANSS? (after scoring; also the post-hoc waveform-only list)

For every certain, probable and mixed event and for every box event deeper than 30 km: nearest ANSS event (cache of
the magnitude piece, 2010-2016, all types) within 30 s of the origin time and 150 km of the epicentre (v6.0, else v4).
Output: WORK/comcat_check.csv, counts printed and added to WORK/summary.json under "comcat_check".
"""
import json

import numpy as np
import pandas as pd

import maf_common as C

ANSS = "/wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag/anss_2010-16.csv"
DT, DKM = 30.0, 150.0


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def main():
    a = pd.read_csv(ANSS)
    a["ts"] = pd.to_datetime(a.time, utc=True, format="ISO8601").astype("int64") / 1e9
    a = a.sort_values("ts").reset_index(drop=True)
    ev = pd.read_csv(C.WORK / "events_scored.csv", low_memory=False)
    ev["ts"] = pd.to_datetime(ev.t_origin, utc=True, format="ISO8601").astype("int64") / 1e9
    fw = C.OUT / "msh_analog_waveform_only.csv"
    wo = set(pd.read_csv(fw).ID) if fw.exists() else set()
    ev.loc[ev.ID.isin(wo) & (ev.cls == "none"), "cls"] = "waveform_only"
    sel = ev[(ev.cls != "none") | (ev.in_box & (ev.depth > 30))].copy()
    lat = sel.lat.where(sel.lat.notna(), sel.v4_lat)
    lon = sel.lon.where(sel.lon.notna(), sel.v4_lon)
    rows = []
    for (i, r), la, lo in zip(sel.iterrows(), lat, lon):
        j0, j1 = np.searchsorted(a.ts.values, [r.ts - DT, r.ts + DT])
        best = None
        for j in range(j0, j1):
            d = hav(la, lo, a.latitude.values[j], a.longitude.values[j])
            if d <= DKM and (best is None or abs(a.ts.values[j] - r.ts) < abs(best[1])):
                best = (j, a.ts.values[j] - r.ts, d)
        row = dict(ID=r.ID, cls=r.cls, in_box=r.in_box, depth=r.depth, M=r.M)
        if best:
            b = a.iloc[best[0]]
            row.update(anss_id=b.id, anss_type=b["type"], anss_mag=b.mag, anss_depth=b.depth, dt_s=round(best[1], 2),
                       dist_km=round(best[2], 1))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(C.WORK / "comcat_check.csv", index=False)
    out["matched"] = out.anss_id.notna() if "anss_id" in out else False
    res = {}
    for k, g in out.groupby(np.where(out.cls != "none", out.cls, "none_deep_box")):
        res[k] = dict(n=int(len(g)), anss_match=int(g.matched.sum()))
    print(json.dumps(res, indent=1))
    print(out[out.matched & (out.cls != "none")].to_string())
    s = json.load(open(C.WORK / "summary.json"))
    s["comcat_check"] = res
    json.dump(s, open(C.WORK / "summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
