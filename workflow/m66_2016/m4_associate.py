"""M66 fix, step 4: associate ELEP picks with the known ComCat origins.

For each target event and station, predicted P (first arrival) and S (first S) from TauP iasp91 at the
ComCat hypocentre; keep the highest-probability ELEP pick within the tolerance window
P: -3 s .. +3 s + 1% of distance (km->s), S: -5 s .. +5 s + 4% of distance (the window reaches Sg/Lg,
which at 150-400 km arrives 5-15 s after Sn). One P and one S per station and event.

Output: /wd1/mdenolle_data/m66_2016/assoc/<event>_assoc.csv and a GTSRCE file of all stations used.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read_inventory
from obspy.taup import TauPyModel

OUT = Path("/wd1/mdenolle_data/m66_2016")
FIX = Path(__file__).resolve().parents[2] / "data/catalog_v4/fixes/m66_2016"
EVENTS = {   # name: comcat id, origin, lat, lon, depth, mag, magtype, window, v4 id (if any)
    "mw66": ("us20007z6r", "2016-12-08T14:49:45.91", 40.4535, -126.1936, 8.45, 6.6, "Mww", "w20161208_mw66", None),
    "m47": ("us20007z7b", "2016-12-08T16:32:46.52", 40.4292, -126.3674, 3.33, 4.7, "Mwr", "w20161208_m47", 172114),
    "gorda1": ("us20004vq3", "2016-01-30T00:28:41.96", 40.9167, -127.2074, 10.0, 5.1, "Mww", "w20160130_gorda", None),
    "gorda2": ("us20004vq9", "2016-01-30T00:30:52.57", 41.0363, -127.3842, 10.0, 5.0, "Mww", "w20160130_gorda", None),
}
MAXDIST = 400.0
GTSRCE_TRACKA = Path("/wd1/mdenolle_data/reloc_v5/model_vpvs176/stations_GTSRCE.in")
model = TauPyModel("iasp91")


def first(arrs, phases):
    t = [a.time for a in arrs if a.name in phases]
    return min(t) if t else np.nan


def main():
    gts = {l.split()[1]: l for l in GTSRCE_TRACKA.read_text().splitlines()}
    used = {}
    for ev, (cid, ot, la, lo, dep, mag, mt, win, v4) in EVENTS.items():
        ot = UTCDateTime(ot)
        picks = pd.read_csv(OUT / "picks" / f"{win}_picks.csv")
        picks["t"] = picks.pick_time.map(UTCDateTime)
        sta = pd.read_csv(FIX / f"stations_requested_{win}.csv")
        rows = []
        for s in sta[sta.returned].itertuples():
            from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
            d, az, _ = gps2dist_azimuth(la, lo, s.lat, s.lon)
            dkm = d / 1e3
            if dkm > MAXDIST:
                continue
            arr = model.get_travel_times(max(dep, 0.1), kilometer2degrees(dkm), ["p", "P", "Pn", "s", "S", "Sn"])
            tp, ts = first(arr, {"p", "P", "Pn"}), first(arr, {"s", "S", "Sn"})
            g = picks[(picks.network == s.network) & (picks.station == s.station)]
            for ph, tpred, lo_, hi in (("P", tp, -3.0, 3.0 + 0.01 * dkm), ("S", ts, -5.0, 5.0 + 0.04 * dkm)):
                c = g[g.label == ph].copy()
                c["res"] = c.t.map(lambda t: t - ot) - tpred
                c = c[(c.res >= lo_) & (c.res <= hi)]
                if c.empty:
                    continue
                b = c.sort_values("max_prob", ascending=False).iloc[0]
                rows.append(dict(event=ev, comcat_id=cid, network=s.network, station=s.station,
                                 phase=ph, pick_time=str(b.t), travel_time=b.t - ot, pred_iasp91=tpred,
                                 res_iasp91=b.res, max_prob=b.max_prob, dist_km=dkm, az=az,
                                 band=b.band_inst, strong_motion_only=b.strong_motion_only,
                                 n_candidates=len(c), in_trackA=f"{s.station}.{s.network}" in gts))
                used[f"{s.station}.{s.network}"] = (s.lat, s.lon, s.elev_m)
        a = pd.DataFrame(rows).sort_values(["dist_km", "phase"])
        (OUT / "assoc").mkdir(exist_ok=True)
        a.to_csv(OUT / "assoc" / f"{ev}_assoc.csv", index=False, float_format="%.4f")
        print(ev, len(a), "picks:", (a.phase == "P").sum(), "P", (a.phase == "S").sum(), "S;",
              a.station.nunique(), "stations; P res iasp91 median", round(a[a.phase == "P"].res_iasp91.median(), 2))
    lines = []
    for code, (la, lo, el) in sorted(used.items()):
        lines.append(gts.get(code, f"GTSRCE {code} LATLON {la:.6f} {lo:.6f} {-el / 1e3:.4f} 0.0"))
    (OUT / "stations_GTSRCE_m66.in").write_text("\n".join(lines) + "\n")
    print(len(lines), "stations;", sum(c in gts for c in used), "with Track A GTSRCE")


if __name__ == "__main__":
    main()
