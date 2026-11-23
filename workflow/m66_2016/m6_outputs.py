"""M66 fix, step 6: patch rows and picks in catalog v4 format, a record section of the Mw 6.6 and a location map.

Patch policy: our NLLoc epicentres are biased west (all picks are Pn/Sn beyond 156 km, azimuthal gap
255-310 deg, no station west of the events) and their depths go to the bottom of every search range, so the
patch rows carry the ComCat hypocentre and magnitude and our associated picks. Our NLLoc solutions are in
nll/locations.csv (copied here as nll_locations.csv) for comparison.

Outputs (data/catalog_v4/fixes/m66_2016/): catalog_v4_patch_2016.csv, picks_v4_patch_2016.csv,
nll_locations.csv, assoc_<event>.csv, fig_m66_record_section.png, fig_m66_locations.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
from obspy import UTCDateTime, read

import m4_associate as A

OUT = Path("/wd1/mdenolle_data/m66_2016")
FIX = Path(__file__).resolve().parents[2] / "data/catalog_v4/fixes/m66_2016"
NEW_ID = {"mw66": 900001, "gorda1": 900002, "gorda2": 900003}


def main():
    L = pd.read_csv(OUT / "nll/locations.csv")
    L.to_csv(FIX / "nll_locations.csv", index=False, float_format="%.4f")
    cat, picks = [], []
    for ev, (cid, ot, lat, lon, dep, mag, mt, win, v4) in A.EVENTS.items():
        a = pd.read_csv(OUT / "assoc" / f"{ev}_assoc.csv")
        a.to_csv(FIX / f"assoc_{ev}.csv", index=False, float_format="%.4f")
        eid = NEW_ID.get(ev, v4)
        l1 = L[(L.event == ev) & (L.model == "1d_iasp91")].iloc[0]
        if ev in NEW_ID:
            cat.append({"ID": eid, "raw_ID": "", "datetime": UTCDateTime(ot).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "latitude": lat, "longitude": lon, "depth(km)": dep, "magnitude": mag,
                        "Num_p_picks": int((a.phase == "P").sum()), "Num_s_picks": int((a.phase == "S").sum()),
                        "spatial_sigma(m)": round(float(l1.dist_to_comcat_km) * 1000, 1), "detection_value": "",
                        "location_source": f"ComCat {cid} ({mt} {mag}); NLLoc 1D epicentre {l1.dist_to_comcat_km:.1f} km away"})
        for r in a.itertuples():
            picks.append({"event_ID": eid, "raw_event_ID": "", "pick_datetime": UTCDateTime(r.pick_time).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                          "travel_time(s)": round(r.travel_time, 4), "station": f"{r.station}.{r.network}", "phase": r.phase,
                          "amplitude(mm)": "", "pick_prob": round(r.max_prob, 4)})
    pd.DataFrame(cat).to_csv(FIX / "catalog_v4_patch_2016.csv", index=False)
    pd.DataFrame(picks).to_csv(FIX / "picks_v4_patch_2016.csv", index=False)
    print(len(cat), "catalog rows;", len(picks), "picks")
    record_section()
    loc_map(L)


def record_section():
    cid, ot, lat, lon, dep, mag, mt, win, _ = A.EVENTS["mw66"]
    ot = UTCDateTime(ot)
    a = pd.read_csv(OUT / "assoc" / "mw66_assoc.csv")
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY="7p", FONT_LABEL="8p", MAP_FRAME_TYPE="plain")
    fig.basemap(region=[0, 120, 140, 410], projection="X12c/14c",
                frame=["WSne", "xa20f5+lseconds after the ComCat origin time", "ya50f10+ldistance (km)"])
    done = set()
    for r in a.sort_values("dist_km").itertuples():
        if (r.network, r.station) in done:
            continue
        done.add((r.network, r.station))
        f = OUT / "waveforms" / win / f"{r.network}.{r.station}.mseed"
        if not f.exists():
            continue
        st = read(str(f)).select(channel="??Z") or read(str(f)).select(channel="??3")
        if not st:
            continue
        tr = st.merge(fill_value=0)[0].copy()
        tr.detrend("demean"); tr.filter("bandpass", freqmin=2, freqmax=8)
        tr.trim(ot, ot + 120)
        if tr.stats.npts < 100:
            continue
        t = np.arange(tr.stats.npts) * tr.stats.delta + (tr.stats.starttime - ot)
        y = tr.data / (np.abs(tr.data).max() or 1) * 1.6 + r.dist_km
        fig.plot(x=t[::2], y=y[::2], pen="0.2p,gray30")
    for ph, col in (("P", "#0072B2"), ("S", "#D55E00")):
        g = a[a.phase == ph]
        fig.plot(x=g.travel_time, y=g.dist_km, style="y0.25c", pen=f"1p,{col}")
        d = np.linspace(140, 410, 50)
        tp = np.interp(d, g.dist_km.sort_values(), (g.sort_values("dist_km").travel_time - g.sort_values("dist_km").res_iasp91))
        fig.plot(x=tp, y=d, pen=f"0.6p,{col},4_2")
    fig.text(position="TL", offset="0.15c/-0.15c", justify="TL", font="8p,Helvetica-Bold",
             text=f"8 Dec 2016 Mw 6.6 ({cid}): {len(done)} stations, {(a.phase == 'P').sum()} P (blue), {(a.phase == 'S').sum()} S (orange)",
             fill="white@20")
    fig.text(position="BR", offset="-0.15c/0.15c", justify="BR", font="6p", text="vertical, 2-8 Hz, normalized; dashed: iasp91", fill="white@20")
    fig.savefig(FIX / "fig_m66_record_section.png", dpi=300)


def loc_map(L):
    reg = [-128.6, -121.5, 38.4, 42.6]
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY="7p", FORMAT_GEO_MAP="ddd.xF", MAP_FRAME_TYPE="plain")
    fig.coast(region=reg, projection="M12c", land="gray92", water="#dbe7ee", shorelines="0.3p,gray40", frame=["WSne", "af"])
    st = pd.concat([pd.read_csv(OUT / "assoc" / f"{e}_assoc.csv") for e in A.EVENTS]).drop_duplicates(["network", "station"])
    sta = pd.concat([pd.read_csv(FIX / f"stations_requested_{w}.csv") for w in
                     ("w20161208_mw66", "w20161208_m47", "w20160130_gorda")]).drop_duplicates(["network", "station"])
    sta = sta.merge(st[["network", "station"]], on=["network", "station"])
    fig.plot(x=sta.lon, y=sta.lat, style="i0.16c", fill="white", pen="0.3p,gray20")
    cols = {"1d_iasp91": "#009E73", "3d_trackA_iter1": "#D55E00", "1d_iasp91_z0-20": "#56B4E9", "3d_trackA_iter1_z0-20": "#CC79A7"}
    for ev, (cid, ot, lat, lon, dep, mag, mt, win, v4) in A.EVENTS.items():
        for m, c in cols.items():
            r = L[(L.event == ev) & (L.model == m)]
            if len(r):
                r = r.iloc[0]
                fig.plot(x=[lon, r.lon], y=[lat, r.lat], pen=f"0.4p,{c}")
                fig.plot(x=[r.lon], y=[r.lat], style="c0.14c", fill=c, pen="0.2p,black")
        fig.plot(x=[lon], y=[lat], style="a0.35c", fill="gold", pen="0.4p,black")
        fig.text(x=lon, y=lat, text=f"{ev} M{mag}", font="6p,Helvetica-Bold", justify="LB", offset="0.15c/0.1c")
    fig.legend(spec=None) if False else None
    fig.text(position="TR", offset="-0.15c/-0.15c", justify="TR", font="6p", fill="white@20",
             text="star: ComCat; green: NLLoc 1D; red: 3D; blue/pink: depth 0-20 km; triangles: stations with picks")
    fig.savefig(FIX / "fig_m66_locations.png", dpi=300)


if __name__ == "__main__":
    main()
