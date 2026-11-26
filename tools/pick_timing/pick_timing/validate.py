"""validate: checks of a corrected pick file against its input and the clock model.

1. byte identity: every row without a time change keeps its original text byte for byte (the appended columns
   aside); rows that changed differ only in the time and travel-time fields, and only at stations of the model.
2. residual re-fit: for each station of the model, the network-relative residuals before and after removing c(t)
   (rd - c), screened again with the detector, and the size of the same model refitted with the epoch times fixed
   (should be ~0). The locations used for the residuals were made with the uncorrected picks and absorbed part of
   the clock error, so a residual-only check leaves an opposite-sign remainder; ``--residuals-after`` takes the
   residuals of a re-location with the corrected picks (relocate-nlloc), where the drift should vanish.
3. co-located station slope test (stations within ``colocated_km``): pick time difference A - B of the same event
   and phase against the correction difference; slope -1 before and 0 after when the correction is right.
4. CC: station dt_cc minus the median of the other stations of the pair, before and after removing
   c(t1) - c(t2); and triangle closure of the CC times (unchanged by construction: a check of the bookkeeping).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import changepoint as cp
from .clockmodel import evaluate, load_model
from .detect import detect_series, huber_slope, station_frames
from .tables import load, provenance, write_json, write_table

DAY = 86400.0


def byte_identity(cfg, picks_in, picks_out, model):
    """Row-by-row comparison. A row may change only in the time / travel-time fields, only by the correction
    written in its own appended columns (clock_correction_s + a newly added timing_correction_s), within 1 us."""
    from .apply import _time_ns
    spec = cfg["picks"]
    cols = spec.get("columns", {})
    a = Path(picks_in).read_text().splitlines()
    b = Path(picks_out).read_text().splitlines()
    out = dict(rows_in=len(a) - 1, rows_out=len(b) - 1)
    hdr_a = [h.strip() for h in a[0].split(",")]
    hdr_b = [h.strip() for h in b[0].split(",")]
    out["header_prefix_identical"] = b[0].startswith(a[0])
    out["new_columns"] = hdr_b[len(hdr_a):]
    na = len(hdr_a)
    it = hdr_a.index(cols.get("time", "time"))
    time_idx = {hdr_a.index(n) for n in (cols.get("time", "time"), cols.get("travel_time", "")) if n in hdr_a}
    i_cc = hdr_b.index("clock_correction_s")
    i_tc = hdr_b.index("timing_correction_s") if "timing_correction_s" in out["new_columns"] else None
    ident, changed, bad, max_err, changed_rows = 0, 0, 0, 0.0, []
    for i in range(1, min(len(a), len(b))):
        fb = b[i].split(",")
        if ",".join(fb[:na]) == a[i]:
            ident += 1
            if float(fb[i_cc]) != 0 or (i_tc is not None and float(fb[i_tc]) != 0):
                bad += 1          # a correction was written but the time was not changed
            continue
        changed += 1
        changed_rows.append(i - 1)
        fa = a[i].split(",")
        diff = {k for k in range(na) if fa[k] != fb[k]}
        shift = float(fb[i_cc]) + (float(fb[i_tc]) if i_tc is not None else 0.0)
        err = abs((_time_ns(fb[it]) - _time_ns(fa[it])) / 1e9 - shift)
        max_err = max(max_err, err)
        if not diff <= time_idx or err > 1e-6 or shift == 0:
            bad += 1
    out.update(rows_identical=ident, rows_changed=changed, rows_changed_outside_rules=bad,
               max_shift_error_s=max_err, rows_dropped=max(0, len(a) - len(b)))
    return out, np.array(changed_rows, int)


def residual_refit(rd: pd.DataFrame, model: dict, p: dict, label="v5.1 residuals"):
    """Detector and fixed-epoch size before / after removing c(t), per model station."""
    rows = []
    frames = station_frames(rd[rd.station.isin(model["stations"])])
    for st, eps in model["stations"].items():
        if st not in frames:
            continue
        x = frames[st]
        ev = evaluate(model, np.full(len(x), st), x.t.values * DAY)
        for when, y in (("before", x.y.values), ("after", x.y.values - ev.clock_c_s.values)):
            sc, epochs = detect_series(x.t.values, y, p, seed=p["seed"], max_epochs=1)
            r = dict(station=st, residuals=label, when=when, n_picks=len(x), T=sc.get("T"), T_model=sc.get("T_model"),
                     flagged=bool(epochs), p_perm=sc.get("p_perm"))
            for e in eps:
                t1 = e["t_start_s"] / DAY
                t2 = e["t_end_s"] / DAY if e["t_end_s"] else np.inf
                g = cp.g_of(e["model"], x.t.values, t1, t2)
                win = (x.t.values >= t1 - 60) & (x.t.values < (t2 if np.isfinite(t2) else x.t.max()) + 60)
                a, se, n = huber_slope(g[win], y[win], scale=0.1, intercept=True)
                r[f"{e['epoch_id']}_size"] = a
                r[f"{e['epoch_id']}_size_se"] = se
                r[f"{e['epoch_id']}_size_model"] = e["size"]
                r[f"{e['epoch_id']}_n"] = n
            rows.append(r)
    return pd.DataFrame(rows)


def colocated_slopes(pin: pd.DataFrame, pout: pd.DataFrame, stations: pd.DataFrame, model, km=3.0, min_rows=20):
    """pin/pout: canonical pick tables (same row order). stations: station, x_km, y_km."""
    st = stations.dropna(subset=["x_km", "y_km"]).set_index("station")
    rows = []
    for s in model["stations"]:
        if s not in st.index:
            continue
        d = np.hypot(st.x_km - st.loc[s, "x_km"], st.y_km - st.loc[s, "y_km"])
        for o, dist in d[(d <= km) & (d.index != s)].items():
            for ph in ("P", "S"):
                A = pin[(pin.station == s) & (pin.phase.str.upper().str[0] == ph)][["event_id", "time_s"]]
                B = pin[(pin.station == o) & (pin.phase.str.upper().str[0] == ph)][["event_id", "time_s"]]
                Ao = pout.loc[A.index, ["event_id", "time_s"]]
                Bo = pout.loc[B.index, ["event_id", "time_s"]]
                m = A.merge(B, on="event_id", suffixes=("_a", "_b"))
                mo = Ao.merge(Bo, on="event_id", suffixes=("_a", "_b"))
                m = m.merge(mo, on="event_id", suffixes=("", "_o"))
                if len(m) < min_rows:
                    continue
                kappa = (m.time_s_a_o - m.time_s_a) - (m.time_s_b_o - m.time_s_b)
                if (kappa != 0).sum() < min_rows:
                    continue
                db = m.time_s_a - m.time_s_b
                da = m.time_s_a_o - m.time_s_b_o
                sb, seb, n = huber_slope(kappa.values, db.values, scale=0.1, intercept=True)
                sa, sea, _ = huber_slope(kappa.values, da.values, scale=0.1, intercept=True)
                rows.append(dict(station=s, other=o, sep_km=float(dist), phase=ph, n=n, n_corrected=int((kappa != 0).sum()),
                                 slope_before=sb, slope_before_se=seb, slope_after=sa, slope_after_se=sea,
                                 expected="-1 before, 0 after"))
    return pd.DataFrame(rows)


def cc_checks(cc: pd.DataFrame, model: dict):
    if cc is None or not len(cc):
        return pd.DataFrame(), {}
    d = cc[cc.accepted.astype(bool)] if "accepted" in cc.columns else cc
    if "pair" not in d.columns:
        d = d.assign(pair=d.event1.astype(str) + "-" + d.event2.astype(str))
    ex = set(model["stations"])
    cons = d[~d.station.isin(ex)].groupby("pair").dt_cc.agg(["median", "size"])
    rows = []
    for s in model["stations"]:
        x = d[d.station == s].copy()
        if not len(x):
            continue
        x["cons"] = x.pair.map(cons["median"])
        x = x[x.pair.map(cons["size"]).fillna(0) >= 5]
        c1 = evaluate(model, np.full(len(x), s), x.time1_s.values).clock_c_s.values
        c2 = evaluate(model, np.full(len(x), s), x.time2_s.values).clock_c_s.values
        y = (x.dt_cc - x.cons).values
        straddle = (c1 - c2) != 0
        for lab, m in (("rows with c(t1) != c(t2)", straddle), ("other rows", ~straddle)):
            if not m.any():
                continue
            rows.append(dict(station=s, rows=lab, n=int(m.sum()),
                             median_before_s=float(np.median(y[m])), median_abs_before_s=float(np.median(np.abs(y[m]))),
                             median_after_s=float(np.median((y - (c1 - c2))[m])),
                             median_abs_after_s=float(np.median(np.abs(y - (c1 - c2))[m])),
                             frac_gt_50ms_before=float(np.mean(np.abs(y[m]) > 0.05)),
                             frac_gt_50ms_after=float(np.mean(np.abs((y - (c1 - c2))[m]) > 0.05))))
    # triangle closure at the model stations (bookkeeping: unchanged by a clock correction)
    tri = {}
    for s in model["stations"]:
        x = d[d.station == s]
        for ph, g in x.groupby("phase"):
            dt = {}
            tt = {}
            for r in g.itertuples():
                dt[(r.event1, r.event2)] = r.dt_cc
                dt[(r.event2, r.event1)] = -r.dt_cc
                tt[r.event1], tt[r.event2] = r.time1_s, r.time2_s
            ev = sorted(tt)
            nb = {e: set() for e in ev}
            for (a, b) in dt:
                nb[a].add(b)
            cl_b, cl_a = [], []
            for a in ev:
                for b in nb[a]:
                    if b <= a:
                        continue
                    for c in nb[a] & nb[b]:
                        if c <= b:
                            continue
                        cb = dt[(a, b)] + dt[(b, c)] + dt[(c, a)]
                        cc_ = {e: evaluate(model, [s], [tt[e]]).clock_c_s.values[0] for e in (a, b, c)}
                        ca = (dt[(a, b)] - (cc_[a] - cc_[b])) + (dt[(b, c)] - (cc_[b] - cc_[c])) + \
                             (dt[(c, a)] - (cc_[c] - cc_[a]))
                        cl_b.append(cb)
                        cl_a.append(ca)
            if cl_b:
                tri[f"{s} {ph}"] = dict(n_triangles=len(cl_b), median_abs_closure_before_ms=float(np.median(np.abs(cl_b)) * 1e3),
                                        median_abs_closure_after_ms=float(np.median(np.abs(cl_a)) * 1e3),
                                        max_abs_change_ms=float(np.max(np.abs(np.array(cl_a) - np.array(cl_b))) * 1e3))
    return pd.DataFrame(rows), tri


def cli_validate(cfg, a):
    from .residuals import network_relative, station_xy
    out = Path(cfg["output_dir"]) / "validate"
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.clock_model)
    p = dict(cfg["detect"])
    res = {}
    print("1. byte identity ...", flush=True)
    bi, changed_rows = byte_identity(cfg, a.picks_in, a.picks_out, model)
    res["byte_identity"] = bi
    print(f"   {bi}", flush=True)
    print("2. residual re-fit ...", flush=True)
    rd_path = Path(cfg["output_dir"]) / "rd.parquet"
    tabs = []
    if cfg.get("residuals"):
        rdat = load(cfg, "residuals")
        rdat = station_xy(rdat, load(cfg, "stations") if cfg.get("stations") else None)
        ev = rdat[rdat.station.isin(model["stations"])].event_id.unique()
        sub = rdat[rdat.event_id.isin(ev)]
        rd = network_relative(sub, p, workers=cfg["workers"])
        tabs.append(residual_refit(rd, model, p, "input residuals (locations from uncorrected picks)"))
        if a.residuals_after:
            ra = load(dict(cfg, residuals=dict(cfg["residuals"], path=a.residuals_after)), "residuals")
            ra = station_xy(ra, load(cfg, "stations") if cfg.get("stations") else None)
            merged = pd.concat([sub[~sub.event_id.isin(ra.event_id.unique())], ra[ra.event_id.isin(ev)]],
                               ignore_index=True)
            rda = network_relative(merged, p, workers=cfg["workers"])
            t2 = residual_refit(rda, model, p, "re-located with corrected picks")
            # in the re-located residuals the picks are already corrected: 'before' is the corrected state
            t2["when"] = t2["when"].map({"before": "corrected picks, re-located", "after": "corrected twice (ignore)"})
            tabs.append(t2[t2.when == "corrected picks, re-located"])
    rf = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    write_table(rf, out / "residual_refit.csv")
    print(rf.to_string(), flush=True)
    print("3. co-located station slope test ...", flush=True)
    pin = load(dict(cfg, picks=dict(cfg["picks"], path=a.picks_in)), "picks")
    spec_out = dict(cfg["picks"], path=a.picks_out)
    pout = load(dict(cfg, picks=spec_out), "picks")
    stt = load(cfg, "stations") if cfg.get("stations") else None
    if stt is not None and not {"x_km", "y_km"} <= set(stt.columns):
        lat0, lon0 = np.nanmedian(stt.lat), np.nanmedian(stt.lon)
        stt = stt.assign(x_km=(stt.lon - lon0) * 111.195 * np.cos(np.radians(lat0)), y_km=(stt.lat - lat0) * 111.195)
    km = a.colocated_km if a.colocated_km else cfg["validate"]["colocated_km"]
    print(f"   station pairs within {km} km", flush=True)
    sl = colocated_slopes(pin, pout, stt, model, km, cfg["validate"]["slope_min_rows"]) \
        if stt is not None else pd.DataFrame()
    write_table(sl, out / "colocated_slopes.csv")
    print(sl.to_string() if len(sl) else "   no co-located station pairs with enough common picks", flush=True)
    print("4. CC checks ...", flush=True)
    cc = None
    spec = dict(cfg.get("cc") or {})
    if a.cc:
        spec["path"] = a.cc
    if spec.get("path"):
        cc = load(dict(cfg, cc=spec), "cc")
    cct, tri = cc_checks(cc, model)
    write_table(cct, out / "cc_check.csv")
    print(cct.to_string() if len(cct) else "   no CC rows", flush=True)
    print(f"   triangles: {tri}", flush=True)
    res.update(residual_refit=rf.to_dict("records"), colocated=sl.to_dict("records"), cc=cct.to_dict("records"),
               cc_triangles=tri)
    write_json(provenance({"picks_in": a.picks_in, "picks_out": a.picks_out, "clock_model": a.clock_model,
                           "residuals_after": a.residuals_after}, res), out / "validate.json")
    print(f"-> {out}", flush=True)
    return 0


def cli_relocate(cfg, a):
    """Re-locate, with NonLinLoc, the events that have a pick at a model station with c != 0 (validation only)."""
    from .nlloc import relocate_subset
    from .residuals import network_relative  # noqa: F401
    model = load_model(a.clock_model)
    res = load(dict(cfg, residuals=dict(cfg["residuals"], path=a.residuals or cfg["residuals"]["path"])), "residuals")
    r = res[res.station.isin(model["stations"])]
    ev = evaluate(model, r.station.values, r.time_s.values)
    r = r.assign(c=ev.clock_c_s.values)
    hit = r[r.c != 0]
    events = hit.event_id.unique()
    if a.max_events:
        events = events[: a.max_events]
    hit = hit[hit.event_id.isin(events)]
    tiles = res[res.event_id.isin(events)].drop_duplicates("event_id")
    tiles = tiles.assign(tile=tiles.hyp_file.str.split("/").str[0])
    te = {t: [f"{a.id_prefix}{e}" for e in g.event_id] for t, g in tiles.groupby("tile")}
    shifts = {(f"{a.id_prefix}{x.event_id}", x.station, x.phase): -x.c for x in hit.itertuples()}
    out = Path(cfg["output_dir"]) / "relocate"
    print(f"relocate-nlloc: {len(events)} events in {len(te)} tiles, {len(shifts)} shifted picks -> {out}", flush=True)
    ra = relocate_subset(a.run_dir, te, shifts, str(out / "runs"), a.nlloc, a.id_prefix, workers=cfg["workers"])
    p = write_table(ra, out / "residuals_relocated.parquet")
    write_json(provenance({"clock_model": a.clock_model}, {"run_dir": a.run_dir, "events": int(len(events)),
                                                          "tiles": len(te), "shifted_picks": len(shifts),
                                                          "relocated_events": int(ra.event_id.nunique()) if len(ra) else 0}),
               out / "relocate.provenance.json")
    print(f"{ra.event_id.nunique() if len(ra) else 0} of {len(events)} events re-located -> {p}", flush=True)
    return 0
