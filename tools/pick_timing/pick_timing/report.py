"""report: a self-contained HTML page for human review of clock candidates, and the review CSV.
ingest-review: the filled review CSV -> the confirmed-epoch CSV that model-clock reads.

The review CSV has one row per candidate epoch. A person fills ``decision`` (accept / reject / blank = undecided),
``reviewer`` and optionally ``size_source`` (cc or residuals: which estimate of the size to use),
``t_end_override_utc`` (e.g. the header restart that ends a drift) and ``note``. Only accepted rows become
confirmed epochs. Nothing is applied to picks without a person's accept.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import SIGN_CONVENTION
from . import changepoint as cp
from .tables import posix_to_iso, read_table, to_posix

DAY = 86400.0
REVIEW_COLS = ["candidate_id", "station", "net", "model", "t_start_utc", "sd_t_start_days", "t_end_utc",
               "t_end_lo_utc", "t_end_hi_utc", "restart_utc", "reset_at_restart", "a", "sd_a", "a_units", "cc_a",
               "cc_a_sd", "cc_n", "amp_s", "dbic", "p_perm", "n_picks", "n_bins_in", "a_P", "a_S", "coloc_ratio",
               "coloc_n", "evidence", "decision", "size_source", "t_end_override_utc", "reviewer", "decision_ref",
               "note"]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _dt(days):
    return pd.to_datetime(np.asarray(days, float) * DAY, unit="s")


def candidate_figure(c: pd.Series, x: pd.DataFrame, bins: pd.DataFrame | None, restarts=None, gaps=None, cc=None,
                     boots=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = dict(model=c.model, a=c.a, b=c.b, c1=c.t_start_days,
             c2=c.t_end_days if np.isfinite(c.t_end_days) else np.inf)
    t0 = f["c1"]
    t1 = f["c2"] if np.isfinite(f["c2"]) else x.t.max()
    pad = max(60.0, 0.6 * (t1 - t0))
    lo, hi = t0 - pad, t1 + pad
    xx = x[(x.t >= lo) & (x.t <= hi)]
    ncol = 2 if (cc is not None and len(cc)) else 1
    fig, axs = plt.subplots(2, ncol, figsize=(11 if ncol == 2 else 9, 6), squeeze=False,
                            gridspec_kw=dict(height_ratios=[2.2, 1]))
    ax = axs[0, 0]
    for ph, col in (("P", "#1f77b4"), ("S", "#ff7f0e")):
        z = xx[xx.phase1 == ph]
        ax.plot(_dt(z.t), z.y.clip(-1.2, 1.2), ".", ms=2, alpha=0.35, color=col, label=f"{ph} ({len(z)})")
    if bins is not None:
        bb = bins[(bins.t >= lo) & (bins.t <= hi)]
        ax.errorbar(_dt(bb.t), bb.y, yerr=np.sqrt(bb.s2 + c.tau_s ** 2), fmt="o", ms=3, color="k", lw=0.6,
                    label="bin median")
    tt = np.linspace(lo, hi, 2000)
    ax.plot(_dt(tt), cp.predict(f, tt), "r-", lw=1.5, label=f"{c.model}: residual fit")
    if np.isfinite(c.get("cc_a", np.nan)):
        ax.plot(_dt(tt), f["b"] + c.cc_a * cp.g_of(c.model, tt, f["c1"], f["c2"]), "g--", lw=1.2,
                label=f"size from CC ({c.cc_a:+.4g})")
    if boots is not None and len(boots):
        ax.axvspan(_dt(f["c1"] - np.nanstd(boots[:, 0])), _dt(f["c1"] + np.nanstd(boots[:, 0])), color="r",
                   alpha=0.12, lw=0)
    for r in (restarts if restarts is not None else []):
        if lo <= r <= hi:
            ax.axvline(_dt(r), color="m", lw=0.8, ls=":")
    for g0, g1 in (gaps if gaps is not None else []):
        if g1 >= lo and g0 <= hi:
            ax.axvspan(_dt(max(g0, lo)), _dt(min(g1, hi)), color="0.85", lw=0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("network-relative residual (s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title(f"{c.candidate_id}  {c.model}  start {c.t_start_utc[:16]}  end {str(c.t_end_utc)[:16] or 'open'}  "
                 f"size {c.a:+.4g} {c.a_units}  dBIC {c.dbic:.0f}  (magenta: header restarts, grey: data gaps)",
                 fontsize=8)
    # co-located pairs / P vs S summary
    ax = axs[1, 0]
    z = xx.copy()
    z["g"] = cp.g_of(c.model, z.t.values, f["c1"], f["c2"])
    for ph, col in (("P", "#1f77b4"), ("S", "#ff7f0e")):
        q = z[z.phase1 == ph]
        ax.plot(q.g, q.y.clip(-1.2, 1.2), ".", ms=2, alpha=0.3, color=col)
    gg = np.linspace(z.g.min(), z.g.max(), 10) if len(z) else np.array([0, 1])
    ax.plot(gg, f["b"] + c.a * gg, "r-", lw=1)
    ax.set_xlabel("model shape g(t) (days into drift, or 1 inside an offset epoch)")
    ax.set_ylabel("residual (s)")
    ax.set_title(f"P slope {c.get('a_P', np.nan):+.4g}, S slope {c.get('a_S', np.nan):+.4g}; co-located event pairs: "
                 f"ratio {c.get('coloc_ratio', np.nan):.2f} (n={int(c.get('coloc_n', 0) or 0)})", fontsize=8)
    ax.grid(alpha=0.3)
    if ncol == 2:
        ax = axs[0, 1]
        ax.plot(cc.g, cc.y.clip(-1.2, 1.2), ".", ms=4, alpha=0.6)
        gg = np.linspace(min(cc.g.min(), 0), max(cc.g.max(), 0), 10)
        ax.plot(gg, c.cc_a * gg, "g-", label=f"CC fit {c.cc_a:+.4g} +- {c.cc_a_sd:.2g}")
        ax.plot(gg, c.a * gg, "r:", label=f"residual fit {c.a:+.4g}")
        ax.set_xlabel("g(t1) - g(t2)")
        ax.set_ylabel("dt_cc - median of other stations (s)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_title(f"CC differential times at {c.station} ({len(cc)} rows)", fontsize=8)
        ax = axs[1, 1]
        ax.plot(_dt(cc.t1), cc.y - c.cc_a * cc.g, ".", ms=4, alpha=0.6, label="after model")
        ax.plot(_dt(cc.t1), cc.y, "x", ms=3, alpha=0.4, label="before")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-1, 1)
        ax.legend(fontsize=7)
        ax.set_title("CC rows vs time of event 1", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in [axs[0, 0]] + ([axs[1, 1]] if ncol == 2 else []):
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _png(fig)


def _cc_rows(cc, station, c, exclude):
    if cc is None:
        return None
    d = cc
    if "accepted" in d.columns:
        d = d[d.accepted.astype(bool)]
    if "pair" not in d.columns:
        d = d.assign(pair=d.event1.astype(str) + "-" + d.event2.astype(str))
    cons = d[~d.station.isin(exclude)].groupby("pair").dt_cc.agg(["median", "size"])
    x = d[d.station == station].copy()
    if not len(x):
        return None
    x["cons"] = x.pair.map(cons["median"])
    x = x[x.pair.map(cons["size"]).fillna(0) >= 5]
    c2 = c.t_end_days if np.isfinite(c.t_end_days) else np.inf
    x["t1"], x["t2"] = x.time1_s / DAY, x.time2_s / DAY
    x["g"] = cp.g_of(c.model, x.t1, c.t_start_days, c2) - cp.g_of(c.model, x.t2, c.t_start_days, c2)
    x["y"] = x.dt_cc - x.cons
    return x[x.g != 0]


def fa_table(fa: dict) -> str:
    rows = []
    fmt = lambda v, f: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, f)  # noqa: E731
    for k, v in fa.items():
        rows.append(f"<tr><td>{html.escape(k)}</td><td>{v.get('n')}</td><td>{v.get('flagged')}</td>"
                    f"<td>{fmt(v.get('rate'), '.4f')}</td><td>{fmt(v.get('rate_upper95'), '.4f')}</td>"
                    f"<td>{fmt(v.get('T_p95'), '.1f')}</td><td>{fmt(v.get('T_max'), '.1f')}</td>"
                    f"<td>{fmt(v.get('recovery_rate'), '.3f')}</td></tr>")
    return ("<table><tr><th>run</th><th>stations</th><th>flagged</th><th>rate</th><th>95 % upper</th>"
            "<th>T 95th pct</th><th>T max</th><th>recovery</th></tr>" + "".join(rows) + "</table>")


CSS = """body{font-family:system-ui,sans-serif;margin:16px;max-width:1200px;color:#222;background:#fff}
table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}
th{background:#f0f0f0}td:first-child{text-align:left}h2{border-bottom:1px solid #ccc;margin-top:28px}
.card{border:1px solid #ddd;padding:8px;margin:12px 0}.ok{color:#060}.warn{color:#a50}img{max-width:100%}
code{background:#f4f4f4;padding:1px 3px}"""


def build_report(det_dir, out_html, out_csv, accept=(), decided_by="", restarts=None, cc=None, confirmed=None,
                 title="Station clock candidates", max_stations=200, notes="", decision_ref=""):
    det = Path(det_dir)
    cands = pd.read_csv(det / "candidates.csv") if (det / "candidates.csv").exists() and \
        (det / "candidates.csv").stat().st_size > 1 else pd.DataFrame()
    screen = pd.read_csv(det / "screen_stations.csv")
    fa = json.loads((det / "false_alarm.json").read_text()) if (det / "false_alarm.json").exists() else {}
    prov = json.loads((det / "detect.provenance.json").read_text()) if (det / "detect.provenance.json").exists() else {}
    rd = read_table(det / "rd.parquet")
    rd = rd[np.isfinite(rd.rd)]
    rd["t"] = rd.time_s / DAY
    rd["y"] = rd.rd - rd.groupby(["station", "phase1"]).rd.transform("median")
    # confirmed elsewhere: candidates at stations/times covered by confirmed epochs are listed, not reviewed again
    conf = pd.read_csv(confirmed) if confirmed else pd.DataFrame(columns=["station", "t_start_utc", "t_end_utc"])
    is_conf = np.zeros(len(cands), bool)
    if len(cands) and len(conf):
        for r in conf.itertuples():
            s0 = to_posix([r.t_start_utc])[0] / DAY
            s1 = to_posix([r.t_end_utc])[0] / DAY if isinstance(r.t_end_utc, str) and r.t_end_utc else np.inf
            m = (cands.station == r.station) & (cands.t_start_days < s1) & \
                (cands.t_end_days.fillna(np.inf) > s0)
            is_conf |= m.values
    rv = cands.copy()
    for col in REVIEW_COLS:
        if col not in rv.columns:
            rv[col] = ""
    acc = set(a for a in accept if a)
    for i, r in rv.iterrows():
        if r.candidate_id in acc or r.station in acc:
            rv.at[i, "decision"] = "accept"
            rv.at[i, "reviewer"] = decided_by
            rv.at[i, "decision_ref"] = decision_ref
            rv.at[i, "size_source"] = "cc" if np.isfinite(pd.to_numeric(r.get("cc_a"), errors="coerce")) else "residuals"
            if r.get("reset_at_restart") in (True, "True") and r.get("restart_utc"):
                rv.at[i, "t_end_override_utc"] = r.restart_utc
    # precise start time for ingest
    if len(rv):
        rv["t_start_utc"] = [posix_to_iso(d * DAY, 3) for d in rv.t_start_days]
        rv["t_end_utc"] = [posix_to_iso(d * DAY, 3) if np.isfinite(d) else "" for d in rv.t_end_days]
    review = rv[~is_conf][REVIEW_COLS] if len(rv) else pd.DataFrame(columns=REVIEW_COLS)
    review.to_csv(out_csv, index=False)
    # ---------------- html
    net_tab = screen.assign(net=screen.station.str.split(".").str[-1]).groupby("net").agg(
        stations=("station", "size"), screened=("status", lambda s: int((s == "screened").sum())),
        candidates=("candidate", lambda s: int(s.fillna(False).astype(bool).sum())),
        picks=("n_picks", "sum")).reset_index()
    parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>",
             f"<p>Detector output: <code>{html.escape(str(det))}</code>. Sign convention: {html.escape(SIGN_CONVENTION)}."
             f" {notes}</p>",
             "<h2>1. Screen</h2>",
             f"<p>{len(screen)} stations; {int((screen.status == 'screened').sum())} screened (enough picks and bins); "
             f"{int(screen.candidate.fillna(False).astype(bool).sum())} with candidate epochs, "
             f"{len(cands)} candidate epochs in total; {int(is_conf.sum())} of them are covered by confirmed epochs.</p>",
             net_tab.to_html(index=False),
             "<h2>2. False alarms and detection power</h2>",
             "<p>Same detector (pre-screen, permutation test, thresholds) on stations without a clock problem: "
             "synthetic stations with each real station's pick times and noise level, and each real station with its "
             "residuals shuffled in time. Injection: drifts (rate s/day : duration days) added to shuffled real data; "
             "recovered = flagged as a drift with the start within 25 % of the duration and the rate within 50 %."
             "</p>", fa_table(fa.get("summary", fa)) if fa else "<p>not run</p>",
             f"<p>Thresholds: {html.escape(json.dumps({k: prov.get('detect_params', {}).get(k) for k in ('dbic_min', 'amp_min_s', 'p_max', 'n_perm', 'neighbor_km', 'min_picks')}))}</p>"]
    if len(cands):
        show = cands.assign(confirmed=is_conf)
        cols = ["candidate_id", "model", "t_start_utc", "t_end_utc", "a", "a_units", "amp_s", "dbic", "p_perm",
                "cc_a", "a_P", "a_S", "coloc_ratio", "coloc_n", "reset_at_restart", "evidence", "confirmed"]
        cols = [c for c in cols if c in show.columns]
        parts += ["<h2>3. Candidate epochs</h2>",
                  "<p>evidence: <b>colocated</b> = pairs of nearby events inside/outside the epoch show the same jump "
                  "(a clock survives the pair difference, a change in the event mix does not); <b>P=S</b> = P and S "
                  "move by the same amount; <b>reset@restart</b> = the epoch ends at a MiniSEED data restart; "
                  "<b>CC</b> = cross-correlation differential times confirm the size. A person decides.</p>",
                  show[cols].to_html(index=False, float_format=lambda v: f"{v:.4g}")]
        cc_station_excl = set(cands.station)
        parts.append("<h2>4. Per candidate</h2>")
        for k, c in cands.head(max_stations).iterrows():
            x = rd[rd.station == c.station]
            tag = c.candidate_id.replace("#", "_")
            bins = pd.read_csv(det / "series" / f"{tag}_bins.csv") if (det / "series" / f"{tag}_bins.csv").exists() else None
            boots = pd.read_csv(det / "series" / f"{tag}_boot.csv").values if (det / "series" / f"{tag}_boot.csv").exists() else None
            rr = restarts.get(c.station) if restarts else None
            ccr = _cc_rows(cc, c.station, c, cc_station_excl) if cc is not None else None
            try:
                img = candidate_figure(c, x, bins, rr, None, ccr, boots)
            except Exception as e:  # noqa: BLE001
                img = None
                print(f"figure {c.candidate_id}: {e}")
            st = "confirmed (see confirmed epochs)" if is_conf[k] else "to review"
            parts.append(f"<div class='card'><h3>{html.escape(c.candidate_id)} <small>{st}</small></h3>"
                         f"<p>{html.escape(str(c.evidence))}; start {c.t_start_utc} (sd {c.sd_t_start_days:.2f} d), "
                         f"end {c.t_end_utc} [{c.t_end_lo_utc} .. {c.t_end_hi_utc}], restart "
                         f"{c.get('restart_utc', '')}; size {c.a:+.5g} +- {c.sd_a:.2g} {c.a_units} (residuals), "
                         f"CC {c.get('cc_a', np.nan):+.5g} +- {c.get('cc_a_sd', np.nan):.2g} "
                         f"(n={c.get('cc_n', 0)}); dBIC {c.dbic:.1f}, p {c.p_perm:.3g}</p>"
                         + (f"<img src='{img}'>" if img else "") + "</div>")
    parts.append("<h2>5. How to review</h2><p>Open the review CSV, set <code>decision</code> to accept or reject per "
                 "candidate, fill <code>reviewer</code>; optionally <code>size_source</code> (cc/residuals) and "
                 "<code>t_end_override_utc</code>. Then run <code>python -m pick_timing ingest-review</code> and "
                 "<code>model-clock</code>. Nothing is applied without an accept.</p></body></html>")
    Path(out_html).write_text("\n".join(parts))
    return review


def cli_report(cfg, a):
    from .detect import load_restarts
    from .tables import load
    det = Path(cfg["output_dir"])
    cc = None
    if cfg.get("cc") and cfg["cc"].get("path"):
        try:
            cc = load(cfg, "cc")
        except Exception as e:  # noqa: BLE001
            print(f"CC table not used: {e}")
    restarts = load_restarts(getattr(a, "restarts", None)) if getattr(a, "restarts", None) else None
    rv = build_report(det, det / "report.html", det / "review_candidates.csv", a.accept.split(","), a.decided_by,
                      restarts=restarts, cc=cc, confirmed=getattr(a, "confirmed", None), max_stations=a.max_stations,
                      title=a.title, notes=html.escape(a.notes), decision_ref=a.decision_ref)
    print(f"report {det / 'report.html'}; review CSV {det / 'review_candidates.csv'} ({len(rv)} rows, "
          f"{int((rv.decision == 'accept').sum())} pre-accepted)")
    return 0


def ingest(review_csv, out_csv):
    r = pd.read_csv(review_csv, dtype=str).fillna("")
    acc = r[r.decision.str.strip().str.lower() == "accept"]
    rows = []
    for x in acc.itertuples():
        use_cc = x.size_source.strip().lower() == "cc" and x.cc_a not in ("", "nan")
        size = float(x.cc_a) if use_cc else float(x.a)
        sd = float(x.cc_a_sd) if use_cc and x.cc_a_sd not in ("", "nan") else float(x.sd_a or 0)
        t_end = x.t_end_override_utc.strip() or x.t_end_utc
        rows.append(dict(epoch_id=x.candidate_id, station=x.station, model=x.model, t_start_utc=x.t_start_utc,
                         t_start_sd_days=float(x.sd_t_start_days or 0), t_end_utc=t_end, size=size, size_sd=sd,
                         size_source="cc" if use_cc else "residuals", confirmed_by=x.reviewer,
                         decision_ref=x.decision_ref, evidence=x.evidence, note=x.note))
    e = pd.DataFrame(rows, columns=["epoch_id", "station", "model", "t_start_utc", "t_start_sd_days", "t_end_utc", "size",
                                    "size_sd", "size_source", "confirmed_by", "decision_ref", "evidence", "note"])
    if (e.confirmed_by.str.strip() == "").any():
        raise ValueError("accepted rows need a reviewer (confirmed_by)")
    e.to_csv(out_csv, index=False)
    return e, r


def cli_ingest(cfg, a):
    out = a.output or str(Path(cfg["output_dir"]) / "confirmed_epochs.csv")
    e, r = ingest(a.review, out)
    undecided = int((r.decision.str.strip() == "").sum())
    print(f"ingest-review: {len(e)} accepted, {int((r.decision.str.lower() == 'reject').sum())} rejected, "
          f"{undecided} undecided -> {out}")
    return 0
