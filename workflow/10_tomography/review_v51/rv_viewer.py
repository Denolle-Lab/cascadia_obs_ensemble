"""review_v51_viewer.html: one self-contained file (data, summary figures and record-section thumbnails embedded) to
browse review_v51_events.csv, edit the reviewer columns (localStorage autosave), load returned CSVs (merge reviewers)
and download the CSV with the same columns. The JS core (CSV parse / serialize / merge) is also written to
WORK/viewer/vcore.js for the node round-trip test (rv_viewer_test.js).

  python rv_viewer.py [--nproc 12]
"""
import argparse
import base64
import json
import os
from multiprocessing import Pool

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rv_common as C  # noqa: E402

V = C.WORK / "viewer"
TH = C.WORK / "thumbs"
OUT = C.OUT / "review_v51_viewer.html"
FOLDERS = {"P1": "P1_msh_deep", "P2": "P2_brookings", "P3": "P3_tensor", "P4": "P4_deep_far"}
B1, B2, B3, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#6b6a66", "#d9d8d4"
TITLES = {"P1": "Deep events under Mount St Helens", "P2": "2012 Brookings swarm (v1_GO022)",
          "P3": "Events with ComCat moment tensors", "P4": "Deep-pass and far-only events"}


def _thumb(eid):
    import rv_figs as F
    f = TH / f"{eid}.webp"
    if not f.exists():
        try:
            F.fig_thumb(int(eid), f)
        except Exception as e:
            return eid, f"FAILED {e}"
    return eid, "ok"


def b64(path):
    return "data:image/webp;base64," + base64.b64encode(path.read_bytes()).decode()


def coast(ax, ext):
    lo, la = np.meshgrid(np.linspace(ext[0], ext[1], 260), np.linspace(ext[2], ext[3], 260))
    h = C.surface_km(lo, la)
    if np.nanmin(h) < 0 < np.nanmax(h):
        ax.contour(lo, la, h, levels=[0], colors=[MUTED], linewidths=0.6)


def summary_fig(p, D, out):
    x = D[(D.problem == p) & D.lat.notna()]
    flag = x.auto_indicator_rule_not_verdict.ne("none")
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    ax = axs[0]
    lon = pd.concat([x.lon, x.lon]); lat = pd.concat([x.lat, x.lat])
    pad = 0.15 if p in ("P1", "P2") else 0.6
    ext = [lon.min() - pad, lon.max() + pad, lat.min() - pad * 0.7, lat.max() + pad * 0.7]
    coast(ax, ext)
    ref = x.role.eq("reference")
    ax.plot(x.lon[~flag & ~ref], x.lat[~flag & ~ref], "o", ms=3, color=B1, alpha=0.7, label=f"no rule fired ({int((~flag & ~ref).sum())})")
    ax.plot(x.lon[flag], x.lat[flag], "o", ms=3, color=B2, alpha=0.7, label=f"an automatic rule fired ({int(flag.sum())})")
    if ref.any():
        ax.plot(x.lon[ref], x.lat[ref], "D", ms=5, color=B3, label=f"shallow references ({int(ref.sum())})")
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
    ax.set_aspect(1 / np.cos(np.radians(x.lat.median())))
    ax.grid(color=GRID, lw=0.4)
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"{p}: v5.1 epicentres ({len(x)} events)", loc="left", fontsize=9)
    ax = axs[1]
    ax.plot(x.depth_v4[~flag], x.depth_v51[~flag], "o", ms=3, color=B1, alpha=0.7)
    ax.plot(x.depth_v4[flag], x.depth_v51[flag], "o", ms=3, color=B2, alpha=0.7)
    if ref.any():
        ax.plot(x.depth_v4[ref], x.depth_v51[ref], "D", ms=5, color=B3)
    m = max(10, np.nanmax([x.depth_v4.max(), x.depth_v51.max()]) + 3)
    ax.plot([-5, m], [-5, m], "--", color=MUTED, lw=0.7)
    ax.set_xlim(-5, m); ax.set_ylim(m, -5)
    ax.set_xlabel("v4 depth (km)"); ax.set_ylabel("v5.1 depth (km)")
    ax.grid(color=GRID, lw=0.4)
    ax.set_title("v4 vs v5.1 depth (dashed: 1:1)", loc="left", fontsize=9)
    ax = axs[2]
    if p == "P1":
        y = x[x.role.ne("reference")]
        cal = y.calib_any.eq("yes")
        ax.plot(y.P_span_s.clip(upper=3)[~cal], y.pct_P_at_UW[~cal], "o", ms=3, color=B1, alpha=0.7, label="no calibration signal at the event time")
        ax.plot(y.P_span_s.clip(upper=3)[cal], y.pct_P_at_UW[cal], "o", ms=3, color=B2, alpha=0.8, label="calibration signal at HSR/EDM/STD.UW")
        r = x[x.role.eq("reference")]
        ax.plot(r.P_span_s.clip(upper=3), r.pct_P_at_UW, "D", ms=5, color=B3, label="shallow references")
        ax.axvline(0.5, color=MUTED, lw=0.6, ls="--"); ax.axhline(90, color=MUTED, lw=0.6, ls="--")
        ax.set_xlim(-0.05, 3.1)
        ax.legend(fontsize=6.5, loc="lower right")
        ax.set_xlabel("P-pick time spread (s), values above 3 s drawn at 3 s"); ax.set_ylabel("% of P picks at UW stations")
        ax.set_title("P-pick spread vs UW share (dashed: P1_common_mode limits)", loc="left", fontsize=9)
    elif p == "P2":
        for st, col, mk in (("KBO", B2, "s"), ("KSXB", B1, "o"), ("L02D", B3, "^"), ("L02E", "#6250d6", "v")):
            ax.plot(x[f"sp_pred_v51_{st}"], x[f"sp_obs_{st}"], mk, color=col, ms=5, alpha=0.8, label=f"{st}: predicted at v5.1")
            ax.plot(x[f"sp_pred_v4_{st}"], x[f"sp_obs_{st}"], mk, mfc="none", mec=col, ms=5, alpha=0.8, label=f"{st}: predicted at v4")
        ax.plot([4, 14], [4, 14], "--", color=MUTED, lw=0.7)
        ax.set_xlabel("predicted S-P (s), v5.1 grids"); ax.set_ylabel("observed S-P (s)")
        ax.legend(fontsize=6.5, ncol=2)
        ax.set_title("S-P: observed vs predicted (filled v5.1, open v4)", loc="left", fontsize=9)
    elif p == "P3":
        ax.plot(x.dt_comcat[~flag], x.dz_comcat[~flag], "o", ms=3.5, color=B1, alpha=0.7)
        ax.plot(x.dt_comcat[flag], x.dz_comcat[flag], "o", ms=3.5, color=B2, alpha=0.8)
        ax.axhline(0, color=MUTED, lw=0.6); ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("v5.1 - ComCat origin time (s)"); ax.set_ylabel("v5.1 - ComCat depth (km)")
        ax.set_title("difference to ComCat", loc="left", fontsize=9)
    else:
        tc = x.two_cluster_scatter.eq("yes")
        for flg, mk in (("far_only", "o"), ("deep_pass", "s")):
            y = x[x.role == flg]
            t2 = y.two_cluster_scatter.eq("yes")
            ax.plot(y.depth_v51[~t2], y.sig_z[~t2], mk, ms=3, color=B1, alpha=0.6, label=f"{flg}")
            ax.plot(y.depth_v51[t2], y.sig_z[t2], mk, ms=4, color=B2, alpha=0.9, label=f"{flg}, two-cluster scatter")
        ax.set_xlabel("v5.1 depth (km)"); ax.set_ylabel("sig_z (km)")
        ax.legend(fontsize=7)
        ax.set_title(f"depth vs sig_z ({int(tc.sum())} with two-cluster scatter)", loc="left", fontsize=9)
    ax.grid(color=GRID, lw=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=85, format="webp", pil_kwargs={"quality": 80})
    plt.close(fig)


VCORE = r"""
var VCORE = (function () {
  var REVIEW = ["reviewer", "is_earthquake", "depth_verdict", "corrected_depth_km", "action", "comment", "reviewed_utc"];
  var ANSWER = ["is_earthquake", "depth_verdict", "corrected_depth_km", "action", "comment"];
  var ALLOWED = {is_earthquake: ["", "yes", "no", "unsure"], depth_verdict: ["", "plausible", "implausible", "cannot_tell"],
                 action: ["", "keep", "flag_not_earthquake", "flag_depth_unreliable", "relocate", "other"]};
  function parseCSV(text) {
    var rows = [], row = [], field = "", inq = false, i, ch;
    if (text.charCodeAt(0) === 0xFEFF) { text = text.slice(1); }
    for (i = 0; i < text.length; i++) {
      ch = text.charAt(i);
      if (inq) {
        if (ch === '"') { if (text.charAt(i + 1) === '"') { field += '"'; i++; } else { inq = false; } }
        else { field += ch; }
      } else if (ch === '"') { inq = true; }
      else if (ch === ',') { row.push(field); field = ""; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text.charAt(i + 1) === '\n') { i++; }
        row.push(field); rows.push(row); row = []; field = "";
      } else { field += ch; }
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    return rows;
  }
  function toObjects(rows) {
    var header = rows[0], recs = [], r, o, k;
    for (r = 1; r < rows.length; r++) {
      if (rows[r].length === 1 && rows[r][0] === "") { continue; }
      o = {};
      for (k = 0; k < header.length; k++) { o[header[k]] = rows[r][k] === undefined ? "" : rows[r][k]; }
      recs.push(o);
    }
    return {header: header, records: recs};
  }
  function q(v) { v = (v === null || v === undefined) ? "" : String(v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
  function serialize(header, recs) {
    var lines = [header.map(q).join(",")];
    recs.forEach(function (r) { lines.push(header.map(function (h) { return q(r[h]); }).join(",")); });
    return lines.join("\n") + "\n";
  }
  function key(r) { return String(r.ID) + "|" + String(r.problem); }
  function hasReview(r) { return ANSWER.some(function (k) { return (r[k] || "") !== ""; }); }
  function normalize(header, recs) {
    if (header.indexOf("is_earthquake") >= 0) { return recs; }
    if (header.indexOf("event_id") >= 0 && header.indexOf("earthquake") >= 0) {     // CSV of a report page
      var eq = {earthquake: "yes", not_earthquake: "no", unsure: "unsure"};
      return recs.map(function (r) {
        return {ID: r.event_id, problem: r.problem, reviewer: r.reviewer || "", is_earthquake: eq[r.earthquake] || "",
                depth_verdict: r.v51_depth || "", corrected_depth_km: "", action: "", comment: r.comment || "",
                reviewed_utc: r.answered_at_utc || ""};
      });
    }
    throw new Error("unknown CSV format: no is_earthquake or event_id/earthquake column");
  }
  function sameAnswers(a, b) { return ANSWER.every(function (k) { return (a[k] || "") === (b[k] || ""); }); }
  function merge(base, imported) {
    var idx = {}, st = {applied: 0, identical: 0, older: 0, unknown: [], invalid: [], conflicts: []};
    base.forEach(function (r) { idx[key(r)] = r; });
    imported.forEach(function (r) {
      if (!hasReview(r)) { return; }
      var b = idx[key(r)], k, bad = [];
      if (!b) { st.unknown.push(key(r)); return; }
      for (k in ALLOWED) { if (ALLOWED[k].indexOf(r[k] || "") < 0) { bad.push(k + "=" + r[k]); } }
      if (bad.length) { st.invalid.push(key(r) + " " + bad.join(" ")); return; }
      if (!hasReview(b)) { REVIEW.forEach(function (c) { b[c] = r[c] || ""; }); st.applied++; return; }
      if (sameAnswers(b, r)) { st.identical++; return; }
      if ((b.reviewer || "") === (r.reviewer || "")) {
        if ((r.reviewed_utc || "") >= (b.reviewed_utc || "")) { REVIEW.forEach(function (c) { b[c] = r[c] || ""; }); st.applied++; }
        else { st.older++; }
        return;
      }
      st.conflicts.push({key: key(r), kept: (b.reviewer || "?") + ": " + [b.is_earthquake, b.depth_verdict, b.action].join("/"),
                         other: (r.reviewer || "?") + ": " + [r.is_earthquake, r.depth_verdict, r.action].join("/")});
    });
    return st;
  }
  return {REVIEW: REVIEW, ANSWER: ANSWER, ALLOWED: ALLOWED, parseCSV: parseCSV, toObjects: toObjects, serialize: serialize,
          key: key, hasReview: hasReview, normalize: normalize, merge: merge};
})();
if (typeof module !== "undefined") { module.exports = VCORE; }
"""

UI = r"""
(function () {
  var V = VCORE, LS = "v51review:viewer:v1", LR = "v51review:reviewer";
  var D = V.toObjects(V.parseCSV(DATA_CSV)), HEADER = D.header, R = D.records, IDX = {}, CONFLICT = {};
  R.forEach(function (r) { IDX[V.key(r)] = r; });
  var storageOK = true;
  function msg(t) { document.getElementById("msg").textContent = t; }
  try { var saved = JSON.parse(localStorage.getItem(LS) || "{}"); Object.keys(saved).forEach(function (k) {
    var r = IDX[k]; if (r) { V.REVIEW.forEach(function (c) { r[c] = saved[k][c] || ""; }); } }); }
  catch (e) { storageOK = false; }
  function persist() {
    var o = {};
    R.forEach(function (r) { if (V.hasReview(r)) { var x = {}; V.REVIEW.forEach(function (c) { x[c] = r[c]; }); o[V.key(r)] = x; } });
    try { localStorage.setItem(LS, JSON.stringify(o)); } catch (e) { storageOK = false; }
    document.getElementById("store").textContent = storageOK ? "answers autosaved in this browser" :
      "this browser blocks local storage: download the CSV before closing";
  }
  var who = document.getElementById("who");
  try { who.value = localStorage.getItem(LR) || ""; } catch (e) {}
  who.addEventListener("input", function () { try { localStorage.setItem(LR, who.value); } catch (e) {} });
  function esc(s) { return String(s === undefined ? "" : s).replace(/[&<>"']/g, function (c) {
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]; }); }
  function num(v) { var n = parseFloat(v); return isNaN(n) ? null : n; }
  function f1(v, d) { var n = num(v); return n === null ? "" : n.toFixed(d === undefined ? 1 : d); }
  function metrics(r) {
    if (r.problem === "P1") { return "UW " + f1(r.pct_P_at_UW, 0) + "%, span " + f1(r.P_span_s, 2) + " s, S " + (r.has_S || "") +
      ", Wed " + (r.wed_1914_1921 || "") + ", calib " + (r.calib_any || ""); }
    if (r.problem === "P2") { var s = []; ["KBO", "KSXB", "L02D", "L02E"].forEach(function (k) { if (r["sp_obs_" + k] !== "") {
      s.push(k + " S-P " + f1(r["sp_obs_" + k], 2) + " (v5.1 " + f1(r["sp_pred_v51_" + k], 2) + ", v4 " + f1(r["sp_pred_v4_" + k], 2) + ")"); } });
      return s.join("; "); }
    if (r.problem === "P3") { return r.dz_comcat === "" ? "" : "dz " + f1(r.dz_comcat) + " km, dh " + f1(r.dh_comcat) + " km, dt " + f1(r.dt_comcat, 2) + " s"; }
    return "dz v4->v5.1 " + f1(r.dz_v4_to_v51) + " km, two-cluster " + (r.two_cluster_scatter || "");
  }
  function status(r) { return CONFLICT[V.key(r)] ? "conflict" : (V.hasReview(r) ? "reviewed" : "open"); }
  var SEL = {is_earthquake: ["", "yes", "no", "unsure"], depth_verdict: ["", "plausible", "implausible", "cannot_tell"],
             action: ["", "keep", "flag_not_earthquake", "flag_depth_unreliable", "relocate", "other"]};
  function sel(r, k) { return "<select data-k='" + k + "'>" + SEL[k].map(function (v) {
    return "<option" + (r[k] === v ? " selected" : "") + ">" + v + "</option>"; }).join("") + "</select>"; }
  var COLS = [["ID", "ID"], ["problem", "pb"], ["role", "role"], ["time_utc", "origin UTC"], ["depth_v51", "v5.1 km"],
              ["depth_v4", "v4 km"], ["comcat_depth", "ComCat km"], ["tier", "tier"], ["sig_z", "sig_z"], ["region", "region"],
              ["magnitude", "M"], ["rms_v51", "rms v5.1"], ["rms_v4", "rms v4"], ["rms_comcat", "rms CC"],
              ["_metrics", "problem metrics"], ["auto_indicator_rule_not_verdict", "automatic rule (not a verdict)"],
              ["_thumb", "record section"], ["is_earthquake", "earthquake?"], ["depth_verdict", "v5.1 depth"],
              ["corrected_depth_km", "corrected depth km"], ["action", "action"], ["comment", "comment"],
              ["reviewer", "reviewer"], ["_status", "status"]];
  var sortK = null, sortAsc = true;
  function rowHTML(r) {
    var k = V.key(r), th = THUMBS[r.ID + "|" + r.problem], link = r.view_path ? "<a href='" + esc(r.view_path) + "' target='_blank'>event page</a>" : "";
    return "<tr data-key='" + esc(k) + "' class='" + status(r) + "'>" + COLS.map(function (c) {
      var n = c[0];
      if (n === "_metrics") { return "<td class='l small'>" + esc(metrics(r)) + "</td>"; }
      if (n === "_thumb") { return "<td>" + (th ? "<a href='" + esc(r.view_path) + "' target='_blank'><img src='" + th + "' alt='record section " + esc(r.ID) + "'></a><br>" : "") + link + "</td>"; }
      if (n === "_status") { return "<td class='st'>" + status(r) + "</td>"; }
      if (SEL[n]) { return "<td>" + sel(r, n) + "</td>"; }
      if (n === "corrected_depth_km") { return "<td><input data-k='corrected_depth_km' type='text' size='5' value='" + esc(r[n]) + "'></td>"; }
      if (n === "comment") { return "<td><input data-k='comment' type='text' size='22' value='" + esc(r[n]) + "'></td>"; }
      if (n === "time_utc") { return "<td class='l'>" + esc(String(r[n]).slice(0, 19)) + "</td>"; }
      if (["depth_v51", "depth_v4", "comcat_depth", "sig_z"].indexOf(n) >= 0) { return "<td>" + f1(r[n]) + "</td>"; }
      if (["rms_v51", "rms_v4", "rms_comcat", "magnitude"].indexOf(n) >= 0) { return "<td>" + f1(r[n], 2) + "</td>"; }
      return "<td class='" + (n === "ID" || n === "tier" ? "" : "l") + "'>" + esc(r[n]) + "</td>";
    }).join("") + "</tr>";
  }
  function filtered() {
    var p = document.getElementById("fp").value, ind = document.getElementById("fi").value, t = document.getElementById("ft").value,
        s = document.getElementById("fs").value, vw = document.getElementById("fv").checked, txt = document.getElementById("fx").value.toLowerCase();
    return R.filter(function (r) {
      if (p && r.problem !== p) { return false; }
      if (ind === "__any" && r.auto_indicator_rule_not_verdict === "none") { return false; }
      if (ind && ind !== "__any" && r.auto_indicator_rule_not_verdict.split(";").indexOf(ind) < 0) { return false; }
      if (t && r.tier !== t) { return false; }
      if (s && status(r) !== s) { return false; }
      if (vw && r.has_view !== "yes") { return false; }
      if (txt && (r.ID + " " + r.region + " " + r.role + " " + r.comment + " " + r.reviewer).toLowerCase().indexOf(txt) < 0) { return false; }
      return true;
    });
  }
  function render() {
    var rows = filtered();
    if (sortK) {
      rows.sort(function (a, b) {
        var x = sortK === "_status" ? status(a) : a[sortK], y = sortK === "_status" ? status(b) : b[sortK], nx = num(x), ny = num(y), c;
        if (nx !== null && ny !== null) { c = nx - ny; } else { c = String(x).localeCompare(String(y)); }
        return sortAsc ? c : -c;
      });
    }
    document.getElementById("tb").innerHTML = rows.map(rowHTML).join("");
    var nrev = R.filter(function (r) { return V.hasReview(r); }).length;
    document.getElementById("count").textContent = rows.length + " of " + R.length + " rows shown; " + nrev + " reviewed; " +
      Object.keys(CONFLICT).length + " conflicts";
  }
  document.getElementById("th").innerHTML = COLS.map(function (c) { return "<th data-k='" + c[0] + "'>" + esc(c[1]) + "</th>"; }).join("");
  document.getElementById("th").addEventListener("click", function (e) {
    var k = e.target.getAttribute("data-k"); if (!k || k === "_thumb" || k === "_metrics") { return; }
    if (sortK === k) { sortAsc = !sortAsc; } else { sortK = k; sortAsc = true; } render();
  });
  var inds = {};
  R.forEach(function (r) { r.auto_indicator_rule_not_verdict.split(";").forEach(function (t) { inds[t] = 1; }); });
  document.getElementById("fi").innerHTML = "<option value=''>any or none</option><option value='__any'>any rule fired</option>" +
    Object.keys(inds).sort().map(function (t) { return "<option>" + esc(t) + "</option>"; }).join("");
  ["fp", "fi", "ft", "fs", "fv", "fx"].forEach(function (id) { document.getElementById(id).addEventListener(id === "fx" ? "input" : "change", render); });
  function edit(e) {
    var el = e.target, k = el.getAttribute("data-k"); if (!k) { return; }
    var tr = el.closest("tr"), r = IDX[tr.getAttribute("data-key")];
    r[k] = el.value.trim();
    if (who.value.trim()) { r.reviewer = who.value.trim(); } else { msg("Type your name in Reviewer so that your answers carry it."); }
    r.reviewed_utc = new Date().toISOString();
    delete CONFLICT[V.key(r)];
    tr.className = status(r); tr.querySelector("td.st").textContent = status(r);
    tr.cells[COLS.length - 2].textContent = r.reviewer;
    persist();
  }
  document.getElementById("tb").addEventListener("change", edit);
  document.getElementById("dl").addEventListener("click", function () {
    var blob = new Blob([V.serialize(HEADER, R)], {type: "text/csv"}), u = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = u; a.download = "review_v51_events_" + (who.value.trim() || "reviewer").replace(/[^A-Za-z0-9_-]+/g, "_") + "_" +
      new Date().toISOString().slice(0, 10) + ".csv";
    document.body.appendChild(a); a.click(); setTimeout(function () { URL.revokeObjectURL(u); a.remove(); }, 1500);
  });
  document.getElementById("load").addEventListener("change", function (e) {
    var files = Array.prototype.slice.call(e.target.files), out = [];
    files.forEach(function (f) {
      var fr = new FileReader();
      fr.onload = function () {
        try {
          var d = V.toObjects(V.parseCSV(fr.result)), st = V.merge(R, V.normalize(d.header, d.records));
          st.conflicts.forEach(function (c) { CONFLICT[c.key] = c; });
          out.push(f.name + ": " + st.applied + " applied, " + st.identical + " identical, " + st.older + " older, " +
                   st.conflicts.length + " conflicts" + (st.conflicts.length ? " (" + st.conflicts.slice(0, 5).map(function (c) {
                     return c.key + " kept " + c.kept + " / other " + c.other; }).join("; ") + ")" : "") +
                   (st.unknown.length ? ", " + st.unknown.length + " unknown rows" : "") +
                   (st.invalid.length ? ", " + st.invalid.length + " invalid (" + st.invalid.slice(0, 3).join("; ") + ")" : ""));
        } catch (err) { out.push(f.name + ": " + err.message); }
        if (out.length === files.length) { persist(); render(); msg(out.join(" | ")); }
      };
      fr.readAsText(f);
    });
  });
  document.getElementById("clear").addEventListener("click", function () {
    if (!confirm("Remove all answers stored in this browser for this viewer?")) { return; }
    R.forEach(function (r) { V.REVIEW.forEach(function (c) { r[c] = ""; }); }); CONFLICT = {};
    try { localStorage.removeItem(LS); } catch (e) {} render(); msg("local answers cleared");
  });
  var tabs = document.querySelectorAll("button.tab");
  Array.prototype.forEach.call(tabs, function (b) { b.addEventListener("click", function () {
    Array.prototype.forEach.call(document.querySelectorAll(".panel"), function (p) { p.style.display = p.id === "pan-" + b.getAttribute("data-p") ? "block" : "none"; });
    Array.prototype.forEach.call(tabs, function (x) { x.classList.toggle("on", x === b); });
    document.getElementById("fp").value = b.getAttribute("data-p") === "all" ? "" : b.getAttribute("data-p"); render();
  }); });
  persist(); render();
})();
"""

CSS = """
:root{--bg:#fcfcfb;--panel:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#e2e1dd;--chip:#eef3fb;--accent:#2a78d6;--ok:#1b7f4b;--bad:#b3261e;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#222221;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--chip:#26303d;--accent:#3987e5;--ok:#4cc38a;--bad:#ef6b62;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#222221;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--chip:#26303d;--accent:#3987e5;--ok:#4cc38a;--bad:#ef6b62;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1800px;margin:0 auto;padding:0 16px 60px}h1{font-size:21px;margin:16px 0 4px}h2{font-size:17px;margin:22px 0 6px}
a{color:var(--accent)}.note{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--accent);padding:8px 12px;border-radius:4px;margin:8px 0}
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:8px 0}
.bar input,.bar select,button{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:3px 6px}
button{background:var(--chip);cursor:pointer}button.tab.on{background:var(--accent);color:#fff}
.panel img{max-width:100%;border:1px solid var(--line);border-radius:4px;background:#fff;margin:4px 0}
.tw{overflow:auto;max-height:78vh;border:1px solid var(--line);border-radius:4px}
table{border-collapse:collapse;font-size:12.5px}th,td{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;vertical-align:top}
th{position:sticky;top:0;background:var(--chip);cursor:pointer;white-space:nowrap;z-index:1}td.l{text-align:left}td.small{font-size:11.5px;max-width:260px}
td img{width:190px;height:auto;background:#fff;border:1px solid var(--line)}tr.reviewed td.st{color:var(--ok);font-weight:600}
tr.conflict td.st{color:var(--bad);font-weight:600}#msg{color:var(--ink2);font-size:13px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=12)
    a = ap.parse_args()
    V.mkdir(parents=True, exist_ok=True)
    TH.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(C.OUT / "review_v51_events.csv", low_memory=False, dtype=str, keep_default_na=False)
    Dn = pd.read_csv(C.OUT / "review_v51_events.csv", low_memory=False)
    views = Dn[Dn.has_view == "yes"]
    with Pool(min(a.nproc, 12)) as p:
        res = p.map(_thumb, sorted(set(views.ID)), chunksize=4)
    bad = [r for r in res if r[1] != "ok"]
    print("thumbs", len(res), "failed", bad[:5], flush=True)
    thumbs = {f"{r.ID}|{r.problem}": b64(TH / f"{r.ID}.webp") for r in views.itertuples() if (TH / f"{r.ID}.webp").exists()}
    panels = []
    ctx = {"P1": [("week_hist.webp", "Time of week of the box events by depth class"),
                  ("weekly_scan.webp", "Weekly scan: quiet stretch at HSR.UW, STD.UW, STD.CC, Wednesdays vs Tuesdays"),
                  ("raw_examples.webp", "Raw counts around Wednesday-window events: analog UW vs digital STD.CC")],
           "P2": [("sp_depth.webp", "S-P against depth at the nearest stations (v5.1 grids) and observed")],
           "P3": [("comcat_diff.webp", "v5.1 minus ComCat by region")],
           "P4": [("depth_hist.webp", "v4 vs v5.1 depth histograms by region")]}
    S = json.loads((C.WORK / "export_counts.json").read_text())
    for p in ("P1", "P2", "P3", "P4"):
        f = V / f"summary_{p}.webp"
        summary_fig(p, Dn, f)
        cnt = S["counts"][p]
        imgs = [f"<img src='{b64(f)}' alt='{p} summary'>"]
        for name, cap in ctx[p]:
            g = C.OUT / FOLDERS[p] / "ctx" / name
            if g.exists():
                imgs.append(f"<div><b>{cap}</b><br><img src='{b64(g)}' alt='{cap}'></div>")
        rules = ", ".join(f"{k} {v}" for k, v in cnt["by_rule"].items())
        panels.append(f"<div class='panel' id='pan-{p}' style='display:{'block' if p == 'P1' else 'none'}'><h2>{p}. {TITLES[p]}</h2><p>{cnt['rows']} rows "
                      f"({', '.join(f'{k} {v}' for k, v in cnt['by_role'].items())}); {cnt['with_view']} with an event page; "
                      f"automatic rules: {rules}. Full report: <a href='{FOLDERS[p]}/index.html' target='_blank'>"
                      f"{FOLDERS[p]}/index.html</a>.</p>{''.join(imgs)}</div>")
    rules = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in S["rules"].items())
    csv_text = (C.OUT / "review_v51_events.csv").read_text()
    data_js = "var DATA_CSV = " + json.dumps(csv_text).replace("</", "<\\/") + ";\nvar THUMBS = " + \
              json.dumps(thumbs).replace("</", "<\\/") + ";\n"
    (V / "vcore.js").write_text(VCORE)
    howto = ("<div class='note'><b>How to use.</b> 1. Type your name in <i>Reviewer</i>. 2. Pick a problem tab or filter the "
             "table (problem, automatic rule, tier, status, only events with an event page, text). 3. For each event, open "
             "its event page (link or thumbnail; the four report folders must sit next to this file) and fill "
             "<i>earthquake?</i> (yes / no / unsure), <i>v5.1 depth</i> (plausible / implausible / cannot_tell), an optional "
             "corrected depth in km, the <i>action</i> (keep / flag_not_earthquake / flag_depth_unreliable / relocate / "
             "other) and a comment. Answers are saved in this browser as you type. 4. Click <i>Download CSV</i> and send "
             "the file back: it has every row and the same columns as review_v51_events.csv. To continue later or merge "
             "several reviewers, load one or more CSVs with <i>Load CSV</i> (this viewer's CSV or a report page's CSV); an "
             "event answered differently by two reviewers is kept as it is here and listed as a conflict. The column "
             "<i>auto_indicator_rule_not_verdict</i> names the automatic rules that fired (list below); it is a pointer "
             "for the reviewer, not a verdict.</div>")
    body = (f"<h1>Catalog v5.1 location review: events and answers</h1><p>{len(D)} rows from review_v51_events.csv "
            f"(PI request 2026-11-25). Built {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC.</p>{howto}"
            "<div class='bar'><b>Reviewer</b><input id='who' type='text' placeholder='your name'>"
            "<button id='dl' type='button'>Download CSV</button><label>Load CSV <input id='load' type='file' accept='.csv' multiple></label>"
            "<button id='clear' type='button'>clear local answers</button><span id='store'></span></div><div id='msg'></div>"
            "<div class='bar'>" + "".join(f"<button class='tab{' on' if p == 'P1' else ''}' type='button' data-p='{p}'>{p}: {TITLES[p]}</button>" for p in ("P1", "P2", "P3", "P4"))
            + "<button class='tab' type='button' data-p='all'>all problems</button></div>" + "".join(panels) +
            "<details><summary>Automatic rules (the indicator column; not verdicts)</summary><ul>" + rules + "</ul></details>"
            "<div class='bar'>problem <select id='fp'><option value=''>all</option><option>P1</option><option>P2</option>"
            "<option>P3</option><option>P4</option></select> rule <select id='fi'></select> tier <select id='ft'><option value=''>all</option>"
            "<option>A</option><option>B</option><option>C</option><option>D</option><option>U</option></select> status <select id='fs'>"
            "<option value=''>all</option><option value='open'>not reviewed</option><option value='reviewed'>reviewed</option>"
            "<option value='conflict'>conflict</option></select><label><input id='fv' type='checkbox'> only events with a page</label>"
            "search <input id='fx' type='text' size='16' placeholder='ID, region, comment'><span id='count'></span></div>"
            "<div class='tw'><table><thead><tr id='th'></tr></thead><tbody id='tb'></tbody></table></div>")
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>v5.1 review viewer</title><style>{CSS}</style></head><body><main>{body}</main>"
            f"<script>{data_js}</script><script>{VCORE}</script><script>{UI}</script></body></html>")
    OUT.write_text(html)
    (V / "ui.js").write_text(UI)
    print(OUT, OUT.stat().st_size / 1e6, "MB", len(thumbs), "thumbnails", flush=True)


if __name__ == "__main__":
    main()
