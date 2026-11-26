"""apply: write a corrected pick file. Never drops or reorders a pick; rows that are not corrected keep their
original text byte for byte, and new columns are appended to every row.

Corrections:
* clock (clock model from model-clock): clock_correction_s = -c(t) is added to the pick time.
* re-timing (corrections CSV from retime, optional): timing_correction_s for the rows applied there. If the input
  already has a timing_correction_s column (it was re-timed before) that column is kept as it is.
In corrected rows the time column (and the travel-time column, when mapped) are rewritten with the same number of
decimals and suffix as the original text. New columns: clock_correction_s, clock_correction_sd_s, clock_epoch,
clock_flag, total_correction_s (re-timing + clock: corrected time minus the time before any timing correction),
and timing_corrected/timing_correction_s when they were not there. A provenance JSON (inputs with md5, clock model,
code commit, counts) is written next to the output.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .clockmodel import evaluate, load_model
from .tables import load, md5, provenance, write_json

_DEC = re.compile(r"\.(\d+)")


def _fmt_time_like(old: str, ns: int) -> str:
    """Format integer ns POSIX like `old` (same decimals, same 'Z' suffix)."""
    s = old.strip()
    suffix = "Z" if s.endswith("Z") else ""
    m = _DEC.search(s.split("T")[-1])
    dec = len(m.group(1)) if m else 0
    q = 10 ** (9 - dec)
    ns_r = (ns + q // 2) // q * q
    ts = pd.Timestamp(ns_r, unit="ns")
    out = ts.strftime("%Y-%m-%dT%H:%M:%S")
    if dec:
        out += "." + f"{ns_r % 10**9:09d}"[:dec]
    return out + suffix


def _fmt_num_like(old: str, val: float) -> str:
    s = old.strip()
    m = _DEC.search(s)
    dec = len(m.group(1)) if m else 0
    return f"{val:.{dec}f}"


def _time_ns(s: str) -> int:
    return int(pd.Timestamp(s.strip().rstrip("Z")).value)


def apply_corrections(cfg, picks_path, clock_model=None, retime_csv=None, output=None):
    spec = dict(cfg["picks"])
    spec["path"] = picks_path
    cols = spec.get("columns", {})
    tcol = cols.get("time", "time")
    ttcol = cols.get("travel_time")
    tccol = cols.get("timing_correction")
    df = load(dict(cfg, picks=spec), "picks")
    n = len(df)
    shift = np.zeros(n)
    # ---- clock
    if clock_model:
        m = load_model(clock_model)
        ev = evaluate(m, df.station.values, df.time_s.values)
    else:
        ev = pd.DataFrame({"clock_c_s": np.zeros(n), "clock_sd_s": np.zeros(n), "clock_epoch": "",
                           "clock_flag": "none"})
    clock_corr = -ev.clock_c_s.values + 0.0   # + 0.0: no "-0.000000"
    # ---- re-timing
    have_tc = tccol is not None and "timing_correction" in df.columns
    if have_tc:
        tc_existing = pd.to_numeric(df["timing_correction"], errors="coerce").fillna(0.0).values
        tc_new = np.zeros(n)
    else:
        tc_existing = np.zeros(n)
        tc_new = np.zeros(n)
        if retime_csv:
            r = pd.read_csv(retime_csv)
            r = r[r.applied.astype(bool)]
            tc_new[r.pick_row.values.astype(int)] = r.timing_correction_s.values
    if have_tc and retime_csv:
        print("note: the pick file already has a timing_correction column; --retime-corrections ignored")
    shift = clock_corr + tc_new
    changed = np.round(shift, 9) != 0
    total = tc_existing + tc_new + clock_corr + 0.0
    # ---- write line by line
    raw = Path(picks_path).read_text().splitlines(keepends=True)
    header, lines = raw[0], raw[1:]
    if len(lines) != n:
        raise ValueError(f"{picks_path}: {len(lines)} data lines but {n} rows parsed (blank or comment lines?)")
    nl = "\r\n" if header.endswith("\r\n") else "\n"
    hdr_fields = header.rstrip("\r\n").split(",")
    sep = "," + (hdr_fields[1][: len(hdr_fields[1]) - len(hdr_fields[1].lstrip())] if len(hdr_fields) > 1 else "")
    names = [h.strip() for h in hdr_fields]
    it = names.index(tcol)
    itt = names.index(ttcol) if ttcol and ttcol in names else None
    new_cols = []
    if not have_tc:
        new_cols += ["timing_corrected", "timing_correction_s"]
    new_cols += ["clock_correction_s", "clock_correction_sd_s", "clock_epoch", "clock_flag", "total_correction_s"]
    if cfg["apply"].get("corrected_column"):
        new_cols.append(cfg["apply"]["corrected_column"])
    out_lines = [header.rstrip("\r\n") + "".join(sep + c for c in new_cols) + nl]
    ep = ev.clock_epoch.values
    fl = ev.clock_flag.values
    sdv = ev.clock_sd_s.values
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        end = line[len(body):] or nl
        if '"' in body and changed[i]:
            raise ValueError(f"line {i + 2}: quoted fields are not supported for rewritten rows")
        if changed[i]:
            f = body.split(",")
            lead = f[it][: len(f[it]) - len(f[it].lstrip())]
            ns = _time_ns(f[it]) + int(round(shift[i] * 1e9))
            f[it] = lead + _fmt_time_like(f[it], ns)
            if itt is not None and f[itt].strip() not in ("", "nan"):
                lead2 = f[itt][: len(f[itt]) - len(f[itt].lstrip())]
                f[itt] = lead2 + _fmt_num_like(f[itt], float(f[itt]) + shift[i])
            body = ",".join(f)
        extra = []
        if not have_tc:
            extra += [str(bool(tc_new[i] != 0)), f"{tc_new[i]:.6f}"]
        extra += [f"{clock_corr[i]:.6f}", f"{sdv[i]:.6f}", ep[i], fl[i], f"{total[i]:.6f}"]
        if cfg["apply"].get("corrected_column"):
            fcur = body.split(",")[it]
            extra.append(fcur.strip())
        out_lines.append(body + "".join(sep + x for x in extra) + end)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as fh:
        fh.writelines(out_lines)
    summ = dict(rows_in=n, rows_out=len(out_lines) - 1, rows_time_changed=int(changed.sum()),
                rows_clock_corrected=int((clock_corr != 0).sum()), rows_retimed_here=int((tc_new != 0).sum()),
                clock_flags=pd.Series(fl).value_counts().to_dict(),
                by_epoch={k: dict(n=int(len(g)), min_corr_s=float(g.min()), max_corr_s=float(g.max()),
                                  median_corr_s=float(np.median(g)))
                          for k, g in pd.Series(clock_corr[ep != ""]).groupby(ep[ep != ""])},
                new_columns=new_cols)
    prov = provenance({"picks": picks_path, "clock_model": clock_model, "retime_corrections": retime_csv},
                      {"output": {"path": str(output), "md5": md5(output), "bytes": Path(output).stat().st_size},
                       "summary": summ, "config": cfg.get("_config_path")})
    if clock_model:
        prov["clock_model_epochs"] = {s: [{k: e[k] for k in ("epoch_id", "model", "t_start_utc", "t_end_utc", "size",
                                                              "size_sd", "size_units", "t_start_sd_days")}
                                          for e in eps] for s, eps in load_model(clock_model)["stations"].items()}
    write_json(prov, str(output) + ".provenance.json")
    return summ, prov


def cli_apply(cfg, a):
    picks = a.picks or cfg["picks"]["path"]
    summ, prov = apply_corrections(cfg, picks, a.clock_model, a.retime_corrections, a.output)
    print(f"apply: {summ['rows_in']} rows in, {summ['rows_out']} out, {summ['rows_time_changed']} times changed "
          f"({summ['rows_clock_corrected']} clock, {summ['rows_retimed_here']} re-timing); flags {summ['clock_flags']}")
    print(f"output {a.output} md5 {prov['output']['md5']}")
    return 0
