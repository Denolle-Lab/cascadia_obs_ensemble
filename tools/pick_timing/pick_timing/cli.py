"""Command line: python -m pick_timing <command> [options]. Every command takes --config CONFIG.yaml."""
from __future__ import annotations

import argparse
import sys

from .config import load_config

COMMANDS = {
    "scan-headers": "per channel-day record-header continuity vs sample count (day-merge drift), restarts, gaps, "
                    "timing quality",
    "retime": "re-time picks from their own MiniSEED record headers (day-merge drift fix)",
    "extract-nlloc": "NonLinLoc hyp files -> residual table (+ station table)",
    "detect-clock": "network-relative residual screen of every station for clock offsets / drifts, with false-alarm "
                    "estimate",
    "report": "self-contained HTML report + review CSV of the clock candidates",
    "ingest-review": "filled review CSV -> confirmed-epoch file",
    "model-clock": "confirmed epochs -> clock correction model c(t) per station (JSON)",
    "apply": "apply re-timing and clock corrections to a pick file (new columns, never drops a pick, provenance)",
    "validate": "re-fit residuals after correction, co-located station slope test, CC checks, byte identity",
    "relocate-nlloc": "re-locate the events touched by a clock model with NonLinLoc (validation)",
    "make-example": "write the small synthetic example dataset",
}


def _p(sub, name):
    p = sub.add_parser(name, help=COMMANDS[name], description=COMMANDS[name])
    p.add_argument("--config", "-c", default=None, help="YAML config (column mapping, paths, parameters)")
    p.add_argument("--workers", type=int, default=None, help="parallel processes (default: config workers)")
    p.add_argument("--out", default=None, help="output directory (default: config output_dir)")
    return p


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m pick_timing", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = _p(sub, "scan-headers")
    p.add_argument("--stations", help="comma list of station ids (with --start/--end)")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--from-picks", action="store_true", help="station-days that have picks (config picks)")
    p.add_argument("--sample", choices=["all", "weekly", "monthly"], default="all",
                   help="with --from-picks: first pick day per station and week/month only")
    p.add_argument("--networks", help="comma list of networks to keep")
    p.add_argument("--exclude-networks", help="comma list of networks to skip")
    p.add_argument("--headers-only", action="store_true", help="read only record headers (no day-merge metrics)")
    p.add_argument("--tag", default="scan")

    p = _p(sub, "retime")
    p.add_argument("--tasks", help="CSV of station-days (columns station, day) to re-time; default: drifting "
                                   "station-days from a scan-headers output")
    p.add_argument("--scan", help="scan-headers channel_days CSV: re-time the station-days with drift >= threshold")
    p.add_argument("--all-days", action="store_true", help="re-time every station-day with picks (slow)")
    p.add_argument("--tag", default="retime")

    p = _p(sub, "extract-nlloc")
    p.add_argument("--hyp-glob", action="append", required=True,
                   help="glob of per-event *.grid0.loc.hyp files; repeat, later globs win per event")
    p.add_argument("--id-regex", default=r"(\d+)$", help="regex on PUBLIC_ID giving the event id")
    p.add_argument("--stations-glob", default=None, help="optional glob of 'Label x y z lat lon depth' files")
    p.add_argument("--output", default=None, help="residual table path (.parquet or .csv.gz)")

    p = _p(sub, "detect-clock")
    p.add_argument("--residuals", help="residual table (overrides config residuals.path)")
    p.add_argument("--stations-only", help="comma list: screen only these stations (default: every station)")
    p.add_argument("--restarts", help="scan-headers events CSV (restarts) to use as reset times")
    p.add_argument("--cc", help="CC differential-time table to refine candidates (overrides config cc.path)")
    p.add_argument("--no-false-alarm", action="store_true", help="skip the synthetic/permuted false-alarm runs")
    p.add_argument("--start", help="only residuals at or after this time")
    p.add_argument("--end", help="only residuals before this time")

    p = _p(sub, "report")
    p.add_argument("--accept", default="", help="comma list of candidate ids or stations pre-marked accept "
                                                "(already confirmed by a person; say who in --decided-by)")
    p.add_argument("--decided-by", default="", help="reviewer / decision reference written next to --accept")
    p.add_argument("--decision-ref", default="", help="decision reference written next to --accept")
    p.add_argument("--restarts", help="scan-headers events CSV: restarts drawn on the figures")
    p.add_argument("--confirmed", help="confirmed-epoch CSV: candidates it covers are shown, not put up for review")
    p.add_argument("--title", default="Station clock candidates")
    p.add_argument("--notes", default="", help="text shown at the top of the report")
    p.add_argument("--max-stations", type=int, default=200)

    p = _p(sub, "ingest-review")
    p.add_argument("--review", required=True, help="filled review CSV (decision = accept / reject / blank)")
    p.add_argument("--output", default=None)

    p = _p(sub, "model-clock")
    p.add_argument("--epochs", required=True, help="confirmed-epoch CSV (from ingest-review)")
    p.add_argument("--output", default=None)

    p = _p(sub, "apply")
    p.add_argument("--picks", help="pick file (overrides config picks.path)")
    p.add_argument("--clock-model", help="clock model JSON (model-clock)")
    p.add_argument("--retime-corrections", help="corrections CSV from retime (adds timing_correction_s)")
    p.add_argument("--output", required=True, help="corrected pick file")

    p = _p(sub, "validate")
    p.add_argument("--picks-in", required=True)
    p.add_argument("--picks-out", required=True)
    p.add_argument("--clock-model", required=True)
    p.add_argument("--residuals-after", help="residuals of a re-location with the corrected picks")
    p.add_argument("--colocated-km", type=float, default=None, help="station separation for the slope test "
                   "(default: config validate.colocated_km)")
    p.add_argument("--cc", help="CC table (overrides config cc.path)")

    p = _p(sub, "relocate-nlloc")
    p.add_argument("--run-dir", required=True, help="NLLoc run folder: <tile>/nlloc.in and <tile>/events.obs")
    p.add_argument("--clock-model", required=True)
    p.add_argument("--nlloc", required=True, help="NLLoc binary")
    p.add_argument("--residuals", help="residual table of the run (event -> tile via hyp_file)")
    p.add_argument("--id-prefix", default="", help="PUBLIC_ID = prefix + event_id in the obs files")
    p.add_argument("--max-events", type=int, default=None)

    p = _p(sub, "make-example")
    p.add_argument("--seed", type=int, default=1)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    ov = {}
    if a.workers:
        ov["workers"] = a.workers
    if a.out:
        ov["output_dir"] = a.out
    cfg = load_config(a.config, ov)
    if a.cmd == "scan-headers":
        from .headers import cli_scan
        return cli_scan(cfg, a)
    if a.cmd == "retime":
        from .retime import cli_retime
        return cli_retime(cfg, a)
    if a.cmd == "extract-nlloc":
        from .nlloc import cli_extract
        return cli_extract(cfg, a)
    if a.cmd == "detect-clock":
        from .detect import cli_detect
        return cli_detect(cfg, a)
    if a.cmd == "report":
        from .report import cli_report
        return cli_report(cfg, a)
    if a.cmd == "ingest-review":
        from .report import cli_ingest
        return cli_ingest(cfg, a)
    if a.cmd == "model-clock":
        from .clockmodel import cli_model
        return cli_model(cfg, a)
    if a.cmd == "apply":
        from .apply import cli_apply
        return cli_apply(cfg, a)
    if a.cmd == "validate":
        from .validate import cli_validate
        return cli_validate(cfg, a)
    if a.cmd == "relocate-nlloc":
        from .validate import cli_relocate
        return cli_relocate(cfg, a)
    if a.cmd == "make-example":
        from .synth import cli_example
        return cli_example(cfg, a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
