"""YAML configuration: defaults, loading, and the merged view used by every command."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "project": "unnamed",
    "output_dir": "pick_timing_out",
    "station_id": "{sta}.{net}",
    "workers": 4,
    # tables (path + read_kwargs + columns); only those a command needs must be present
    "picks": None,
    "origins": None,
    "residuals": None,
    "stations": None,
    "cc": None,
    "waveforms": {
        "reader": "fdsn",                 # sqlite_index | fdsn | sds | routed
        "fdsn": {"client": "IRIS"},
    },
    "scan": {
        "channels": ["HH?", "BH?", "EH?"],   # channel patterns scanned (vertical selection below)
        "components": "Z3",
        "drift_threshold_s": 0.005,       # half a sample at 100 Hz
        "tear_samples": 0.5,              # a record start off by more than this many samples = time tear / restart
    },
    "retime": {
        "channel_priority": ["HH", "BH", "EH"],   # picker rule: first band with a vertical
        "alternate_rules": [["EH"]],               # also try these band lists (the picker may have used EH only)
        "components": "Z3",
        "day_seconds": 86400,
        "nominal_rate": 100.0,
        "grid_tolerance": 1e-3,            # fraction of a sample: pick must sit on the merged trace's grid
        "apply_threshold_s": 0.005,        # correct a station-day only if its header-vs-count error reaches this
        "value_window": 50,                # samples each side compared between day trace and short read
    },
    "detect": {
        "neighbor_km": 100.0,              # reference stations for the network-relative residual
        "min_neighbors": 3,
        "fallback": "event",               # event | none: reference when fewer neighbours
        "same_phase": True,
        "min_weight": 0.0,                 # residual rows with weight <= this are ignored
        "max_abs_residual_s": 3.0,
        "bin_picks": 20,
        "bin_max_days": 7.0,
        "bin_min_picks": 3,
        "min_picks": 60,
        "min_bins": 12,
        "grid_max": 160,
        "min_epoch_days": 2.0,
        "min_bins_in_epoch": 3,
        "models": ["step", "offset_epoch", "drift_open", "drift_reset"],
        "dbic_min": 30.0,
        "amp_min_s": 0.05,
        "p_max": 0.01,
        "n_perm": 200,
        "n_boot": 200,
        "colocated_event_km": 5.0,
        "reset_tolerance_days": 1.0,
        "false_alarm": {"n_synthetic": 2, "n_permuted": 1, "inject": [[0.004, 60.0], [0.008, 90.0]]},
        "seed": 0,
    },
    "cc_refine": {"min_rows": 10, "min_consensus": 5, "huber_s": 0.03, "accepted_only": True},
    "model": {"near_start_sigmas": 2.0},
    "apply": {"time_decimals": None, "corrected_column": None},
    "validate": {"colocated_km": 3.0, "slope_min_rows": 20},
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | None, overrides: dict | None = None) -> dict:
    user = {}
    if path:
        user = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _merge(DEFAULTS, user)
    cfg = _merge(cfg, overrides or {})
    cfg["_config_path"] = str(Path(path).resolve()) if path else None
    return cfg


def out_path(cfg: dict, *parts) -> Path:
    p = Path(cfg["output_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
