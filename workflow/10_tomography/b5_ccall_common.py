"""CC differential times for every clustered family of the v5.1 catalog (D-B093, NLL-coherence stage of v6.0).

Shared settings of b5_ccall_*.py. The measurement itself is the tested D-B040/D-B078 machinery of b5_cc_*.py
(branch tomo/cc-dt; README /wd1/mdenolle_data/tomo_v1/cc_dt/README.md): pnwstore short reads on the
record-header time base (b5_cc_common), the same band-pass, windows, lag rule, sub-sample fit and acceptance
(b5_cc_params.yaml). What changes here is the scope (all families + neighbours), the pick file (v4.1tc, D-B090
clock-corrected), the NC/BK route (NCEDC FDSN, as utils/data_client.py) and the job layout (tiles of a spatially
ordered event list, so that a worker holds at most two groups of events in memory).

Time bases. pnwstore snippets are on the record-header time base. The v4.1tc picks are on the header time base
except the 1,024 MH07.XD / MG07.XD picks corrected for the free-running clocks (clock_correction_s = -c(t), added
to the pick time). The header-time pick is t_hdr = t_pick - clock_correction_s; windows are placed on t_hdr and
CC measures dt_cc_hdr on the header base. dt_cc = dt_cc_hdr - clock_dt_s with clock_dt_s = c1 - c2 puts the CC dt
on the v4.1tc time base (the base of the picks NonLinLoc uses); dt_cat = (t1 - T0_1) - (t2 - T0_2) from v4.1tc.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CCALL_OUT", "/wd1/mdenolle_data/tomo_v1/cc_dt_all"))
TASKS = OUT / "tasks"
# snippet caches do not depend on CCALL_OUT (a test run fills the same cache as the full run; same content)
SNIP = Path("/wd1/mdenolle_data/tomo_v1/cc_dt_all/snippets")        # pnwstore, <ID // 1000>/<ID>.{P,S}.mseed, meta
SNIP_NC = Path("/wd1/mdenolle_data/tomo_v1/cc_dt_all/snippets_ncedc")   # NC/BK from NCEDC FDSN, same layout
SNIP_OLD = Path("/wd1/mdenolle_data/tomo_v1/cc_dt/snippets")   # D-B078 cache: read only, never written
WORK = OUT / "work"
STATE = OUT / "state"
QC = OUT / "qc"

PICKS = Path("/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/Cascadia_picks_catalog_v4.1tc.csv")
PICKS_MD5 = "1b21ab1c2cf5aba8b48c64a58fc37356"
CORR = Path("/wd1/mdenolle_data/tomo_v1/checks/pick_retiming/corrections.csv.gz")
CLOCK_MODEL = Path("/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/clock_model.json")
CATALOG = Path("/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv")
CLUSTERS = Path("/wd1/mdenolle_data/catalog_v5/v5.1_analysis/clusters/events_zbz_v51.csv.gz")
# the v5.1 families (2,097) committed on main by PR #42 (blob 4378a0aa); the CC job of 2026-11-25 read the same file from
# the former worktree /home/mdenolle/wt_a51_clusters, which no longer exists (pipeline, D-B110)
FAMILIES = HERE.parents[1] / "data/catalog_v5/v5.1_analysis/clusters/families_zbz_v51.csv"
PARAMS = HERE / "b5_cc_params.yaml"

# scope and pairs (D-B093 brief): neighbours within 3 km of a family event; kNN as in b5_cc_tasks (D-B078 added pairs)
NEIGH_R_KM = 3.0
ADD_K, ADD_R_KM, ADD_NMIN = 20, 3.0, 4

# blocks = v5.1 regions; Mount St Helens split off the Cascades (its own band, 2-12 Hz, b5_cc_params.yaml msh)
REGION_BLOCK = {"Axial": "axial", "Blanco": "blanco", "Cascades": "cascades", "Endeavour": "endeavour",
                "Explorer": "explorer", "Gorda": "gorda", "JdF ridge/plate": "jdf", "Mendocino": "mendocino",
                "Offshore forearc": "forearc", "Other onshore": "onshore", "Puget/Georgia": "puget"}
MSH_LAT, MSH_LON, MSH_R_KM = 46.200, -122.186, 15.0     # crater; events within 15 km (epicentral) -> block msh
BLOCKS = ["axial", "blanco", "msh", "cascades", "endeavour", "explorer", "gorda", "jdf", "mendocino", "forearc",
          "onshore", "puget"]
BAND = {"msh": [2.0, 12.0]}                  # every other block: common band of b5_cc_params.yaml (3-15 Hz)
NCEDC_NETS = {"NC", "BK"}
OBS_NETS = {"OO", "NV", "Z5", "7D", "X9", "7A", "ZZ", "YS"}    # seafloor, as b5_cc_write

R_EARTH = 6371.0


def xyz(lat, lon, depth):
    """3-D Cartesian (km) on a spherical Earth, depth km below sea level: chord distances = 3-D separations."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    r = R_EARTH - np.asarray(depth, float)
    return np.c_[r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)]


def epi_km(lat, lon, lat0, lon0):
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin((la - np.radians(lat0)) / 2) ** 2 + np.cos(la) * np.cos(np.radians(lat0)) * np.sin((lo - np.radians(lon0)) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def shard(eid):
    return f"{int(eid) // 1000:03d}"


def snip_dir(base, eid):
    return base / shard(eid)


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return float("nan")


def log(msg, f=None):
    s = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(s, flush=True)
    if f:
        with open(f, "a") as fh:
            fh.write(s + "\n")


def write_state(stage, **kw):
    STATE.mkdir(parents=True, exist_ok=True)
    p = STATE / f"{stage}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(kw, updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(old, indent=1, default=str))
    os.replace(tmp, p)


def read_state(stage):
    p = STATE / f"{stage}.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------------------------------- station clock model
def clock_model():
    """Confirmed clock epochs (D-B090, tools/pick_timing clock_model.json): c(t) = header - true time,
    drift_reset: c = size * (t - t_start) / 86400 for t_start <= t < t_end, 0 elsewhere."""
    return json.loads(CLOCK_MODEL.read_text())["stations"]


def clock_c(station, t, M=None):
    M = M or clock_model()
    t = np.asarray(t, float)
    c = np.zeros(len(t))
    for e in M.get(station, []):
        assert e["model"] == "drift_reset" and e["size_units"] == "s/day"
        m = (t >= e["t_start_s"]) & (t < e["t_end_s"])
        c[m] = e["size"] * (t[m] - e["t_start_s"]) / 86400.0
    return c


STEP = 0.02     # s of modelled drift per sub-epoch (as b5_cc_clock.py)


def clock_epoch_edges(M=None):
    """{station: edges (POSIX)} of the sub-epochs of each drift (offset change <= 20 ms each), as b5_cc_clock."""
    M = M or clock_model()
    out = {}
    for st, eps in M.items():
        for e in eps:
            dur = (e["t_end_s"] - e["t_start_s"]) / 86400.0
            n = int(np.ceil(abs(e["size"]) * dur / STEP))
            out[st] = np.array([e["t_start_s"] + (e["t_end_s"] - e["t_start_s"]) * k / n for k in range(n + 1)])
    return out


def clock_epoch_id(t, edges):
    k = np.searchsorted(edges, np.asarray(t, float), side="right")
    return np.where((k >= 1) & (k <= len(edges) - 1), k, 0)


def load_params(block):
    import yaml
    P = yaml.safe_load(PARAMS.read_text())
    cfg = dict(P["common"])
    if block in BAND:
        cfg["band"] = BAND[block]
    return cfg, P["threshold"]
