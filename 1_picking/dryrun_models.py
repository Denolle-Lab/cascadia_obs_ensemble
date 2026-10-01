"""v5 base-model dry run: pick station-days with option (a) and option (b) of V5_PLAN §7.

One pass per station-day: the six EQTransformer weights (the five of v3 plus
OBSTransformer ``obst2024``) predict once, then three ensembles are formed from the
same predictions and picked with the v3 trigger (semblance > 0.05):

    a  original, ethz, instance, scedc, stead          (v3/v4 set)
    b  a + obst2024                                     (option b)
    o  obst2024 alone

Channels follow the v5 rule (V5_PLAN §4.6): HH, else BH, else EH; vertical Z or 3;
Z copied to the horizontals when a horizontal is missing. Preprocessing is that of
the region runs (4-15 Hz, 100 Hz). Semblance is the vectorized form of ELEP's
``ensemble_semblance`` (V5_PLAN §4.1); ``--check-elep K`` repeats option (a) with
ELEP's own function on the first K station-days and records whether the picks agree.

    python 1_picking/dryrun_models.py --tasks tasks.csv --out DIR --workers 30
tasks.csv: network,station,day (YYYY-MM-DD). Output: DIR/picks/NET.STA.DAY.csv
(column ``config``) and DIR/log.csv (one row per station-day). Resumable.
"""
import argparse
import logging
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.ndimage as nd
import seisbench.models as sbm
import torch
from obspy import Stream, UTCDateTime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import elep_picker as ep  # noqa: E402
from ELEP.elep.ensemble_coherence import ensemble_semblance  # noqa: E402
from utils.data_client import get_waveforms  # noqa: E402

Logger = logging.getLogger(__name__)

MODELS = ep.PRETRAIN_LIST + ["obst2024"]
STD_NORM = {"original", "obst2024"}            # the others use peak normalization
CONFIGS = {"a": ep.PRETRAIN_LIST, "b": MODELS, "o": ["obst2024"]}
BATCH = 16
VERTICAL = "[3Z]"

_models = None


def load(name):
    if name == "obst2024":
        m = sbm.OBSTransformer.from_pretrained("obst2024")
        m.to(ep.device).eval()
        return m
    return ep.load_model(name)


def init_worker():
    global _models
    torch.set_num_threads(1)
    logging.getLogger("seisbench").setLevel(logging.ERROR)
    _models = {n: load(n) for n in MODELS}


def select_v5(st):
    """HH, else BH, else EH, among bands that have a vertical; Z copied if a horizontal is missing."""
    for band in ("HH", "BH", "EH"):
        sel = st.select(channel=f"{band}?")
        if sel.select(channel=f"??{VERTICAL}"):
            return sel, band
    return Stream(), None


def windows(arr):
    """ELEP windows, as in elep_picker.ensemble_semblance_traces: (std, peak) normalized."""
    npts = arr.shape[1]
    nseg = int(np.floor((npts - ep.TWIN) / ep.STEP)) + 1
    idx = np.arange(nseg)[:, None] * ep.STEP + np.arange(ep.TWIN)[None, :]
    w = arr[:, idx].transpose(1, 0, 2).astype(np.float32)          # (nseg, 3, TWIN)
    w -= w.mean(axis=-1, keepdims=True)
    tap = 0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, 6)))
    w_std = w / np.std(w, axis=(1, 2), keepdims=True) + 1e-10
    with np.errstate(invalid="ignore", divide="ignore"):
        w_max = w / np.max(np.abs(w), axis=-1, keepdims=True)
    for x in (w_std, w_max):
        x[:, :, :6] *= tap
        x[:, :, -6:] *= tap[::-1]
    return w_std, w_max, nseg, npts


def predict(w_std, w_max, nseg):
    pred = np.zeros([2, len(MODELS), nseg, ep.TWIN], dtype=np.float32)
    with torch.inference_mode():
        for i, name in enumerate(MODELS):
            x = w_std if name in STD_NORM else w_max
            for b in range(0, nseg, BATCH):
                out = _models[name](torch.from_numpy(x[b:b + BATCH]))
                pred[0, i, b:b + BATCH] = out[1].numpy()
                pred[1, i, b:b + BATCH] = out[2].numpy()
    return pred


def semblance_vec(sig, dt, order=2, win=0.5):
    """ELEP ensemble_semblance (window, weight 'max') for all windows at once. sig: (ntr, nseg, npts)."""
    ntr, n = sig.shape[0], int(win / dt)
    sq = np.sum(sig, axis=0) ** 2
    ss = np.sum(sig ** 2, axis=0)
    num = nd.uniform_filter1d(sq.astype(np.float64), n, axis=-1, mode="constant") * n
    den = ntr * nd.uniform_filter1d(ss.astype(np.float64), n, axis=-1, mode="constant") * n
    with np.errstate(invalid="ignore", divide="ignore"):
        s0 = (num / den).astype(np.float32)
    return s0 ** order * np.amax(sig, axis=0)


def semblance_elep(sig, dt):
    paras = dict(ep.PARAS_SEMBLANCE, dt=dt)
    return np.stack([ensemble_semblance(sig[:, i, :], paras) for i in range(sig.shape[1])])


def picks_for(pred, members, dt, npts, nseg, tr0, semb=semblance_vec):
    ii = [MODELS.index(m) for m in members]
    out = []
    for k, (label, thr) in enumerate((("P", ep.P_THRD), ("S", ep.S_THRD))):
        smb = semb(pred[k, ii], dt)
        stack = ep.stacking(smb, npts, ep.L_BLND, ep.R_BLND, nseg)
        out.append(ep.pred_trigger_pick(stack, tr0, label, thrd=thr))
    return pd.concat(out, ignore_index=True)


def run_one(task):
    net, sta, day, outdir, check = task
    fn = os.path.join(outdir, "picks", f"{net}.{sta}.{day}.csv")
    rec = {"network": net, "station": sta, "day": day, "status": "ok"}
    t0 = time.time()
    try:
        t1 = UTCDateTime(day)
        raw = get_waveforms(network=net, station=sta, channel="?H?", starttime=t1,
                            endtime=t1 + 86400, source="pnwstore")
        sdata, band = select_v5(raw)
        rec["band"] = band
        if not sdata:
            rec["status"] = "no_vertical"
            return rec
        if np.abs(np.mean(np.diff(sdata.select(channel=f"??{VERTICAL}")[0].data))) <= 1e-8:
            rec["status"] = "constant"
            return rec
        sdata.filter(type="bandpass", freqmin=4, freqmax=15)
        sdata.resample(100)
        sdata.merge(fill_value="interpolate")
        a = max(tr.stats.starttime for tr in sdata)
        b = min(tr.stats.endtime for tr in sdata)
        for tr in sdata:
            tr.trim(starttime=a, endtime=b, nearest_sample=True)
        has_h = [bool(sdata.select(channel=f"??{c}")) for c in ("[1E]", "[2N]")]
        if all(has_h):
            sdata = ep.order_components(sdata, raw, "hh_bh_eh", VERTICAL)
        else:
            z = sdata.select(channel=f"??{VERTICAL}")[0]
            sdata = Stream([z, z.copy(), z.copy()])
        rec["z_copied"] = not all(has_h)
        n = min(len(tr.data) for tr in sdata[:3])     # trim can leave one-sample differences
        arr = np.array([tr.data[:n] for tr in sdata[:3]])
        dt = sdata[0].stats.delta
        rec["t_pre"] = round(time.time() - t0, 1)

        t = time.time()
        w_std, w_max, nseg, npts = windows(arr)
        pred = predict(w_std, w_max, nseg)
        del w_std, w_max
        rec["t_infer"] = round(time.time() - t, 1)

        t = time.time()
        dfs = []
        for cfg, members in CONFIGS.items():
            d = picks_for(pred, members, dt, npts, nseg, sdata[0])
            d["config"] = cfg
            dfs.append(d)
            rec[f"n_{cfg}"] = len(d)
        rec["t_semb"] = round(time.time() - t, 1)
        if check:
            t = time.time()
            ref = picks_for(pred, CONFIGS["a"], dt, npts, nseg, sdata[0], semb=semblance_elep)
            mine = dfs[0]
            same = (len(ref) == len(mine) and
                    (ref.pick_time.astype(str).values == mine.pick_time.astype(str).values).all())
            rec["elep_check"] = "same" if same else f"differ {len(ref)} vs {len(mine)}"
            rec["t_elep"] = round(time.time() - t, 1)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        pd.concat(dfs, ignore_index=True).to_csv(fn + ".tmp", index=False)
        os.replace(fn + ".tmp", fn)
    except Exception as e:  # keep going; the log records why
        rec["status"] = f"error: {type(e).__name__}: {e}"[:300]
        Logger.debug(traceback.format_exc())
    rec["t_total"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--check-elep", type=int, default=0)
    a = ap.parse_args()
    tasks = pd.read_csv(a.tasks, dtype=str)
    logf = os.path.join(a.out, "log.csv")
    done = set()
    if os.path.exists(logf):
        lg = pd.read_csv(logf, dtype=str)
        done = set(zip(lg.network, lg.station, lg.day))
    todo = [(r.network, r.station, r.day, a.out, i < a.check_elep)
            for i, r in enumerate(tasks.itertuples()) if (r.network, r.station, r.day) not in done]
    print(f"{len(todo)} station-days to do ({len(done)} done)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    cols = ["network", "station", "day", "status", "band", "z_copied", "n_a", "n_b", "n_o",
            "t_pre", "t_infer", "t_semb", "elep_check", "t_elep", "t_total"]
    new = not os.path.exists(logf)
    with Pool(a.workers, initializer=init_worker) as pool, open(logf, "a") as f:
        if new:
            f.write(",".join(cols) + "\n")
        for k, rec in enumerate(pool.imap_unordered(run_one, todo), 1):
            f.write(",".join(str(rec.get(c, "")).replace(",", ";") for c in cols) + "\n")
            f.flush()
            if k % 20 == 0 or k == len(todo):
                print(f"{time.strftime('%H:%M:%S')} {k}/{len(todo)}", flush=True)
    open(os.path.join(a.out, "DONE"), "w").close()


if __name__ == "__main__":
    main()
