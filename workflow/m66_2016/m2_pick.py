"""M66 fix, step 2: ELEP ensemble picks on the downloaded windows.

Same preprocessing and picker as workflow/01_picking/elep_picker.py:run_detection (channel_mode
hh_bh_eh, vertical [3Z], 4-15 Hz bandpass, 100 Hz, merge, common trim, ELEP semblance of 5 EQTransformer
models, 0.05 thresholds), but reading the local MiniSEED windows instead of fetching day files.
Difference: a station with only a vertical channel gets Z copied to the three inputs (as the eh_only
runs did) instead of being skipped; stations with only HN (strong motion) are picked on HN (flagged).

Output: /wd1/mdenolle_data/m66_2016/picks/<win>_picks.csv (all picks, picker columns + dist_km)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from obspy import read, Stream

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_picking"))
import elep_picker as E  # noqa: E402

OUT = Path("/wd1/mdenolle_data/m66_2016")
FIX = Path(__file__).resolve().parents[2] / "data/catalog_v4/fixes/m66_2016"


def pick_stream(st, models):
    sdata = E.select_channels(st, "hh_bh_eh", "[3Z]")
    hn = False
    if len(sdata) == 0 and st.select(channel="HN?"):
        sdata, hn = st.select(channel="HN?").copy(), True
    if len(sdata) == 0:
        return None, "no HH/BH/EH/HN vertical"
    sdata = sdata.copy()
    # one location code only (the first, sorted)
    locs = sorted({tr.stats.location for tr in sdata})
    sdata = Stream([tr for tr in sdata if tr.stats.location == locs[0]])
    if np.abs(np.mean(sdata[0].data[1:] - sdata[0].data[:-1])) <= 1e-8:
        return None, "constant data"
    sdata.detrend("demean")
    sdata.filter(type="bandpass", freqmin=4, freqmax=15)
    sdata.resample(100)
    sdata.merge(fill_value="interpolate")
    t0 = max(tr.stats.starttime for tr in sdata)
    t1 = min(tr.stats.endtime for tr in sdata)
    for tr in sdata:
        tr.trim(t0, t1, nearest_sample=True)
    sdata = E.order_components(sdata, st, "hh_bh_eh", "[3Z]")
    if len(sdata) < 3:   # Z only: copy Z to the horizontals (as the eh_only runs did)
        z = sdata.select(channel="??[3Z]")[0]
        sdata = Stream([z, z.copy(), z.copy()])
    sdata = sdata[:3]
    npts = min(tr.stats.npts for tr in sdata)
    arr = np.array([tr.data[:npts] for tr in sdata], dtype=np.float32)
    if npts < E.TWIN:
        return None, "too short"
    smb_p, smb_s = E.ensemble_semblance_traces(arr, 0.01, models)
    df = pd.concat([E.pred_trigger_pick(smb_p, sdata[0], "P", thrd=E.P_THRD),
                    E.pred_trigger_pick(smb_s, sdata[0], "S", thrd=E.S_THRD)], ignore_index=True)
    df["strong_motion_only"] = hn
    return df, f"ok {sdata[0].stats.channel[:2]} {len(df)} picks"


def main(names):
    models = E.load_models()
    (OUT / "picks").mkdir(exist_ok=True)
    for name in names:
        sta = pd.read_csv(FIX / f"stations_requested_{name}.csv")
        out, log = [], []
        for r in sta[sta.returned].itertuples():
            st = read(str(OUT / "waveforms" / name / f"{r.network}.{r.station}.mseed"))
            try:
                df, msg = pick_stream(st, models)
            except Exception as ex:
                df, msg = None, f"error {type(ex).__name__}: {ex}"
            print(name, r.network, r.station, f"{r.dist_km:.0f} km", msg, flush=True)
            log.append(dict(network=r.network, station=r.station, status=msg))
            if df is not None and len(df):
                df["network"], df["station"] = r.network, r.station   # trace id may carry location
                df["dist_km"] = r.dist_km
                out.append(df)
        pd.concat(out).to_csv(OUT / "picks" / f"{name}_picks.csv", index=False)
        pd.DataFrame(log).to_csv(OUT / "picks" / f"{name}_pick_log.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
