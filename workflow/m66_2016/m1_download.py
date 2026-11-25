"""M66 fix, step 1: find stations within RADIUS km of each target event (NCEDC + IRIS FDSN),
download MiniSEED windows and StationXML.

Writes /wd1/mdenolle_data/m66_2016/waveforms/<win>/NET.STA.mseed, stationxml/<win>/NET.STA.xml,
and data/catalog_v4/fixes/m66_2016/stations_requested_<win>.csv (requested vs returned).
"""
import sys
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

OUT = Path("/wd1/mdenolle_data/m66_2016")
FIX = Path(__file__).resolve().parents[2] / "data/catalog_v4/fixes/m66_2016"
RADIUS_DEG = 4.0      # station search radius (deg, ~445 km); picks used to 400 km
WINDOWS = {   # name: (start, end, centre lat, lon)
    "w20161208_mw66": ("2016-12-08T14:40:00", "2016-12-08T15:10:00", 40.4535, -126.1936),
    "w20161208_m47": ("2016-12-08T16:25:00", "2016-12-08T16:45:00", 40.4292, -126.3674),
    "w20160130_gorda": ("2016-01-30T00:20:00", "2016-01-30T00:45:00", 40.98, -127.30),
}
CHA = "HH?,BH?,EH?,HN?"
known = pd.read_csv("/wd1/mdenolle_data/picks_share/stations/stations_ver5_v6.csv")
known_ids = set(known.network + "." + known.station)


def main(names):
    clients = {"NCEDC": Client("NCEDC", timeout=40), "IRIS": Client("EARTHSCOPE", timeout=40)}
    for name in names:
        t0, t1, la, lo = WINDOWS[name]
        t0, t1 = UTCDateTime(t0), UTCDateTime(t1)
        (OUT / "waveforms" / name).mkdir(parents=True, exist_ok=True)
        (OUT / "stationxml" / name).mkdir(parents=True, exist_ok=True)
        rows = {}
        invs = {}
        for cn, c in clients.items():
            try:
                inv = c.get_stations(latitude=la, longitude=lo, maxradius=RADIUS_DEG, starttime=t0,
                                     endtime=t1, channel=CHA, level="channel")
            except Exception as ex:
                print(cn, "station query failed:", ex)
                continue
            for net in inv:
                for sta in net:
                    sid = f"{net.code}.{sta.code}"
                    if sid in rows:
                        continue
                    if cn == "IRIS" and net.code in ("NC", "BK"):
                        continue   # NC/BK from NCEDC (as in utils/data_client.py)
                    d = gps2dist_azimuth(la, lo, sta.latitude, sta.longitude)
                    chans = sorted({ch.code for ch in sta})
                    rows[sid] = dict(network=net.code, station=sta.code, lat=sta.latitude,
                                     lon=sta.longitude, elev_m=sta.elevation, dist_km=d[0] / 1e3,
                                     az=d[1], channels=",".join(chans), datacenter=cn,
                                     in_stations_ver5_v6=sid in known_ids)
                    invs[sid] = inv.select(network=net.code, station=sta.code)
                    if d[0] / 1e3 > RADIUS_DEG * 111.19:
                        del rows[sid]
        def fetch(item):
            sid, r = item
            f = OUT / "waveforms" / name / f"{sid}.mseed"
            try:
                inv = clients[r["datacenter"]].get_stations(network=r["network"], station=r["station"],
                                                            starttime=t0, endtime=t1, channel=CHA,
                                                            level="response")
            except Exception:
                inv = invs[sid]
            inv.write(str(OUT / "stationxml" / name / f"{sid}.xml"), format="STATIONXML")
            if f.exists():
                st = read(str(f))
            else:
                try:
                    st = clients[r["datacenter"]].get_waveforms(r["network"], r["station"], "*", CHA,
                                                                t0, t1)
                except Exception as ex:
                    st = None
                    r["error"] = type(ex).__name__
                if st:
                    st.write(str(f), format="MSEED")
            r["returned_channels"] = ",".join(sorted({tr.stats.channel for tr in st})) if st else ""
            r["returned"] = bool(st)
            print(name, sid, f"{r['dist_km']:.0f} km", r["returned_channels"] or r.get("error"), flush=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(fetch, sorted(rows.items(), key=lambda kv: kv[1]["dist_km"])))
        pd.DataFrame(rows.values()).sort_values("dist_km").to_csv(
            FIX / f"stations_requested_{name}.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main(sys.argv[1:] or list(WINDOWS))
