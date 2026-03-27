"""
utils/data_client.py — Centralized waveform data access

Provides a single get_waveforms() function that routes to the correct
backend depending on the *source* argument:

    source='fdsn'     → EarthScope FDSN (public; works anywhere)
    source='pnwstore' → UW pnwstore archive for PNW networks;
                        NC/BK networks are fetched via NCEDC FDSN instead

Usage
-----
from utils.data_client import get_waveforms

# Public (EarthScope)
st = get_waveforms('UW', 'TUCA', '*', starttime=t0, endtime=t1)

# UW internal
st = get_waveforms('UW', 'TUCA', '*', starttime=t0, endtime=t1,
                   source='pnwstore')
"""

from __future__ import annotations

import logging
from datetime import datetime

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client

Logger = logging.getLogger(__name__)

# Networks that are not in pnwstore — fetch from NCEDC FDSN instead
_NCEDC_NETWORKS = frozenset(['NC', 'BK'])

# Guard pnwstore import — only available in the 'internal' pixi environment
try:
    from pnwstore.mseed import WaveformClient as _WaveformClient
    _pnwstore_available = True
except ImportError:
    _WaveformClient = None  # type: ignore[assignment,misc]
    _pnwstore_available = False


def get_waveforms(
    network: str,
    station: str,
    channel: str,
    starttime,
    endtime,
    location: str = "*",
    source: str = "fdsn",
    fdsn_client: str = "IRIS",
) -> Stream:
    """Download waveforms from the appropriate backend.

    Parameters
    ----------
    network, station, channel, location:
        Standard SEED identifiers.
    starttime, endtime:
        ``obspy.UTCDateTime`` or any value accepted by ``UTCDateTime()``.
    source:
        ``'fdsn'``  — EarthScope FDSN (default; public).
        ``'pnwstore'`` — UW waveform archive (requires pnwstore install).
    fdsn_client:
        FDSN datacenter name used when *source* is ``'fdsn'`` or when
        routing NC/BK networks via NCEDC.  Default ``'IRIS'``.

    Returns
    -------
    obspy.Stream
    """
    t0 = UTCDateTime(starttime)
    t1 = UTCDateTime(endtime)

    if source == "fdsn":
        client = Client(fdsn_client)
        return client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=t0,
            endtime=t1,
            attach_response=True,
        )

    if source == "pnwstore":
        if not _pnwstore_available:
            raise RuntimeError(
                "pnwstore is not installed. "
                "Install the 'internal' pixi environment with:\n"
                "    pixi install --environment internal\n"
                "or: pip install git+https://github.com/niyiyu/pnwstore.git"
            )

        if network in _NCEDC_NETWORKS:
            # pnwstore does not hold NC/BK data — use NCEDC FDSN
            client_ncedc = Client("NCEDC")
            return client_ncedc.get_waveforms(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=t0,
                endtime=t1,
            )

        # Convert UTCDateTime to datetime for pnwstore year/month/day API
        dt: datetime = t0.datetime
        client_waveform = _WaveformClient()
        st = client_waveform.get_waveforms(
            network=network,
            station=station,
            channel=channel,
            year=dt.strftime("%Y"),
            month=dt.strftime("%m"),
            day=dt.strftime("%d"),
        )
        # Trim to the requested window (pnwstore returns full day volumes)
        st.trim(starttime=t0, endtime=t1)
        return st

    raise ValueError(f"Unknown source '{source}'. Choose 'fdsn' or 'pnwstore'.")


SOURCES = ["fdsn", "pnwstore"]


if __name__ == "__main__":
    # Quick self-test (FDSN only — no pnwstore required)
    import sys

    logging.basicConfig(level=logging.INFO)
    Logger.info("pnwstore available: %s", _pnwstore_available)

    t0 = UTCDateTime("2011-03-01T00:00:00")
    t1 = t0 + 60

    try:
        st = get_waveforms("UW", "TUCA", "HH?", t0, t1, source="fdsn")
        Logger.info("FDSN test OK — %d trace(s)", len(st))
    except Exception as e:
        Logger.warning("FDSN test failed (network may be unavailable in CI): %s", e)

    if _pnwstore_available:
        try:
            st = get_waveforms("UW", "TUCA", "HH?", t0, t1, source="pnwstore")
            Logger.info("pnwstore test OK — %d trace(s)", len(st))
        except Exception as e:
            Logger.warning("pnwstore test failed: %s", e)

    Logger.info("data_client self-test complete")
    sys.exit(0)
