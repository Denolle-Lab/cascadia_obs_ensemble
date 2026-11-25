"""Re-timing of a synthetic day-merged trace, and the header scan of the same day."""
import tempfile
import unittest

import numpy as np
import pandas as pd

from pick_timing import headers, synth
from pick_timing.config import DEFAULTS
from pick_timing.readers import SDSReader
from pick_timing.retime import retime_day

DAY0 = pd.Timestamp("2020-01-01").timestamp()


def _merged_start(fn):
    import obspy
    from obspy import UTCDateTime
    st = obspy.read(fn).trim(UTCDateTime(DAY0), UTCDateTime(DAY0 + 86400))
    st.merge(method=0, fill_value=0)
    return st[0]


class TestRetime(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        # records of 100 samples taken at 99.8 Hz, headers right; merged at 100 Hz -> 2 ms drift per record
        self.fn, self.starts = synth.drifting_day_mseed(self.root, seconds=600, true_rate=99.8)
        self.q = dict(DEFAULTS["retime"], components="Z3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_day_merge_drift_recovered(self):
        tr = _merged_start(self.fn)
        self.assertEqual(len(self.starts) * 100, tr.stats.npts)  # one merged segment
        t_m = float(tr.stats.starttime)
        idx = np.array([150, 20_000, 45_123, 59_000])
        picks = pd.DataFrame({"pick_row": np.arange(len(idx)), "event_id": 1, "phase": "P",
                              "time_s": t_m + idx / 100.0})
        out = pd.DataFrame(retime_day(SDSReader(self.root), "XX", "DRFT", DAY0, picks, self.q))
        r, pos = idx // 100, idx % 100
        truth = self.starts[r] + pos / 100.0 - (t_m + idx / 100.0)
        self.assertTrue((out.flag.isin(["ok", "ok_nonunique"])).all(), out.flag.tolist())
        np.testing.assert_allclose(out.correction_s.values, truth, atol=1e-4)
        self.assertGreater(truth.max(), 1.0)  # the test is not trivial: up to ~1.2 s
        self.assertGreater(out.day_absmax_s.iloc[0], 1.0)

    def test_off_grid_pick_flagged(self):
        tr = _merged_start(self.fn)
        picks = pd.DataFrame({"pick_row": [0], "event_id": 1, "phase": "P",
                              "time_s": [float(tr.stats.starttime) + 12.3456]})
        out = retime_day(SDSReader(self.root), "XX", "DRFT", DAY0, picks, self.q)
        self.assertEqual(out[0]["flag"], "off_grid")

    def test_scan_headers_sees_drift_and_restart(self):
        root2 = tempfile.mkdtemp(dir=self.root)
        synth.drifting_day_mseed(root2, seconds=300, true_rate=99.8, gap_at=100)
        h = SDSReader(root2).day_heads("XX", "DRFT", DAY0)
        m, ev, _ = headers.channel_day_metrics(h, None, 100.0, 0.005, 0.5)
        self.assertTrue(m["drifting"])
        self.assertEqual(m["n_segments"], 2)
        self.assertEqual(len(ev), 1)
        self.assertAlmostEqual(ev[0]["gap_s"], 2.5 + 100 * (1 / 99.8 - 1 / 100), places=3)
        self.assertLessEqual(m["tq_max"], 0)  # ObsPy writes timing quality 0 (or no blockette 1001)


if __name__ == "__main__":
    unittest.main()
