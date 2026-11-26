"""Column-mapping round trip, time handling, clock model, apply (never drops, byte identity), review ingest."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pick_timing import tables
from pick_timing.apply import apply_corrections
from pick_timing.clockmodel import build_model, evaluate, read_epochs
from pick_timing.config import load_config
from pick_timing.report import ingest
from pick_timing.validate import byte_identity

PICKS = """evid,arrival,net,sta,ph,tt,prob
7,2016-03-01T10:00:00.120000Z,XD,MH07,P,3.2000,0.9
7,2016-03-01T10:00:02.500000Z,XD,MG07,S,5.5800,0.8
8,2016-06-01T00:00:00.000000Z,XD,MH07,P,1.0000,0.7
9,2016-01-01T00:00:00.000000Z,UW,TDL,P,2.0000,0.5
"""

CFG = """
station_id: "{net}.{sta}"
picks:
  path: PICKS
  columns: {event_id: evid, time: arrival, network: net, station_code: sta, phase: ph, travel_time: tt}
"""


def _write(tmp, text, name):
    p = Path(tmp) / name
    p.write_text(text)
    return str(p)


class TestColumns(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            pk = _write(tmp, PICKS, "p.csv")
            cfg = load_config(_write(tmp, CFG.replace("PICKS", pk), "c.yaml"))
            raw = tables.read_raw(cfg["picks"])
            can = tables.canonical(raw, cfg["picks"], "picks", cfg)
            self.assertEqual(list(can.station), ["XD.MH07", "XD.MG07", "XD.MH07", "UW.TDL"])
            back = tables.to_file_columns(can.drop(columns=["station"]), cfg["picks"])
            pd.testing.assert_frame_equal(back[raw.columns], raw)

    def test_station_patterns(self):
        n, s = tables.split_station(["MH07.XD", "TDL.UW"], "{sta}.{net}")
        self.assertEqual((list(n), list(s)), (["XD", "UW"], ["MH07", "TDL"]))
        self.assertEqual(tables.join_station(n, s, "{net}.{sta}"), ["XD.MH07", "UW.TDL"])

    def test_time_round_trip(self):
        s = ["2016-02-07T15:46:08.773125Z", "2010-01-01T00:00:00.000001Z"]
        t = tables.to_posix(s)
        self.assertEqual([tables.posix_to_iso(x, 6) for x in t], s)


EPOCHS = """epoch_id,station,model,t_start_utc,t_start_sd_days,t_end_utc,size,size_sd,confirmed_by
XD.MH07#1,XD.MH07,drift_reset,2016-02-01T00:00:00Z,2.0,2016-05-01T00:00:00Z,0.01,0.001,tester
XD.MG07#1,XD.MG07,offset_epoch,2016-02-20T00:00:00Z,0.5,2016-03-20T00:00:00Z,-0.2,0.01,tester
"""


class TestModelApply(unittest.TestCase):
    def test_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = build_model(read_epochs(_write(tmp, EPOCHS, "e.csv")))
            t = tables.to_posix(["2016-03-02T00:00:00Z", "2016-05-02T00:00:00Z", "2016-03-01T00:00:00Z"])
            ev = evaluate(m, ["XD.MH07", "XD.MH07", "XD.MG07"], t)
            np.testing.assert_allclose(ev.clock_c_s, [0.30, 0.0, -0.2])
            self.assertEqual(list(ev.clock_flag), ["drift", "none", "offset"])
            self.assertAlmostEqual(ev.clock_sd_s[0], np.hypot(30 * 0.001, 0.01 * 2.0))

    def test_overlap_rejected(self):
        bad = EPOCHS + "XD.MH07#2,XD.MH07,offset_epoch,2016-04-01T00:00:00Z,0,2016-06-01T00:00:00Z,0.1,0,x\n"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_model(read_epochs(_write(tmp, bad, "e.csv")))

    def test_apply_never_drops_and_keeps_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pk = _write(tmp, PICKS, "p.csv")
            cfg = load_config(_write(tmp, CFG.replace("PICKS", pk), "c.yaml"))
            ep = _write(tmp, EPOCHS, "e.csv")
            mp = str(Path(tmp) / "m.json")
            tables.write_json(build_model(read_epochs(ep), ep), mp)
            out = str(Path(tmp) / "out.csv")
            summ, prov = apply_corrections(cfg, pk, mp, None, out)
            a = Path(pk).read_text().splitlines()
            b = Path(out).read_text().splitlines()
            self.assertEqual(len(a), len(b))
            self.assertEqual(summ["rows_time_changed"], 2)
            # MH07 2016-03-01T10:00 is 29.4167 days into the drift: c = 0.294167 s, pick moves earlier
            self.assertTrue(b[1].startswith("7,2016-03-01T09:59:59.825833Z,XD,MH07,P,2.9058,0.9,"))
            self.assertTrue(b[2].startswith("7,2016-03-01T10:00:02.700000Z,XD,MG07,S,5.7800,0.8,"))
            self.assertTrue(b[3].startswith(a[3] + ","))
            self.assertTrue(b[4].startswith(a[4] + ","))
            bi, rows = byte_identity(cfg, pk, out, tables.json.loads(Path(mp).read_text()))
            self.assertEqual(bi["rows_identical"], 2)
            self.assertEqual(bi["rows_changed_outside_rules"], 0)
            self.assertEqual(prov["inputs"]["picks"]["md5"], tables.md5(pk))


REVIEW = """candidate_id,station,model,t_start_utc,sd_t_start_days,t_end_utc,a,sd_a,cc_a,cc_a_sd,evidence,decision,size_source,t_end_override_utc,reviewer,decision_ref,note
A#1,A.XX,drift_reset,2016-01-01T00:00:00Z,1,2016-02-01T00:00:00Z,0.002,0.0002,0.003,0.0001,x,accept,cc,2016-02-01T05:00:00Z,someone,ref,
B#1,B.XX,offset_epoch,2016-01-01T00:00:00Z,1,2016-02-01T00:00:00Z,0.1,0.01,,,x,reject,,,someone,ref,
C#1,C.XX,step,2016-01-01T00:00:00Z,1,,0.1,0.01,,,x,,,,,,
"""


class TestReview(unittest.TestCase):
    def test_ingest_only_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            e, _ = ingest(_write(tmp, REVIEW, "r.csv"), str(Path(tmp) / "e.csv"))
            self.assertEqual(list(e.epoch_id), ["A#1"])
            self.assertAlmostEqual(e["size"].iloc[0], 0.003)
            self.assertEqual(e.t_end_utc.iloc[0], "2016-02-01T05:00:00Z")

    def test_accept_needs_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                ingest(_write(tmp, REVIEW.replace("someone,ref,\nB", ",ref,\nB"), "r.csv"), str(Path(tmp) / "e.csv"))


if __name__ == "__main__":
    unittest.main()
