"""Synthetic stations: injected drift, offset and reset are recovered; clean stations are not flagged.

Run:  python -m unittest discover -s tools/pick_timing/tests   (or pytest tools/pick_timing/tests)
"""
import unittest

import numpy as np

from pick_timing import changepoint as cp
from pick_timing import detect, residuals, synth
from pick_timing.config import DEFAULTS

P = dict(DEFAULTS["detect"], n_perm=100, n_boot=50)


def _station(seed, days=360, n=2500, sigma=0.12, clock=None):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, days, n))
    y = sigma / 1.134 * rng.standard_t(3, n)
    if clock:
        y = y + synth.clock_term(t_days=t, **clock)
    return t, y


class TestRecovery(unittest.TestCase):
    """Tolerances: start/end within 3 days, size within 10 % (drift) or 0.03 s (offset)."""

    def test_drift_reset(self):
        clock = dict(model="drift_reset", t1=120.0, t2=210.0, a=0.006)
        t, y = _station(1, clock=clock)
        sc, ep = detect.detect_series(t, y, P, seed=1, max_epochs=1)
        self.assertTrue(sc["candidate"])
        f = ep[0]["fit"]
        self.assertEqual(ep[0]["model"], "drift_reset")
        self.assertLess(abs(f["c1"] - 120.0), 3.0)
        self.assertTrue(f["c2_lo"] - 3 <= 210.0 <= f["c2_hi"] + 3)
        self.assertLess(abs(f["a"] / 0.006 - 1), 0.10)

    def test_offset_epoch(self):
        clock = dict(model="offset_epoch", t1=200.0, t2=240.0, a=-0.25)
        t, y = _station(2, clock=clock)
        sc, ep = detect.detect_series(t, y, P, seed=2, max_epochs=1)
        f = ep[0]["fit"]
        self.assertEqual(ep[0]["model"], "offset_epoch")
        self.assertLess(abs(f["a"] + 0.25), 0.03)
        self.assertLess(abs(f["c1"] - 200.0), 3.0)
        self.assertLess(abs(f["c2"] - 240.0), 3.0)

    def test_step(self):
        clock = dict(model="step", t1=150.0, a=0.15)
        t, y = _station(3, clock=clock)
        sc, ep = detect.detect_series(t, y, P, seed=3, max_epochs=1)
        f = ep[0]["fit"]
        self.assertIn(ep[0]["model"], ("step", "offset_epoch"))
        self.assertLess(abs(f["a"] - 0.15), 0.03)
        self.assertLess(abs(f["c1"] - 150.0), 3.0)

    def test_reset_snaps_to_header_restart(self):
        """No picks between the last drifted pick and the restart: the reset is put on the restart."""
        clock = dict(model="drift_reset", t1=100.0, t2=200.0, a=0.005)
        t, y = _station(4, clock=clock)
        keep = (t < 196.0) | (t > 204.0)          # data gap around the restart
        sc, ep = detect.detect_series(t[keep], y[keep], P, restarts=np.array([200.0]), seed=4, max_epochs=1)
        f = ep[0]["fit"]
        r = detect.restart_check(np.array([200.0]), f, 1.0)
        self.assertAlmostEqual(f["c2"], 200.0, places=6)
        self.assertTrue(r["reset_at_restart"])


class TestNoFalseAlarm(unittest.TestCase):
    def test_clean_stations(self):
        """40 clean synthetic stations (heavy-tailed noise plus bin-level scatter): none flagged at the default
        thresholds (dbic_min 30, amp_min 0.05 s, p <= 0.01)."""
        flagged = 0
        for k in range(40):
            rng = np.random.default_rng(100 + k)
            n = int(rng.integers(300, 3000))
            t, y = _station(100 + k, days=float(rng.uniform(120, 900)), n=n, sigma=float(rng.uniform(0.05, 0.25)))
            ids = cp.bin_assign(t, P["bin_picks"], P["bin_max_days"], P["bin_min_picks"])
            y = y + rng.normal(0, 0.02, ids[-1] + 1)[ids]
            sc, ep = detect.detect_series(t, y, P, seed=k, max_epochs=1)
            flagged += bool(ep)
        self.assertEqual(flagged, 0)


class TestNetwork(unittest.TestCase):
    def test_network_relative_and_evidence(self):
        """Full chain on a synthetic network: only the two stations with clocks are candidates, the drift is
        classified clock-like (co-located pairs and P = S)."""
        clocks = {"S03.XX": dict(model="drift_reset", t1=150.0, t2=240.0, a=0.006),
                  "S07.XX": dict(model="offset_epoch", t1=300.0, t2=340.0, a=-0.25)}
        res, st, ori = synth.network(seed=5, n_ev=1200, clocks=clocks)
        p = dict(P, n_perm=120, n_boot=30)
        rd = residuals.network_relative(res, p)
        screen, cands, fa, frames, series = detect.run_detect(rd, p, workers=1, false_alarm=False)
        self.assertEqual(set(cands.station), set(clocks))
        d = cands.set_index("station")
        self.assertTrue(d.loc["S03.XX", "evidence"].startswith("clock-like"))
        self.assertLess(abs(d.loc["S03.XX", "t_start_days"] - (synth.T0 / 86400 + 150)), 3)
        self.assertLess(abs(d.loc["S07.XX", "a"] + 0.25), 0.04)


if __name__ == "__main__":
    unittest.main()
