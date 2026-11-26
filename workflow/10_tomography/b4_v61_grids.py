"""Track B, catalog v6.1 depth test (D-B100): small helpers shared by the diagnosis and the test.

* NonLinLoc grid files (.hdr/.buf, TIME or SSST_TIMECORR, float32, x fastest-varying last: index (ix, iy, iz) with
  z the fastest, as Grid2Time writes them) read into memory with their header;
* trilinear interpolation of a grid at points (km, the grid's frame) and its spatial gradient (central differences);
* a parser of concatenated NLLoc .hyp files (the v6.0 pass files it<k>/hyp/<tile>.hyp, or one per-event file) that
  returns events (ID, status, x, y, z in the grid frame, lat, lon, depth, OT, quality) and phase rows (station, phase,
  observed time, predicted time, residual, weight, distance).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd


class Grid:
    """A NonLinLoc 3D grid (header line 1: nx ny nz x0 y0 z0 dx dy dz TYPE FLOAT; line 2 for TIME grids: station)."""

    def __init__(self, hdr):
        hdr = Path(hdr)
        h = hdr.read_text().splitlines()
        t = h[0].split()
        self.n = tuple(int(v) for v in t[:3])
        self.o = np.array([float(v) for v in t[3:6]])
        self.d = np.array([float(v) for v in t[6:9]])
        self.kind = t[9]
        self.sta = None
        if len(h) > 1 and h[1].split() and h[1].split()[0] not in ("TRANSFORM",):
            s = h[1].split()
            self.sta = (s[0], float(s[1]), float(s[2]), float(s[3]))
        buf = hdr.with_suffix(".buf")
        self.v = np.fromfile(buf, "<f4").reshape(self.n)
        self.path = str(buf)

    def _idx(self, p):
        f = (np.atleast_2d(p) - self.o) / self.d
        n = np.array(self.n)
        f = np.clip(f, 0, n - 1 - 1e-9)
        i = np.floor(f).astype(int)
        i = np.minimum(i, np.maximum(n - 2, 0))
        return i, f - i

    def at(self, p):
        """trilinear value at points p (N x 3, km); points outside are clamped to the grid edge (callers check)."""
        p = np.atleast_2d(np.asarray(p, float))
        n = np.array(self.n)
        if (n == 1).any():                       # degenerate axis (a 2D grid): no interpolation along it
            p = p.copy()
            for ax in np.flatnonzero(n == 1):
                p[:, ax] = self.o[ax]
        i, w = self._idx(p)
        out = np.zeros(len(p))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ii = np.minimum(i + [dx, dy, dz], n - 1)
                    ww = (w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1]) * \
                         (w[:, 2] if dz else 1 - w[:, 2])
                    out += ww * self.v[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def inside(self, p):
        p = np.atleast_2d(p)
        hi = self.o + (np.array(self.n) - 1) * self.d
        return np.all((p >= self.o - 1e-6) & (p <= hi + 1e-6), axis=1)

    def grad(self, p, h=0.25):
        """central-difference gradient (s/km) at points p, step h km (trilinear values)."""
        p = np.atleast_2d(np.asarray(p, float))
        g = np.zeros_like(p)
        for ax in range(3):
            e = np.zeros(3)
            e[ax] = h
            g[:, ax] = (self.at(p + e) - self.at(p - e)) / (2 * h)
        return g


_HYPO = re.compile(r"^HYPOCENTER\s+x\s+(\S+)\s+y\s+(\S+)\s+z\s+(\S+)\s+OT\s+(\S+)")


def parse_hyp(path, ids=None):
    """events and phases of a (concatenated) NLLoc .hyp file. ids: keep only these event IDs (set) or all."""
    evs, phs = [], []
    cur, on = None, False
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("NLLOC "):
                q = re.findall(r'"([^"]*)"', line)
                cur = {"status": q[1]}
                on = False
                continue
            if cur is None:
                continue
            if line.startswith("PUBLIC_ID"):
                cur["ID"] = int(line.split()[1].split("_")[1])
                if ids is not None and cur["ID"] not in ids:
                    cur = None
                continue
            if line.startswith("HYPOCENTER"):
                m = _HYPO.match(line)
                cur.update(x=float(m.group(1)), y=float(m.group(2)), z=float(m.group(3)))
            elif line.startswith("GEOGRAPHIC"):
                t = line.split()
                cur.update(time=f"{t[2]}-{t[3]}-{t[4]}T{t[5]}:{t[6]}:{float(t[7]):09.6f}",
                           lat=float(t[9]), lon=float(t[11]), depth=float(t[13]))
            elif line.startswith("SEARCH OCTREE"):
                m = re.search(r"smallestNodeSide \S+/\S+/(\S+)", line)
                cur["ns_z"] = float(m.group(1))
            elif line.startswith("QUALITY"):
                t = line.split()
                d = dict(zip(t[1::2], t[2::2]))
                cur.update(rms=float(d["RMS"]), nphs=int(d["Nphs"]), gap=float(d["Gap"]), dist_min=float(d["Dist"]))
            elif line.startswith("STATISTICS"):
                t = line.split()
                d = dict(zip(t[1::2], t[2::2]))
                cur.update(sig_z=float(np.sqrt(float(d["ZZ"]))), ell_len3=float(d["Len3"]))
            elif line.startswith("QML_OriginUncertainty"):
                t = line.split()
                d = dict(zip(t[1::2], t[2::2]))
                cur.update(h_min=float(d["minHorUnc"]), h_max=float(d["maxHorUnc"]))
            elif line.startswith("QML_OriginQuality"):
                t = line.split()
                d = dict(zip(t[1::2], t[2::2]))
                cur.update(n_sta=int(d["usedStaCt"]))
            elif line.startswith("PHASE "):
                on = True
            elif line.startswith("END_PHASE"):
                on = False
            elif line.startswith("END_NLLOC"):
                evs.append(cur)
                cur = None
            elif on and ">" in line:
                a, b = line.split(">")
                t, u = a.split(), b.split()
                phs.append((cur["ID"], t[0], t[4], float(t[7]), float(t[8]), float(u[0]), float(u[1]), float(u[2]),
                            float(u[6]), float(u[7])))
    ev = pd.DataFrame(evs)
    ph = pd.DataFrame(phs, columns=["ID", "sta", "phase", "hrmn", "sec", "ttpred", "res", "weight", "sdist", "sazim"])
    return ev, ph
