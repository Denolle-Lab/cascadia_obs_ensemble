#!/usr/bin/env python3
"""Segments before and after an axes change (PB_AXES_OLD, default gmrt = D-B113 axes v1, against the current PB_AXES,
default v2 = adopted boundaries v2, D-B117): events moved per segment and a before/after map, on the catalog of the
PB_* variables (v6.1 for the paper). The D-B113 comparison was PB_AXES_OLD=pb2002 PB_AXES=gmrt. Event set: located earthquakes, tiers
A-C, review P1 left out (as the metrics).

Writes <PB_OUT>/axes_moved_events.csv (old x new segment counts), <PB_OUT>/axes_moved_by_segment.csv (per segment:
before, after, out, in) and <PB_BIG>/figures/fig_axes_before_after.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pb_common as P  # noqa: E402
import os  # noqa: E402

OLD_AX = os.environ.get("PB_AXES_OLD", "gmrt")
LAB = {"pb2002": "PB2002", "gmrt": "D-B113 axes v1", "v2": "boundaries v2 (D-B117)", "v2proposal": "v2 proposal"}
from pb_figures import TYPE_COL  # noqa: E402


def main():
    c = P.load_catalog()
    old = P.assign_segments(c, axes=OLD_AX)
    new = P.assign_segments(c, axes=P.AXES)
    sel = (c.is_eq & c.abc & ~c.review_P1).values
    o, n = old.seg.values[sel], new.seg.values[sel]
    ct = pd.crosstab(pd.Series(o, name="before"), pd.Series(n, name="after"))
    ct.to_csv(P.OUT / "axes_moved_events.csv")
    rows = []
    for s in P.SEG_ORDER:
        rows.append(dict(seg=s, before=int((o == s).sum()), after=int((n == s).sum()),
                         moved_out=int(((o == s) & (n != s)).sum()), moved_in=int(((n == s) & (o != s)).sum()),
                         out_to=";".join(f"{k}:{v}" for k, v in pd.Series(n[(o == s) & (n != s)]).value_counts().items()),
                         in_from=";".join(f"{k}:{v}" for k, v in pd.Series(o[(n == s) & (o != s)]).value_counts().items())))
    t = pd.DataFrame(rows)
    t.to_csv(P.OUT / "axes_moved_by_segment.csv", index=False)
    P.log(t[(t.moved_out > 0) | (t.moved_in > 0)].to_string(index=False))
    P.log(f"moved: {int((o != n).sum())} of {int(sel.sum())} tier A-C earthquakes")
    # map
    e = c[sel].assign(old=o, new=n)
    mv = e[e.old != e.new]
    pts_o, _, _ = P.pb2002(OLD_AX)
    pts_n, _, _ = P.pb2002(P.AXES)
    fig, axs = plt.subplots(1, 3, figsize=(14, 6.4), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, (lab, col, pts) in zip(axs[:2], ((f"before ({LAB[OLD_AX]})", "old", pts_o),
                                             (f"after ({LAB[P.AXES]})", "new", pts_n))):
        g = e[(e.lon > -128.0) & (e.lon < -124.6) & (e.lat > 40.2) & (e.lat < 43.1)]
        for s_, h in g.groupby(col):
            ax.scatter(h.lon, h.lat, s=0.6, color=TYPE_COL[P.SEGMENTS[s_][1]], lw=0, rasterized=True)
        for k in ("OSR", "OTF"):
            ax.plot(pts[k][:, 0], pts[k][:, 1], ".", ms=0.6, color="k")
        ax.set_xlim(-128.0, -124.6)
        ax.set_ylim(40.2, 43.1)
        ax.set_aspect(1 / np.cos(np.radians(41.6)))
        ax.set_title(f"Gorda: {lab}", fontsize=9)
    for t_ in P.TYPE_ORDER:
        axs[0].plot([], [], "o", color=TYPE_COL[t_], label=t_, ms=4)
    axs[0].legend(fontsize=6.5, loc="lower right")
    ax = axs[2]
    ax.scatter(e.lon, e.lat, s=0.3, color="0.8", lw=0, rasterized=True)
    for (a, b), h in mv.groupby(["old", "new"]):
        ax.scatter(h.lon, h.lat, s=2, lw=0, label=f"{a} -> {b} ({len(h)})")
    for k in ("OSR", "OTF"):
        ax.plot(pts_o[k][:, 0], pts_o[k][:, 1], ".", ms=0.5, color="c")
        ax.plot(pts_n[k][:, 0], pts_n[k][:, 1], ".", ms=0.5, color="k")
    ax.set_xlim(-131.0, -124.6)
    ax.set_ylim(40.2, 48.6)
    ax.set_aspect(1 / np.cos(np.radians(44.4)))
    ax.legend(fontsize=6.5, loc="lower left", markerscale=3)
    ax.set_title(f"events that change segment (cyan {LAB[OLD_AX]}, black {LAB[P.AXES]})", fontsize=9)
    fig.suptitle(f"DRAFT: segment rule before and after the axes change, catalog {P.CATLABEL}, tier A-C "
                 f"earthquakes; {len(mv):,} events change segment", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(P.BIG / "figures" / "fig_axes_before_after.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
