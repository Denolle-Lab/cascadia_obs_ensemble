"""Text of the Q1 report (D-B115). Numbers come from the tables; the interpretation was written after the results,
against the pre-registered decision rules (data/paper_plan/q1_slip_style/RESUME.md, 13:26:14 PST)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

PRE_TIME = "Fri Nov 27 13:26:14 PST 2026"
PRE_COMMIT = "109915b8"


def _p(v):
    if v is None or not np.isfinite(v):
        return "-"
    return f"{v:.4f}" if v >= 0.0001 else f"{v:.1e}"


def render(d, label, h):
    esc, figure, fmt, table, FIG, Q = h["esc"], h["figure"], h["fmt"], h["table"], h["FIG"], h["Q"]
    T, C, N, S, V, dec = d["T"], d["C"], d["N"], d["S"], d["V"], d["dec"]
    t = T.set_index("test")
    prim = T[T.family == "primary"].copy()
    info = d["info"]["main"]
    out = []
    A = out.append

    # ------------------------------------------------------------ header and verdict
    A(f"<h1>Q1: what sets how a plate boundary releases slip?</h1>")
    A(f'<p class="muted">Swarms or mainshock-aftershock sequences, across the offshore Cascadia plate system. Catalog '
      f"v6.1, plate boundaries v2 (D-B117), D-B114 external data. Decision D-B115. Report built {esc(label)}. "
      f"Pre-registration "
      f"written {PRE_TIME} and committed as {PRE_COMMIT} (branch science/q1-slip-style) before any covariate "
      f"correlation was computed.</p>")
    A(TXT_VERDICT(d, h))
    A("<h2>Boundaries v2 (D-B117)</h2>")
    A("<p>This version uses the adopted plate boundaries v2. The first version of the report used the D-B113 axes; "
      "the table gives the key numbers of both runs, read from their tables.</p>")
    A(TXT_V2(d, h))

    # ------------------------------------------------------------ 1 question
    A("<h2>1. Question and hypothesis</h2>")
    A("<p><b>Question.</b> What sets how a plate boundary releases slip: in swarms, or in mainshock-aftershock "
      "sequences?</p>")
    A("<p><b>Working hypothesis (H1).</b> Release is swarm-dominated where the lithosphere is young, hot, magmatic or "
      "fluid-rich (ridges, the Blanco pull-aparts, Mount St Helens) and mainshock-aftershock-dominated where it is colder "
      "and stronger (Gorda plate, Mendocino fault zone, Gorda slab). In testable form: the share of earthquakes in "
      "swarms rises with thermal state (younger plate, higher heat flow, shallower isotherms) and with fluid content "
      "(higher in-situ Vp/Vs), and it does so across spatial cells drawn without the segment boundaries, not only as a "
      "contrast between boundary types.</p>")
    A("<p>What was known before the registration (as written in it), and so cannot confirm H1: the v6.1 "
      "plate-boundary table of 10:07 PST (PB2002 segments) gave swarm "
      "shares of 29 % at Axial, 28 % at Mount St Helens, 25 % at Blanco and 19 % on the Gorda Ridge, against 2.8 % or "
      "less at the Mendocino fault zone, Gorda plate, triple junction and Gorda slab. That contrast is where H1 came "
      "from. It is shown below for completeness, but it carries no test.</p>")

    # ------------------------------------------------------------ 2 pre-registration
    A("<h2>2. Pre-registered tests</h2>")
    A(f"<p>Written {PRE_TIME} in <code>data/paper_plan/q1_slip_style/RESUME.md</code> and committed ({PRE_COMMIT}) "
      "before any covariate was sampled. Event set: located earthquakes (event_flag empty, so blasts, explosions and the "
      "Mount St Helens analog test signal are out), tiers A-C, not in the waveform-review set P1, at or above the "
      "regional Mc of the v6.1 nearest-neighbour clustering. Swarm share = percentage of these events that belong to "
      "families typed swarm.</p>")
    rows = [("T1", "0.2 deg cells (>= 50 events)", "swarm share vs plate age", "rho < 0", "one-sided permutation"),
            ("T2", "0.2 deg cells", "swarm share vs heat flow", "rho > 0", "one-sided permutation"),
            ("T3", "families", "Vp/Vs, swarm minus mainshock-aftershock", "> 0",
             "labels permuted within boundary type; needs >= 20 + 20 families"),
            ("T4a", "0.2 deg cells", "within-type rho, swarm share vs age", "< 0", "covariate permuted within type"),
            ("T4b", "0.2 deg cells", "within-type rho, swarm share vs heat flow", "> 0", "covariate permuted within type")]
    A(table(pd.DataFrame(rows, columns=["test", "unit", "statistic", "prediction", "p-value"]),
            left=("unit", "statistic", "prediction", "p-value")))
    A("<p>Holm correction over these five tests (alpha 0.05). Secondary tests (b, productivity, family mix, background "
      "fraction against age and heat flow; swarm share against spreading rate; segment-level correlations) share one "
      "Benjamini-Hochberg family; roughness, sediment, the isotherm depth and the slab-surface temperature are "
      "exploratory. Every cell-level rho also has a 95 % interval from bootstraps of 1 x 1 degree blocks of cells.</p>")
    A("<p><b>Decision rules.</b> D1 (thermal control supported): T1 or T2 significant after Holm with the predicted "
      "sign, its block interval excludes 0, the same sign in at least 18 of the 22 control runs, and the time-shuffle "
      "null rule passes (at least 3 of Axial, Blanco, Gorda Ridge and Mount St Helens above the 95th percentile of "
      "their swarm share in time-shuffled catalogs). D2 (beyond boundary type): D1 plus the matching T4. D3 (fluid "
      "signal): T3 significant. Falsified: T1 and T2 both reversed with two-sided p &lt; 0.05, or the hot group not "
      "above the cold group after reclustering at a uniform Mc, or the null rule failing; for Vp/Vs, mainshock-"
      "aftershock families above swarm families with two-sided p &lt; 0.05. Anything else is inconclusive.</p>")
    A("<h3>What changed after the registration</h3><ul>")
    for li in TXT_CHANGES(d, h):
        A(f"<li>{li}</li>")
    A("</ul>")

    # ------------------------------------------------------------ 3 data
    A("<h2>3. Data</h2>")
    A(TXT_DATA(d, h))

    # ------------------------------------------------------------ 4 results
    A("<h2>4. Results</h2>")
    A(TXT_RESULTS(d, h))

    # ------------------------------------------------------------ 5 controls
    A("<h2>5. Controls</h2>")
    A(TXT_CONTROLS(d, h))

    # ------------------------------------------------------------ 6 claims
    A("<h2>6. What the paper can claim, and what it cannot</h2>")
    A(TXT_CLAIMS(d, h))

    # ------------------------------------------------------------ 7 bullets
    A("<h2>7. Manuscript bullets</h2>")
    A('<p class="muted">Bullets only (PI rule of 2026-11-23); the authors write the prose.</p><ul>')
    for li in TXT_BULLETS(d, h):
        A(f"<li>{li}</li>")
    A("</ul>")

    # ------------------------------------------------------------ 8 limitations
    A("<h2>8. Limitations</h2><ul>")
    for li in TXT_LIMITS(d, h):
        A(f"<li>{li}</li>")
    A("</ul>")
    A(TXT_FILES(d, h))
    return "\n".join(out)


# the result-dependent sections are filled in q1_report_text_results.py (imported below), written after the run
from q1_report_text_results import (TXT_CHANGES, TXT_CLAIMS, TXT_CONTROLS, TXT_DATA, TXT_FILES,  # noqa: E402,F401
                                    TXT_LIMITS, TXT_BULLETS, TXT_RESULTS, TXT_V2, TXT_VERDICT)
