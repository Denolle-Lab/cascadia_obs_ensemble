"""Short comparison of the Mc maps of several catalogs (v4_mc_maps.py outputs), as one HTML page.

Per region: number of grid cells with an Mc and the median cell Mc (mc_best, all years; the 'all' rows of
mc_nodes_by_region.csv) for each catalog; optionally the same for earlier maps of the same catalogs (--before, e.g.
the maps that kept the events flagged as not earthquakes, D-B105); difference maps on the cells both maps have.

  python mc_maps_compare.py --maps v4.1=<dir> v5.1=<dir> v6.0=<dir> --before v4.1=<dir> v5.1=<dir> v6.0=<dir> \
      --diff v6.0-v4.1 v6.0-v5.1 v4.1-before --out <dir>/compare.html

A difference 'a-b' is map a minus map b (labels of --maps); 'a-before' is map a minus its --before map.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_mc_maps as mm  # noqa: E402

ROWS = mm.REG_ORDER + ["ALL"]


def kv(items):
    return dict(s.split("=", 1) for s in items)


def regional(d) -> pd.DataFrame:
    """cells and median cell Mc_best per region (all years), plus ALL."""
    s = pd.read_csv(Path(d) / "mc_nodes_by_region.csv")
    s = s[s.period == "all"].set_index("region")
    g = pd.read_csv(Path(d) / "mc_grid.csv")
    out = pd.DataFrame(index=ROWS, columns=["cells", "median"], dtype=float)
    for reg in mm.REG_ORDER:
        if reg in s.index and np.isfinite(s.loc[reg, "count"]):
            out.loc[reg] = [s.loc[reg, "count"], s.loc[reg, "50%"]]
        else:
            out.loc[reg] = [0, np.nan]
    out.loc["ALL"] = [g.mc_best.notna().sum(), g.mc_best.median()]
    return out


def pooled(d) -> pd.Series:
    """pooled regional Mc_best (mc_regional.csv, variant all, period all)."""
    r = pd.read_csv(Path(d) / "mc_regional.csv")
    return r[(r.variant == "all") & (r.period == "all")].set_index("region").mc_best


def diff_grid(da, db) -> pd.DataFrame:
    a, b = pd.read_csv(Path(da) / "mc_grid.csv"), pd.read_csv(Path(db) / "mc_grid.csv")
    assert np.array_equal(a.lon.values, b.lon.values) and np.array_equal(a.lat.values, b.lat.values), "grids differ"
    return a[["lon", "lat"]].assign(d=a.mc_best.values - b.mc_best.values)


def diff_stats(g: pd.DataFrame) -> dict:
    d = g.d.dropna()
    return dict(common=len(d), median=float(d.median()), mean=float(d.mean()), up=int((d >= 0.15).sum()),
                down=int((d <= -0.15).sum()), p05=float(d.quantile(0.05)), p95=float(d.quantile(0.95)))


def diff_figure(grids: list, png: Path):
    import pygmt
    import map_context as mc
    mc.PB2002 = str(mm.ROOT / "data/tectonics/PB2002_steps.json")
    pygmt.config(FONT_ANNOT_PRIMARY="6p", FONT_LABEL="7p", MAP_FRAME_TYPE="plain",
                 MAP_TICK_LENGTH_PRIMARY="0.06c", MAP_FRAME_PEN="0.5p", FORMAT_GEO_MAP="ddd")
    cpt = png.with_name("compare_diff.cpt")
    pygmt.makecpt(cmap="vik", series=[-0.6, 0.6, 0.1], continuous=True, output=str(cpt))
    fig = pygmt.Figure()
    sta = mm._stations()
    for i, (title, g) in enumerate(grids):
        if i:
            fig.shift_origin(xshift="5.5c")
        mm._map_panel(fig, mm._grid_da(g, "d"), str(cpt), f"{chr(97 + i)}) {title}", sta, frame_left=(i == 0))
    fig.colorbar(cmap=str(cpt), position=f"JBC+w6c/0.22c+h+o{-2.75 * (len(grids) - 1)}c/0.7c",
                 frame=["xa0.2f0.1+ldelta Mc (best, all years)"])
    fig.savefig(png, dpi=200)
    cpt.unlink()


def f(x, nd=1):
    return "" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--maps", nargs="+", required=True, help="label=folder of v4_mc_maps.py outputs, in table order")
    ap.add_argument("--before", nargs="*", default=[], help="label=folder of earlier maps of the same catalog")
    ap.add_argument("--before-name", default="before", help="what the earlier maps are, for the page")
    ap.add_argument("--diff", nargs="+", default=[], help="a-b (labels of --maps) or a-before")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", action="append", default=[], help="a line for the notes list (repeatable)")
    a = ap.parse_args()
    dirs, before = kv(a.maps), kv(a.before)
    labs = list(dirs)
    R = {l: regional(d) for l, d in dirs.items()}
    RB = {l: regional(d) for l, d in before.items()}
    P = {l: pooled(d) for l, d in dirs.items()}
    PB = {l: pooled(d) for l, d in before.items()}
    params = {l: json.loads((Path(d) / "run_params.json").read_text()) for l, d in dirs.items()}

    grids = []
    for s in a.diff:
        x, y = s.split("-", 1)                           # labels carry no '-': v6.0-v4.1, v4.1-before
        if y == "before":
            grids.append((f"{x}: now - {a.before_name}", f"Mc {x} now &minus; {a.before_name}",
                          diff_grid(dirs[x], before[x])))
        else:
            grids.append((f"Mc {x} - Mc {y}", f"Mc {x} &minus; Mc {y}", diff_grid(dirs[x], dirs[y])))
    png = a.out.with_name("compare_diff.png")
    if grids:
        diff_figure([(t, g) for t, _, g in grids], png)
    img = base64.b64encode(png.read_bytes()).decode() if grids else ""

    # table 1: now
    head = "".join(f"<th>{html.escape(l)}<br>cells</th>" for l in labs) + \
        "".join(f"<th>{html.escape(l)}<br>median Mc</th>" for l in labs) + \
        "".join(f"<th>{html.escape(l)}<br>pooled Mc</th>" for l in labs)
    body = []
    for reg in ROWS:
        cls = ' class="tot"' if reg == "ALL" else ""
        body.append(f"<tr{cls}><td>{reg}</td><td class=l>{html.escape(mm.REGION_NAMES.get(reg, 'All cells'))}</td>"
                    + "".join(f"<td>{int(R[l].loc[reg, 'cells']):,}</td>" for l in labs)
                    + "".join(f"<td>{f(R[l].loc[reg, 'median'])}</td>" for l in labs)
                    + "".join(f"<td>{f(P[l].get(reg, np.nan))}</td>" for l in labs) + "</tr>")
    # table 2: before -> now
    t2 = ""
    if before:
        bl = [l for l in labs if l in before]
        h2 = "".join(f"<th>{html.escape(l)}<br>cells</th><th>{html.escape(l)}<br>median Mc</th>"
                     f"<th>{html.escape(l)}<br>pooled Mc</th>" for l in bl)
        b2 = []
        for reg in ROWS:
            cls = ' class="tot"' if reg == "ALL" else ""
            cells = []
            for l in bl:
                for (o, n, ints) in ((RB[l].loc[reg, "cells"], R[l].loc[reg, "cells"], True),
                                     (RB[l].loc[reg, "median"], R[l].loc[reg, "median"], False),
                                     (PB[l].get(reg, np.nan), P[l].get(reg, np.nan), False)):
                    fo = f"{int(o):,}" if ints else f(o)
                    fn = f"{int(n):,}" if ints else f(n)
                    cells.append(f"<td>{fn}</td>" if fo == fn else f"<td class=ch>{fo} &rarr; {fn}</td>")
            b2.append(f"<tr{cls}><td>{reg}</td>" + "".join(cells) + "</tr>")
        t2 = (f"<h2>{html.escape(a.before_name.capitalize())} &rarr; now</h2><div class='wrap'><table><tr>"
              f"<th class=l>Region</th>{h2}</tr>{''.join(b2)}</table></div><p class='sub'>A single value: unchanged.</p>")

    drows = []
    for _, lab, g in grids:
        s = diff_stats(g)
        drows.append(f"<tr><td class=l>{lab}</td><td>{s['common']:,}</td><td>{s['median']:+.2f}</td>"
                     f"<td>{s['mean']:+.3f}</td><td>{s['p05']:+.2f} / {s['p95']:+.2f}</td><td>{s['down']:,}</td>"
                     f"<td>{s['up']:,}</td></tr>")

    sel = []
    for lab, p in params.items():
        s, q = p.get("selection", {}), p.get("selection_qc", {})
        bits = [f"{s['used']:,} events used" if "used" in s else "every event with a magnitude"]
        if "no_hypocentre" in s:
            bits.append(f"no hypocentre {s['no_hypocentre']:,}")
        if "located_no_magnitude" in s:
            bits.append(f"located without a magnitude {s['located_no_magnitude']:,}")
        if "flagged_with_magnitude" in s:
            bits.append(f"flagged as not earthquakes {'and dropped' if s.get('exclude_flagged') else 'but kept'} "
                        f"{s['flagged_with_magnitude']:,}")
        if p.get("flags_from"):
            bits.append("flags from " + ", ".join(Path(x).name for x in p["flags_from"]))
        if s.get("magnitude_source"):
            bits.append("magnitude_source " + ", ".join(f"{k} {v:,}" for k, v in s["magnitude_source"].items()))
        if "used" in q:
            bits.append(f"QC variant (4 P + 4 S picks) {q['used']:,} events")
        where = (f"catalog {Path(p['catalog']).name} (md5 {p['catalog_md5']}), its own lat, lon and M" if "catalog" in p
                 else "catalog v4 epicentres (data/catalog_v4/Cascadia_relocated_catalog_v4.csv) with the converged v4.1 "
                      f"magnitudes ({Path(str(p.get('magnitudes', ''))).name})")
        sel.append(f"<li><b>{html.escape(lab)}</b>: {html.escape(where)}; " + html.escape("; ".join(bits)) + ".</li>")
    p0 = params[labs[0]]
    par = (f"the {p0['n_events']} nearest events per cell (periods {p0['n_events_period']}) within {p0['rmax_km']:g} km, "
           f"{p0['nboot']} bootstrap resamples, {p0['step_deg']} deg grid; Mc best = GFT95, else GFT90, else MAXC+0.2")
    commit = subprocess.run(["git", "-C", str(mm.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    stamp = subprocess.run(["date", "+%Y-%m-%d %H:%M %Z"], capture_output=True, text=True).stdout.strip()
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in a.note)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mc maps by catalog</title>
<style>
:root {{ --fg:#1d1d1f; --mute:#5f6368; --line:#d9d9de; --bg:#ffffff; --band:#f4f4f6; --ch:#fff3d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e8e8ea; --mute:#a0a0a8; --line:#3a3a40; --bg:#17171a; --band:#222226; --ch:#3a3020; }} }}
body {{ font: 14px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--fg); background: var(--bg);
       max-width: 1040px; margin: 24px auto; padding: 0 16px; }}
h1 {{ font-size: 20px; margin: 0 0 4px; }} h2 {{ font-size: 16px; margin: 28px 0 8px; }}
.sub {{ color: var(--mute); margin: 4px 0 16px; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; width: 100%; }}
th, td {{ padding: 4px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
th {{ font-weight: 600; font-size: 12px; color: var(--mute); vertical-align: bottom; }}
td.l, th.l {{ text-align: left; }} tr.tot td {{ font-weight: 600; background: var(--band); }} td.ch {{ background: var(--ch); }}
.wrap {{ overflow-x: auto; }} img {{ max-width: 100%; height: auto; background: #fff; }}
li {{ margin: 4px 0; }}
</style></head><body>
<h1>Mc maps by catalog</h1>
<p class="sub">Magnitude of completeness from v4_mc_maps.py with the same parameters for every catalog: {par}.
Each catalog on its own hypocentres and its own magnitude column (D-B102, D-B104). Built {stamp}, code {commit}.</p>
<h2>Cells and Mc by region (all years)</h2>
<div class="wrap"><table><tr><th class=l>Region</th><th class=l></th>{head}</tr>{''.join(body)}</table></div>
<p class="sub">Cells: 0.1&deg; grid nodes with an Mc. Median: of the cell Mc (best) in the region. Pooled: Mc best of the
region's pooled frequency-magnitude distribution (mc_regional.csv; ALL = whole catalog). Regions are the boxes of
v4_clusters_zbz.py, most specific first; ON = everything else. Cell values sit on 0.1 magnitude bins, so medians move in
0.1 steps.</p>
{t2}
<h2>Difference maps</h2>
<img src="data:image/png;base64,{img}" alt="Mc difference maps">
<div class="wrap"><table><tr><th class=l>Difference</th><th>common cells</th><th>median</th><th>mean</th>
<th>5th / 95th pct</th><th>cells &le; &minus;0.15</th><th>cells &ge; +0.15</th></tr>{''.join(drows)}</table></div>
<h2>Events</h2><ul>{''.join(sel)}</ul>
{'<h2>Notes</h2><ul>' + notes + '</ul>' if notes else ''}
</body></html>
"""
    a.out.write_text(page)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
