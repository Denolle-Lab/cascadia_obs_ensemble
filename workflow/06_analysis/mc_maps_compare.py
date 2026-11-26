"""Short comparison of the Mc maps of several catalogs (v4_mc_maps.py outputs), as one HTML page.

Per region: number of grid cells with an Mc and the median cell Mc (mc_best, all years; the 'all' rows of
mc_nodes_by_region.csv), for each catalog; difference maps on the cells both catalogs have; optionally the
change when event_flag rows (blasts, explosions) are left out (a --stage compute run with --exclude-flagged).

  python mc_maps_compare.py --maps v4.1=<dir> v5.1=<dir> v6.0=<dir> --diff v6.0-v4.1 v6.0-v5.1 \
      [--noflag v6.0=<dir>] --out <dir>/compare.html
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


def kv(items):
    return dict(s.split("=", 1) for s in items)


def regional(dirs: dict) -> pd.DataFrame:
    rows = []
    for lab, d in dirs.items():
        s = pd.read_csv(Path(d) / "mc_nodes_by_region.csv")
        s = s[s.period == "all"].set_index("region")
        g = pd.read_csv(Path(d) / "mc_grid.csv")
        for reg in mm.REG_ORDER:
            r = s.loc[reg] if reg in s.index else None
            rows.append(dict(catalog=lab, region=reg, cells=int(r["count"]) if r is not None and np.isfinite(r["count"]) else 0,
                             median=float(r["50%"]) if r is not None else np.nan))
        rows.append(dict(catalog=lab, region="ALL", cells=int(g.mc_best.notna().sum()), median=float(g.mc_best.median())))
    return pd.DataFrame(rows)


def diff_grid(da: str, db: str) -> pd.DataFrame:
    a, b = pd.read_csv(Path(da) / "mc_grid.csv"), pd.read_csv(Path(db) / "mc_grid.csv")
    assert np.allclose(a.lon, b.lon) and np.allclose(a.lat, b.lat), "grids differ"
    return a[["lon", "lat"]].assign(d=a.mc_best.values - b.mc_best.values)


def diff_stats(g: pd.DataFrame) -> dict:
    d = g.d.dropna()
    return dict(common=len(d), median=float(d.median()), mean=float(d.mean()), up=int((d >= 0.15).sum()),
                down=int((d <= -0.15).sum()), p05=float(d.quantile(0.05)), p95=float(d.quantile(0.95)))


def diff_figure(dirs: dict, pairs: list, png: Path):
    import pygmt
    import map_context as mc
    mc.PB2002 = str(mm.ROOT / "data/tectonics/PB2002_steps.json")
    pygmt.config(FONT_ANNOT_PRIMARY="6p", FONT_LABEL="7p", MAP_FRAME_TYPE="plain",
                 MAP_TICK_LENGTH_PRIMARY="0.06c", MAP_FRAME_PEN="0.5p", FORMAT_GEO_MAP="ddd")
    cpt = png.with_name("compare_diff.cpt")
    pygmt.makecpt(cmap="vik", series=[-0.6, 0.6, 0.1], continuous=True, output=str(cpt))
    fig = pygmt.Figure()
    sta = mm._stations()
    for i, (a, b) in enumerate(pairs):
        if i:
            fig.shift_origin(xshift="5.5c")
        g = diff_grid(dirs[a], dirs[b])
        mm._map_panel(fig, mm._grid_da(g, "d"), str(cpt), f"{chr(97 + i)}) Mc {a} - Mc {b}", sta, frame_left=(i == 0))
    fig.colorbar(cmap=str(cpt), position=f"JBC+w6c/0.22c+h+o{-2.75 * (len(pairs) - 1)}c/0.7c",
                 frame=["xa0.2f0.1+ldelta Mc (best, all years)"])
    fig.savefig(png, dpi=200)
    cpt.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--maps", nargs="+", required=True, help="label=folder of v4_mc_maps.py outputs, in table order")
    ap.add_argument("--diff", nargs="+", default=[], help="maps to difference, a-b (labels of --maps)")
    ap.add_argument("--noflag", nargs="*", default=[], help="label=folder of a --exclude-flagged run of that catalog")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", action="append", default=[], help="a line for the notes list (repeatable)")
    a = ap.parse_args()
    dirs = kv(a.maps)
    pairs = [tuple(s.split("-", 1)) for s in a.diff]       # labels carry no '-': v6.0-v4.1
    t = regional(dirs)
    labs = list(dirs)
    piv = t.pivot(index="region", columns="catalog", values=["cells", "median"]).reindex(mm.REG_ORDER + ["ALL"])
    params = {lab: json.loads((Path(d) / "run_params.json").read_text()) for lab, d in dirs.items()}

    png = a.out.with_name("compare_diff.png")
    if pairs:
        diff_figure(dirs, pairs, png)
    img = base64.b64encode(png.read_bytes()).decode() if pairs else ""

    def f(x, nd=2):
        return "" if not np.isfinite(x) else f"{x:.{nd}f}"

    head = "".join(f"<th>{html.escape(l)}<br>cells</th>" for l in labs) + \
        "".join(f"<th>{html.escape(l)}<br>median Mc</th>" for l in labs) + \
        (f"<th>{html.escape(labs[-1])} &minus; {html.escape(labs[0])}<br>median Mc</th>" if len(labs) > 1 else "")
    body = []
    for reg in mm.REG_ORDER + ["ALL"]:
        name = mm.REGION_NAMES.get(reg, "All cells")
        c = [int(piv.loc[reg, ("cells", l)]) for l in labs]
        m = [float(piv.loc[reg, ("median", l)]) for l in labs]
        dm = m[-1] - m[0] if len(m) > 1 else np.nan
        cls = ' class="tot"' if reg == "ALL" else ""
        body.append(f"<tr{cls}><td>{reg}</td><td class=l>{html.escape(name)}</td>" + "".join(f"<td>{x:,}</td>" for x in c)
                    + "".join(f"<td>{f(x)}</td>" for x in m) + (f"<td>{f(dm)}</td>" if len(m) > 1 else "") + "</tr>")

    drows = []
    for (x, y) in pairs:
        s = diff_stats(diff_grid(dirs[x], dirs[y]))
        drows.append(f"<tr><td class=l>{x} &minus; {y}</td><td>{s['common']:,}</td><td>{s['median']:+.2f}</td>"
                     f"<td>{s['mean']:+.3f}</td><td>{s['p05']:+.2f} / {s['p95']:+.2f}</td><td>{s['down']:,}</td>"
                     f"<td>{s['up']:,}</td></tr>")
    nrows = []
    for lab, d in kv(a.noflag).items():
        g = pd.read_csv(Path(d) / "mc_grid.csv")
        base = pd.read_csv(Path(dirs[lab]) / "mc_grid.csv")
        dd = (g.mc_best - base.mc_best).dropna()
        p = json.loads((Path(d) / "run_params.json").read_text())
        nrows.append(f"<tr><td class=l>{lab}</td><td>{p['selection']['flagged_with_magnitude']:,}</td>"
                     f"<td>{int(base.mc_best.notna().sum()):,} &rarr; {int(g.mc_best.notna().sum()):,}</td>"
                     f"<td>{(dd.abs() >= 0.05).sum():,}</td><td>{(dd.abs() >= 0.15).sum():,}</td>"
                     f"<td>{dd.min():+.1f} / {dd.max():+.1f}</td></tr>")

    sel = []
    for lab, p in params.items():
        s = p.get("selection")
        if s:
            ms = ", ".join(f"{k} {v:,}" for k, v in s.get("magnitude_source", {}).items())
            sel.append(f"<li><b>{lab}</b>: {s['used']:,} events used of {s['rows']:,} rows; no hypocentre "
                       f"{s['no_hypocentre']:,}; located without a magnitude {s['located_no_magnitude']:,}; "
                       f"event_flag rows with a magnitude kept {s['flagged_with_magnitude']:,}; magnitude_source: {ms}. "
                       f"QC variant (4 P + 4 S picks): {p['selection_qc']['used']:,} events. "
                       f"Catalog md5 {p['catalog_md5']}.</li>")
        else:
            sel.append(f"<li><b>{lab}</b>: catalog v4 epicentres (data/catalog_v4/Cascadia_relocated_catalog_v4.csv) "
                       f"with the converged v4.1 magnitudes ({html.escape(str(p.get('magnitudes', mm.mag_paths.v41('converged'))))}); "
                       f"every event with a magnitude.</li>")
    p0 = params[labs[0]]
    par = (f"n {p0['n_events']} events per cell (periods {p0['n_events_period']}), radius at most {p0['rmax_km']:g} km, "
           f"{p0['nboot']} bootstrap resamples, {p0['step_deg']} deg grid, Mc best = GFT95, else GFT90, else MAXC+0.2")
    commit = subprocess.run(["git", "-C", str(mm.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    stamp = subprocess.run(["date", "+%Y-%m-%d %H:%M %Z"], capture_output=True, text=True).stdout.strip()
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in a.note)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mc maps by catalog</title>
<style>
:root {{ --fg:#1d1d1f; --mute:#5f6368; --line:#d9d9de; --bg:#ffffff; --band:#f4f4f6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e8e8ea; --mute:#a0a0a8; --line:#3a3a40; --bg:#17171a; --band:#222226; }} }}
body {{ font: 14px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--fg); background: var(--bg);
       max-width: 1000px; margin: 24px auto; padding: 0 16px; }}
h1 {{ font-size: 20px; margin: 0 0 4px; }} h2 {{ font-size: 16px; margin: 28px 0 8px; }}
.sub {{ color: var(--mute); margin: 0 0 16px; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; width: 100%; }}
th, td {{ padding: 4px 8px; border-bottom: 1px solid var(--line); text-align: right; }}
th {{ font-weight: 600; font-size: 12px; color: var(--mute); vertical-align: bottom; }}
td.l, th.l {{ text-align: left; }} tr.tot td {{ font-weight: 600; background: var(--band); }}
.wrap {{ overflow-x: auto; }} img {{ max-width: 100%; height: auto; background: #fff; }}
li {{ margin: 4px 0; }} code {{ font-size: 12px; }}
</style></head><body>
<h1>Mc maps by catalog</h1>
<p class="sub">Magnitude of completeness (v4_mc_maps.py, same parameters for every catalog: {par}).
Each catalog on its own hypocentres and its own magnitude column (D-B102, D-B104). Built {stamp}, code {commit}.</p>
<h2>Cells and median Mc by region (all years)</h2>
<div class="wrap"><table><tr><th class=l>Region</th><th class=l></th>{head}</tr>{''.join(body)}</table></div>
<p class="sub">Cells: 0.1&deg; grid nodes with an Mc (the 250 nearest events within 50 km). Median: of the cell Mc (best) in
the region; regions are the boxes of v4_clusters_zbz.py (most specific first), ON = everything else.</p>
<h2>Difference maps</h2>
<img src="data:image/png;base64,{img}" alt="Mc difference maps">
<div class="wrap"><table><tr><th class=l>Difference</th><th>common cells</th><th>median</th><th>mean</th>
<th>5th / 95th pct</th><th>cells &le; &minus;0.15</th><th>cells &ge; +0.15</th></tr>{''.join(drows)}</table></div>
{'<h2>Blasts and explosions left out</h2><div class="wrap"><table><tr><th class=l>Catalog</th><th>event_flag rows with M</th><th>cells</th><th>cells |&Delta;Mc| &ge; 0.05</th><th>cells |&Delta;Mc| &ge; 0.15</th><th>min / max &Delta;Mc</th></tr>' + ''.join(nrows) + '</table></div><p class="sub">The maps above keep event_flag rows, as the v4.1 maps do; this run (--exclude-flagged, compute only) shows what leaving them out would change.</p>' if nrows else ''}
<h2>Events</h2><ul>{''.join(sel)}</ul>
{'<h2>Notes</h2><ul>' + notes + '</ul>' if notes else ''}
</body></html>
"""
    a.out.write_text(page)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
