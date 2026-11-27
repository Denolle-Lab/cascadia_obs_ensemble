"""HTML helpers for the Q1 report (self-contained page: figures embedded as PNG data URIs)."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd

CSS = """
:root { --fg:#1d232a; --muted:#5b6670; --bg:#ffffff; --rule:#d9dee3; --accent:#1b5e8a; --warn:#9a5b1f; --chip:#eef3f7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --fg:#e3e7eb; --muted:#9aa5af; --bg:#15191d;
  --rule:#323a42; --accent:#7fb4d9; --warn:#e0a86a; --chip:#222a31; } }
:root[data-theme="dark"] { --fg:#e3e7eb; --muted:#9aa5af; --bg:#15191d; --rule:#323a42; --accent:#7fb4d9;
  --warn:#e0a86a; --chip:#222a31; }
body { background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width:1080px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:1.55em; margin:0.2em 0 0.1em; } h2 { font-size:1.2em; margin-top:2em; border-bottom:1px solid var(--rule);
  padding-bottom:0.2em; } h3 { font-size:1.02em; margin-top:1.4em; }
p, li { max-width:78ch; } .muted { color:var(--muted); font-size:0.92em; }
table { border-collapse:collapse; font-size:0.86em; margin:0.6em 0 1em; display:block; overflow-x:auto; max-width:100%; }
th, td { border-bottom:1px solid var(--rule); padding:3px 8px; text-align:right; white-space:nowrap; }
th { background:var(--chip); position:sticky; top:0; } td:first-child, th:first-child { text-align:left; }
td.l, th.l { text-align:left; white-space:normal; }
img { max-width:100%; height:auto; border:1px solid var(--rule); background:#fff; }
.verdict { border-left:4px solid var(--accent); background:var(--chip); padding:8px 12px; margin:1em 0; }
.warn { border-left:4px solid var(--warn); background:var(--chip); padding:8px 12px; margin:1em 0; }
code { background:var(--chip); padding:0 3px; border-radius:3px; font-size:0.92em; }
figure { margin:1.2em 0; } figcaption { color:var(--muted); font-size:0.9em; margin-top:0.3em; max-width:90ch; }
"""


def esc(x) -> str:
    return html.escape(str(x))


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if v != 0 and abs(v) < 0.001:
            return f"{v:.1e}"
        return f"{v:.{nd}f}"
    return esc(v)


def table(df: pd.DataFrame, cols=None, names=None, nd=None, left=()) -> str:
    cols = cols or list(df.columns)
    names = names or cols
    nd = nd or {}
    h = "<table><thead><tr>" + "".join(
        f'<th class="{"l" if c in left else ""}">{esc(n)}</th>' for c, n in zip(cols, names)) + "</tr></thead><tbody>"
    for _, r in df.iterrows():
        h += "<tr>" + "".join(f'<td class="{"l" if c in left else ""}">{fmt(r[c], nd.get(c, 2))}</td>'
                              for c in cols) + "</tr>"
    return h + "</tbody></table>"


def figure(path: Path, caption: str) -> str:
    if not path.exists():
        return f'<p class="muted">[figure {esc(path.name)} missing]</p>'
    b = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img alt="{esc(path.stem)}" src="data:image/png;base64,{b}"><figcaption>{caption}</figcaption></figure>'


def page(title: str, body: str, description: str = "") -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>{esc(title)}</title>'
            f'<meta name="description" content="{esc(description)}"><style>{CSS}</style></head>'
            f"<body>{body}</body></html>")
