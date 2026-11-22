"""Publish the event-derived fault surfaces (data/faults/vN/) as an open Zenodo record.

Token: $ZENODO_TOKEN or ~/.zenodo_token (scopes deposit:write, deposit:actions).

    python utils/zenodo_fault_surfaces.py create data/faults/v0        # draft + upload + metadata, prints id
    python utils/zenodo_fault_surfaces.py publish ID                   # mint the DOI (cannot be undone)
    python utils/zenodo_fault_surfaces.py newversion ID data/faults/v1 # next version of a published record
Add --sandbox to use sandbox.zenodo.org.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import requests

DESCRIPTION = """<p><b>Version v0: a method test, not for interpretation.</b> Three-dimensional
surfaces fitted to clusters of earthquakes in the Cascadia catalog v4 (172,340 events,
2010&ndash;2016, ELEP ver6 picks on ocean-bottom and land stations; GENIE association and
GraphDD location by I. McBrearty), with per-vertex uncertainties.</p>
<p>74 surfaces: 23 on the Blanco transform (near-vertical segments, parallel strands and the
Cascadia Depression step-over), 20 at the Mendocino triple junction, 6 in the Gorda plate,
2 at Axial Seamount, 11 in the Puget Lowland and 12 onshore elsewhere. 43 are steep (dip
&ge; 45&deg;), and 39 of these are stable under changes of the clustering settings.</p>
<p>v0 uses the v4 locations as they are. Absolute depths are biased shallow, the depth datum
is not confirmed and v4 has no relative relocation, so v0 tests the fitting and uncertainty
steps. Version v1 will use the catalog relocated in a 3D velocity model; v2 will add focal
mechanisms computed on our own locations. Published (ComCat) mechanisms are not used.</p>
<p>Method: DBSCAN clustering; recursive splitting into planar pieces (two planes or a cut
across strike); a plane or quadratic surface by BIC on a 1 km grid. Per-vertex 1-sigma
uncertainties normal to the surface come from three sources, reported separately and
combined: a bootstrap of the events, a Monte Carlo over location errors (worst case), and
the spread over surface-model choices. There are also bootstrap strike and dip intervals,
a clustering-stability test over a grid of settings, the zone width compared with the
location error, and the distances to the CRESCENT Community Fault Model and to Slab2.</p>
<p>Files:</p>
<ul>
<li>GeoJSON meshes (lon, lat, elevation in m), in the layout of the CRESCENT CFM;</li>
<li>a summary table;</li>
<li>per-vertex uncertainties;</li>
<li>the full ensemble (100 draws);</li>
<li>event membership;</li>
<li>a map;</li>
<li>the code.</li>
</ul>
<p>See README.md for every field.</p>"""


def base(sandbox):
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def session():
    t = os.environ.get("ZENODO_TOKEN") or (Path.home() / ".zenodo_token").read_text().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {t}"
    return s


def check(r):
    if not r.ok:
        sys.exit(f"{r.request.method} {r.url} -> {r.status_code}\n{r.text[:2000]}")
    return r.json() if r.content else {}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def metadata(version):
    return {"metadata": {
        "upload_type": "dataset",
        "title": f"Cascadia fault surfaces {version} from the catalog v4 seismicity, with per-vertex uncertainties"
                 + (" (method test)" if version == "v0" else ""),
        "description": DESCRIPTION,
        "creators": [{"name": "Denolle, Marine A.", "affiliation": "University of Washington"}],
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": version,
        "keywords": ["Cascadia", "fault surfaces", "seismicity", "Blanco transform", "Mendocino triple junction",
                     "Gorda plate", "ocean-bottom seismometer", "uncertainty"],
        "related_identifiers": [{"identifier": "https://github.com/Denolle-Lab/cascadia_obs_ensemble",
                                 "relation": "isSupplementTo", "resource_type": "software"}],
    }}


def upload(s, dep, folder):
    bucket = dep["links"]["bucket"]
    for f in sorted(Path(folder).iterdir()):
        if f.name == "run.log" or not f.is_file():
            continue
        with open(f, "rb") as fh:
            r = check(s.put(f"{bucket}/{f.name}", data=fh))
        ok = r.get("checksum", "").endswith(md5(f))
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB, md5 {'ok' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit("checksum mismatch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["create", "publish", "newversion"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--sandbox", action="store_true")
    a = ap.parse_args()
    s, B = session(), base(a.sandbox)
    if a.cmd == "create":
        folder = Path(a.args[0])
        dep = check(s.post(f"{B}/deposit/depositions", json={}))
        upload(s, dep, folder)
        check(s.put(f"{B}/deposit/depositions/{dep['id']}", json=metadata(folder.name)))
        print(f"draft {dep['id']}: {dep['links']['html']}")
    elif a.cmd == "publish":
        d = check(s.post(f"{B}/deposit/depositions/{a.args[0]}/actions/publish"))
        print(f"published: DOI {d['doi']}  {d['links']['record_html'] if 'record_html' in d['links'] else d['links']['html']}")
    else:
        rec, folder = a.args[0], Path(a.args[1])
        d = check(s.post(f"{B}/deposit/depositions/{rec}/actions/newversion"))
        dep = check(s.get(d["links"]["latest_draft"]))
        for f in dep.get("files", []):
            check(s.delete(f"{B}/deposit/depositions/{dep['id']}/files/{f['id']}"))
        upload(s, dep, folder)
        check(s.put(f"{B}/deposit/depositions/{dep['id']}", json=metadata(folder.name)))
        print(f"new version draft {dep['id']}: {dep['links']['html']}")


if __name__ == "__main__":
    main()
