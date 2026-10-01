"""Upload ELEP pick tables to an unpublished Zenodo draft, shared by link with collaborators.

The draft is never published by this script: nothing is public and no DOI is minted until
someone presses Publish on zenodo.org. Collaborators see the draft and its files through
a share link (preview permission).

Token: a Zenodo personal access token with scopes deposit:write and deposit:actions,
in $ZENODO_TOKEN or in ~/.zenodo_token (chmod 600).

    python utils/zenodo_picks_draft.py create                     # new draft, prints its id
    python utils/zenodo_picks_draft.py upload ID FILE [FILE ...]  # add or replace files
    python utils/zenodo_picks_draft.py link ID                    # make a preview share link
    python utils/zenodo_picks_draft.py show ID                    # files, sizes, md5
Add --sandbox to use sandbox.zenodo.org for a test.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import requests

TITLE = ("Cascadia OBS ensemble: ELEP phase picks 2010-2015, ver3 and ver4 "
         "(working copy for the GENIE and GraphDD rerun)")
DESCRIPTION = """<p>ELEP ensemble phase picks (five EQTransformer models, SeisBench) on
the Cascadia Initiative ocean-bottom and nearby land stations, 40-50&deg;N, 131-122&deg;W,
2010-2015. Working copy shared with collaborators before publication; not for redistribution.</p>
<ul>
<li><b>ver3</b> (<code>ver3/all_picks_all_regions_2010_2015_ver3.csv.gz</code>): 39,597,551 picks,
the table used for the published v3 catalog association.</li>
<li><b>ver4</b> (<code>ver4/all_picks_all_regions_2010_2015_ver4.csv.gz</code>): 49,541,718 picks, ver3
byte for byte (same pick_id) followed by 9,944,167 picks on 75,709 station-days that the ver3
merge left out. <code>ver4_added_station_days.csv</code> gives the category and pick_id range of each
added station-day; see <code>README.md</code>.</li>
</ul>
<p>Two ver4 categories (edge_run_dropped, v1_not_in_ver3) are still to be confirmed and can be
dropped by pick_id range.</p>"""


def api(sandbox):
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def token():
    t = os.environ.get("ZENODO_TOKEN")
    p = Path.home() / ".zenodo_token"
    if not t and p.exists():
        t = p.read_text().strip()
    if not t:
        sys.exit("No token: set ZENODO_TOKEN or write it to ~/.zenodo_token")
    return t


def session():
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token()}"
    return s


def check(r):
    if not r.ok:
        sys.exit(f"{r.request.method} {r.url} -> {r.status_code}\n{r.text[:2000]}")
    return r.json() if r.content else {}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def create(s, base):
    meta = {"metadata": {
        "upload_type": "dataset",
        "title": TITLE,
        "description": DESCRIPTION,
        "creators": [{"name": "Denolle, Marine A.", "affiliation": "University of Washington"}],
        "access_right": "restricted",
        "access_conditions": "Shared with collaborators for the GENIE/GraphDD rerun before publication.",
        "keywords": ["Cascadia", "ocean-bottom seismometer", "phase picks", "EQTransformer", "ELEP"],
        "version": "ver3+ver4 working copy",
    }}
    d = check(s.post(f"{base}/deposit/depositions", json=meta))
    print(f"draft id {d['id']}\nedit: {d['links']['html']}")


def upload(s, base, dep_id, files):
    d = check(s.get(f"{base}/deposit/depositions/{dep_id}"))
    bucket = d["links"]["bucket"]
    for f in map(Path, files):
        # Zenodo file keys cannot contain '/': prefix with the parent folder instead
        key = f"{f.parent.name}_{f.name}" if f.parent.name in ("ver3", "ver4") else f.name
        local = md5(f)
        print(f"{key}: {f.stat().st_size / 1e9:.2f} GB, md5 {local}", flush=True)
        with open(f, "rb") as fh:
            r = check(s.put(f"{bucket}/{key}", data=fh, timeout=None))
        remote = r.get("checksum", "").removeprefix("md5:")
        print(f"  uploaded, remote md5 {remote} {'OK' if remote == local else 'MISMATCH'}")


def link(s, base, dep_id):
    r = check(s.post(f"{base}/records/{dep_id}/access/links", json={"permission": "preview"}))
    host = base.removesuffix("/api")
    # the link grants access to the draft: keep it out of terminal logs
    out = Path.home() / "zenodo_picks_share_link.txt"
    out.write_text(f"{host}/records/{dep_id}?preview=1&token={r['token']}\n")
    out.chmod(0o600)
    print(f"share link written to {out}")


def show(s, base, dep_id):
    d = check(s.get(f"{base}/deposit/depositions/{dep_id}"))
    print(d["metadata"]["title"], "| submitted:", d["submitted"])
    for f in d["files"]:
        print(f"  {f['filename']:60s} {f['filesize'] / 1e9:6.2f} GB  {f['checksum']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create")
    u = sub.add_parser("upload"); u.add_argument("id"); u.add_argument("files", nargs="+")
    for name in ("link", "show"):
        sub.add_parser(name).add_argument("id")
    a = ap.parse_args()
    s, base = session(), api(a.sandbox)
    if a.cmd == "create":
        create(s, base)
    elif a.cmd == "upload":
        upload(s, base, a.id, a.files)
    elif a.cmd == "link":
        link(s, base, a.id)
    else:
        show(s, base, a.id)


if __name__ == "__main__":
    main()
