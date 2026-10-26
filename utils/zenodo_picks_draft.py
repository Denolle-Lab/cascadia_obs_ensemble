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
    python utils/zenodo_picks_draft.py describe ID                # set title/description/version below
Add --sandbox to use sandbox.zenodo.org for a test.
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import requests

TITLE = ("Cascadia OBS ensemble: ELEP phase picks 2010-2016 (ver3 to ver6) and PickBlue "
         "picks on the same OBS stations, with Route A amplitudes for every ver5 and ver6 pick (working copy "
         "for the GENIE and GraphDD rerun)")
DESCRIPTION = """<p>ELEP ensemble phase picks (five EQTransformer models, SeisBench) on
the Cascadia Initiative ocean-bottom and nearby land stations, 40-50&deg;N, 131-122&deg;W,
2010-2015. Working copy shared with collaborators before publication; not for redistribution.</p>
<ul>
<li><b>ver3</b> (<code>all_picks_all_regions_2010_2015_ver3.csv.gz</code>): 39,597,551 picks,
the table used for the published v3 catalog association.</li>
<li><b>ver4</b>: 49,541,718 picks, ver3 byte for byte (same pick_id) followed by 9,944,167
picks on 75,709 station-days that the ver3 merge left out (<code>ver4_added_picks.csv.gz</code>).
<code>ver4_added_station_days.csv</code> gives the category and pick_id range of each added
station-day; see <code>ver4_README.md</code>.</li>
<li><b>ver5</b>: 60,918,847 picks, ver4 byte for byte followed by 11,377,129 picks of the
offshore gap fill (48,638 station-days at 212 ocean-bottom stations, 2010-2016, picked with
the model set of the region runs). Rebuild it from ver3, the ver4 added picks and the two
parts of the ver5 added picks (Zenodo rejects the 344 MB file in one piece); see
<code>ver5_HOW_TO_BUILD.txt</code> and <code>ver5_README.md</code>, which also lists stations
to watch (high pick rates on the shallow FN shelf stations).</li>
<li><b>PickBlue</b> (<code>pickblue_picks_part1-8.csv.gz</code>): 18,334,658 picks of the
PickBlue <code>obs</code> model (threshold 0.2) at the 350 OBS stations that also have ELEP
ver5 picks, 2010-2016, taken from the public QuakeScope 2026 OBS catalogue
(<code>s3://quakescope-picks-2026/obs/</code>) and written in the ELEP column layout for a
direct comparison. <code>pickblue_station_days.csv</code> says which station-days PickBlue
ran on; <code>pickblue_README.md</code> gives the column map, the differences with ELEP and a
first comparison.</li>
<li><b>Amplitudes</b> (<code>ver5_amps_NET_partNN.parquet</code>, release r2, 2026-10-18): Route A
Wood-Anderson and 0.05-2 Hz displacement peaks for every ver5 pick, joined on
<code>pick_id</code>, over 69 windows after the pick (1-60 s), since the distance is not known
before association. 57.9 M of the 60.9 M picks (95%) have an amplitude; <code>status</code> says
why the others do not. The amplitude columns are stored log-encoded; read them with
<code>read_amps_v5.py</code>. See <code>ver5_amps_README.md</code>, which lists the changes
since r1.</li>
<li><b>ver6</b> (2026-10-26): 73,636,558 picks, ver5 byte for byte followed by 12,717,711 picks
on the land strip 122-123&deg;W, 40-50&deg;N, 2010-2016 (137,469 station-days, 199 stations,
128 of them new), picked and measured (Route A amplitudes, same settings as r2) in one pass.
The new picks are in <code>ver6_added_picks_part1-3.csv.gz</code>, their amplitudes in
<code>ver6_amps_NET_partNN.parquet</code> (98.8% with an amplitude). 1,846 station-days with
fragmented data (1.15 M artifact picks) were left out (<code>ver6_dropped_station_days.csv</code>).
See <code>ver6_README.md</code>.</li>
<li><b>Stations</b> (<code>stations_ver5_v6.csv</code>): coordinates and epochs of the 728
stations of ver5 and ver6 (<code>stations_README.md</code>).</li>
</ul>
<p>All ver4 categories are kept (decision of the PI, 2026-10-05). Any category can still be
dropped by its pick_id ranges.</p>"""


VERSION = "ver3+ver4+ver5+ver6+PickBlue+amps r2 working copy"
TRIES = 4


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
        "version": VERSION,
    }}
    d = check(s.post(f"{base}/deposit/depositions", json=meta))
    print(f"draft id {d['id']}\nedit: {d['links']['html']}")


def upload(s, base, dep_id, files):
    d = check(s.get(f"{base}/deposit/depositions/{dep_id}"))
    bucket = d["links"]["bucket"]
    for f in map(Path, files):
        # Zenodo file keys cannot contain '/': prefix with the parent folder instead
        prefix = f.parent.name if f.parent.name in ("ver3", "ver4", "ver5") else ""
        key = f.name if not prefix or f.name.startswith(prefix) or prefix in f.name else f"{prefix}_{f.name}"
        local = md5(f)
        print(f"{key}: {f.stat().st_size / 1e9:.2f} GB, md5 {local}", flush=True)
        for attempt in range(1, TRIES + 1):
            with open(f, "rb") as fh:
                resp = s.put(f"{bucket}/{key}", data=fh, timeout=None)
            if resp.ok:
                break
            print(f"  try {attempt}: {resp.status_code}", flush=True)
            # a gateway error can come after the file was stored: check before resending
            have = {x["filename"]: x["checksum"] for x in check(s.get(f"{base}/deposit/depositions/{dep_id}"))["files"]}
            if have.get(key, "").removeprefix("md5:") == local:
                resp = None
                break
            time.sleep(60 * attempt)
        r = {"checksum": f"md5:{local}"} if resp is None else check(resp)
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


def describe(s, base, dep_id):
    """Replace the draft's title, description and version, keeping its other metadata."""
    d = check(s.get(f"{base}/deposit/depositions/{dep_id}"))
    meta = {k: v for k, v in d["metadata"].items() if k not in ("doi", "prereserve_doi")}
    meta.update(title=TITLE, description=DESCRIPTION, version=VERSION)
    check(s.put(f"{base}/deposit/depositions/{dep_id}", json={"metadata": meta}))
    print(f"updated: {TITLE}")


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
    rm = sub.add_parser("delete-file"); rm.add_argument("id"); rm.add_argument("key")
    u = sub.add_parser("upload"); u.add_argument("id"); u.add_argument("files", nargs="+")
    for name in ("link", "show", "describe"):
        sub.add_parser(name).add_argument("id")
    a = ap.parse_args()
    s, base = session(), api(a.sandbox)
    if a.cmd == "create":
        create(s, base)
    elif a.cmd == "upload":
        upload(s, base, a.id, a.files)
    elif a.cmd == "delete-file":
        d = check(s.get(f"{base}/deposit/depositions/{a.id}"))
        check(s.delete(f"{d['links']['bucket']}/{a.key}"))
        print(f"deleted {a.key}")
    elif a.cmd == "link":
        link(s, base, a.id)
    elif a.cmd == "describe":
        describe(s, base, a.id)
    else:
        show(s, base, a.id)


if __name__ == "__main__":
    main()
