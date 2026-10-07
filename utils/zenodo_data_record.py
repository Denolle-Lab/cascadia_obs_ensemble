"""Zenodo draft for the data record of the paper, with a reserved DOI (CLEANUP_PLAN §6).

A reserved DOI can be cited in the manuscript before anything is public. It resolves
only once the record is published; until then the draft, its files and its metadata
stay private and can still change. Publishing is done by hand on zenodo.org.

Token as for zenodo_picks_draft.py ($ZENODO_TOKEN or ~/.zenodo_token).

    python utils/zenodo_data_record.py create             # new draft + reserved DOI
    python utils/zenodo_data_record.py upload ID FILE ...  # add or replace files
    python utils/zenodo_data_record.py show ID
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zenodo_picks_draft import api, check, session, show, upload  # noqa: E402

TITLE = "Cascadia OBS ensemble earthquake catalog 2010-2015, v3"
DESCRIPTION = """<p>Earthquake catalog of offshore Cascadia from the Cascadia Initiative
ocean-bottom seismometers and the onshore networks, 2010-2015, built with an ensemble
machine-learning picker (ELEP), GENIE association and GraphDD relocation with waveform
cross-correlation. Magnitudes are Hutton-Boore local magnitudes with station terms,
replaced by moment magnitudes above M 4.5.</p>
<p>Three ANSS-style tables (origins, arrivals, picks), the magnitude station terms, the
matched ComCat moment tensors and the comparison catalogs. See README.md and
DATA_DICTIONARY.md in the record.</p>"""
CREATORS = [
    ("Denolle, Marine A.", "University of Washington"),
    ("Bito, Hiroto", "University of Washington"),
    ("Shi, Qibin", "University of Washington"),
    ("Ni, Yiyu", "University of Washington"),
    ("McBrearty, Ian W.", "Stanford University"),
    ("Krauss, Zoe", "University of Washington"),
    ("Stevens, Nathan T.", "University of Washington"),
    ("Yu, Yifan", "Stanford University"),
    ("Beroza, Gregory C.", "Stanford University"),
]


def create(s, base):
    meta = {"metadata": {
        "upload_type": "dataset",
        "title": TITLE,
        "description": DESCRIPTION,
        "creators": [{"name": n, "affiliation": a} for n, a in CREATORS],
        "access_right": "open",
        "license": "cc-by-4.0",
        "keywords": ["Cascadia", "earthquake catalog", "ocean-bottom seismometer",
                     "machine learning", "ELEP", "GENIE", "GraphDD"],
        "version": "v3",
        "prereserve_doi": True,
    }}
    d = check(s.post(f"{base}/deposit/depositions", json=meta))
    doi = d["metadata"].get("prereserve_doi", {}).get("doi")
    print(f"draft id {d['id']}\nreserved DOI {doi}\nedit: {d['links']['html']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create")
    u = sub.add_parser("upload"); u.add_argument("id"); u.add_argument("files", nargs="+")
    sub.add_parser("show").add_argument("id")
    a = ap.parse_args()
    s, base = session(), api(a.sandbox)
    if a.cmd == "create":
        create(s, base)
    elif a.cmd == "upload":
        upload(s, base, a.id, a.files)
    else:
        show(s, base, a.id)


if __name__ == "__main__":
    main()
