"""Build a BibTeX file from verified DOIs, with full records fetched from doi.org (content negotiation).

    python utils/build_bib.py --out paper2/references.bib --sources DOI_LIST_FILES...

A source file may be any text: every DOI-like string in it is collected, unless the line marks it as
wrong ("wrong", "not found", "different paper"). Each DOI is resolved with Accept: application/x-bibtex
(Crossref and DataCite both answer). Entries that fail are listed in <out>_failed.csv. Keys are
FirstAuthorYEAR with a/b suffixes; <out>_doi_map.csv maps doi -> key -> source files.
"""
import argparse, csv, re, time, unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

DOI = re.compile(r"10\.\d{4,9}/[^\s;,\[\]<>\"'|`{}]+")
SICI = re.compile(r"10\.1130/\d{4}-\d{4}\(\d{4}\)\d+[<\[][^>\]\s]+[>\]]2\.\d\.co;2", re.I)   # old GSA DOIs with < > ;
BAD = re.compile(r"(?i)wrong|not found|different paper|not verified|unverified|could not verify")


def collect(paths):
    found = {}
    for p in paths:
        for line in Path(p).read_text(errors="ignore").splitlines():
            if BAD.search(line):
                continue
            line = line.replace("&lt;", "<").replace("&gt;", ">").replace("%3c", "<").replace("%3C", "<").replace("%3e", ">").replace("%3E", ">")
            for d in SICI.findall(line):
                found.setdefault(d.lower(), set()).add(Path(p).name)
            line = SICI.sub(" ", line)
            for d in DOI.findall(line):
                d = d.rstrip(".:*_").lower()
                while d.endswith(")") and d.count(")") > d.count("("):   # drop a closing bracket of the text
                    d = d[:-1].rstrip(".:*_")
                found.setdefault(d, set()).add(Path(p).name)
    return found


def fetch(doi):
    for t in range(5):
        try:
            r = requests.get("https://doi.org/" + requests.utils.quote(doi, safe="/():.-_"), headers={"Accept": "application/x-bibtex; charset=utf-8",
                             "User-Agent": "bibtex-builder/0.1"}, timeout=30, allow_redirects=True)
            if r.status_code == 200 and r.text.strip().startswith("@"):
                return doi, r.text.strip(), 200
            if r.status_code in (404, 400):
                return doi, None, r.status_code
        except requests.RequestException:
            pass
        time.sleep(1 + 2 * t)
    return doi, None, -1


def ascii_name(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if c.isalpha() and ord(c) < 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    a = ap.parse_args()
    dois = collect(a.sources)
    print(f"{len(dois)} distinct DOIs")
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch, sorted(dois)))
    entries, keys, rows, failed = [], {}, [], []
    for doi, bib, st in res:
        if not bib:
            failed.append((doi, st, ";".join(sorted(dois[doi])))); continue
        au = re.search(r"author\s*=\s*\{([^}]+)\}", bib)
        yr = re.search(r"year\s*=\s*\{?(\d{4})", bib)
        first = au.group(1).split(" and ")[0].strip() if au else "Anon"
        toks = first.replace("{", " ").replace("}", " ").split()
        sur = first.split(",")[0] if "," in first else (toks[-1] if toks else "Anon")   # "Last, First" or "First Last" (DataCite)
        base = ascii_name(sur) + (yr.group(1) if yr else "")
        k = base
        i = 0
        while k in keys:
            i += 1; k = base + "abcdefghijklmnopqrstuvwxyz"[i - 1]
        keys[k] = doi
        bib = re.sub(r"^@(\w+)\{[^,]*,", lambda m: f"@{m.group(1)}{{{k},", bib, count=1)
        bib = re.sub(r",\s+(\w+)=", r",\n  \1=", bib)          # one field per line
        bib = re.sub(r"\s*\}\s*$", "\n}", bib)
        entries.append(bib)
        rows.append((doi, k, ";".join(sorted(dois[doi]))))
    out = Path(a.out)
    out.write_text("% Verified references: every entry resolved from its DOI at doi.org (Crossref or DataCite)\n"
                   f"% built by utils/build_bib.py on {time.strftime('%Y-%m-%d')}; {len(entries)} entries\n\n" + "\n\n".join(entries) + "\n")
    with open(out.with_name(out.stem + "_doi_map.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("doi", "key", "sources")] + rows)
    with open(out.with_name(out.stem + "_failed.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("doi", "status", "sources")] + failed)
    print(f"{len(entries)} entries written; {len(failed)} failed")


if __name__ == "__main__":
    main()
