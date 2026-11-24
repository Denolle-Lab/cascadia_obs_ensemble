"""Run manifest for audit: call write_manifest() at the end of every Track B script.

Records: script, UTC time, git commit and dirty flag, md5 of params.yaml, the parameters used,
md5 + size of each input and output file (files > 2 GB: size + mtime only), and free-form counts.
Written next to the outputs as <name>.manifest.json and appended to <base>/manifests.jsonl.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARAMS = HERE / "params.yaml"


def load_params():
    return yaml.safe_load(PARAMS.read_text())


def tomodd_bin(kind, where):
    """tomoDD binary of `kind` ('bin', 'bin_syn', 'bin_check') for the region of `where` (a region name, a run
    path relative to tomodd.runs, or an absolute path under it). tomodd.regions.<region>.bins overrides the
    global entry (e.g. a build with larger MAXOBS for one region)."""
    T = load_params()["tomodd"]
    p, runs = Path(where), Path(T["runs"])
    try:
        region = p.relative_to(runs).parts[0] if p.is_absolute() else p.parts[0]
    except ValueError:
        region = None
    return str(T["regions"].get(region, {}).get("bins", {}).get(kind, T[kind]))


def _md5(path, limit=2 * 1024**3):
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    out = {"path": str(p), "size": st.st_size,
           "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    if p.is_file() and st.st_size <= limit:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        out["md5"] = h.hexdigest()
    return out


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def write_manifest(name, outdir, inputs=(), outputs=(), params=None, counts=None, notes=None):
    rec = {
        "name": name,
        "script": str(Path(sys.argv[0]).resolve()),
        "argv": sys.argv[1:],
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", str(HERE))),
        "params_md5": _md5(PARAMS).get("md5"),
        "params": params,
        "inputs": [_md5(p) for p in inputs],
        "outputs": [_md5(p) for p in outputs],
        "counts": counts or {},
        "notes": notes,
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.manifest.json").write_text(json.dumps(rec, indent=1, default=str))
    base = Path(load_params()["paths"]["base"])
    with open(base / "manifests.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec
