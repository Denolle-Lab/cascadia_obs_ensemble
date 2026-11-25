"""Full B5 relocation of one cluster under weighting variants, then a 1-iteration evaluation at the final positions."""
import json, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

cid = sys.argv[1]
SRC = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/clusters") / cid
W = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/diag/s_mismatch") / cid / "var"
EXE = json.loads((SRC / "run.json").read_text())["bin"]
# production sets: NITER WTCCP WTCCS WRCC WDCC WTCTP WTCTS WRCT WDCT WTCD DAMP JOINT THRES
PROD = [l for l in (SRC / "tomoDD-SE.inp").read_text().splitlines() if re.match(r"^\s+3\s+-9", l)]
assert len(PROD) == 4, PROD

def sets(wp=1.0, ws=1.0, rew=True):
    out = []
    for l in PROD:
        p = l.split()
        p[5], p[6] = f"{wp:.2f}", f"{ws:.2f}"
        if not rew:
            p[7], p[8] = "-9", "-9"
        out.append("  " + "    ".join(p))
    return out

VARIANTS = {"prod": sets(), "s_eq": sets(ws=4.0), "norew": sets(rew=False), "s_eq_norew": sets(ws=4.0, rew=False),
            "p_only": sets(ws=0.0), "s_only": sets(wp=0.0, ws=4.0)}

def prep(d, event_dat=None):
    (d / "Input_Files").mkdir(parents=True, exist_ok=True)
    (d / "Output_Files").mkdir(exist_ok=True)
    for f in ("MOD", "ak135.15.SKS", "layer-16.dat"):
        shutil.copy(SRC / f, d / f)
    for f in (SRC / "Input_Files").iterdir():
        shutil.copy(f, d / "Input_Files" / f.name)
    if event_dat:
        (d / "Input_Files" / "event.dat").write_text(event_dat)

def write_inp(d, set_lines, nset):
    L = (SRC / "tomoDD-SE.inp").read_text().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("*  ISTART"))
    L[i + 1] = re.sub(r"(\d+)\s*$", str(nset), L[i + 1])
    j = next(k for k, l in enumerate(L) if l.startswith("* NITER"))
    L[j + 1:j + 5] = set_lines
    (d / "tomoDD-SE.inp").write_text("\n".join(L) + "\n")

def run(d):
    with open(d / "stdout.txt", "w") as so:
        subprocess.run([EXE, "tomoDD-SE.inp"], cwd=d, stdout=so, stderr=subprocess.STDOUT, timeout=3600)

def reloc_event_dat(d):
    rel = {int(l.split()[0]): l.split()[1:4] for l in (d / "Output_Files" / "tomoDD.reloc").read_text().splitlines()}
    out = []
    for l in (SRC / "Input_Files" / "event.dat").read_text().splitlines():
        p = l.split()
        eid = int(p[9])
        if str(eid) and eid in rel:
            la, lo, z = map(float, rel[eid])
            out.append(f"{p[0]}  {p[1]} {la:10.6f} {lo:11.6f} {z:8.3f} {' '.join(p[5:9])} {eid:10d} 0")
    return "\n".join(out) + "\n"

def one(name):
    d = W / name
    if not (d / "Output_Files" / "tomoDD.reloc").exists():
        prep(d)
        write_inp(d, VARIANTS[name], 4)
        run(d)
    e = d / "eval"
    if not (e / "tomoDD-SE.res.ini").exists():
        prep(e, reloc_event_dat(d))
        ev1 = PROD[0].split()
        ev1[0] = "1"
        write_inp(e, ["  " + "    ".join(ev1)], 1)
        run(e)
    return name

with ThreadPoolExecutor(len(VARIANTS)) as ex:
    for n in ex.map(one, VARIANTS):
        print("done", n, flush=True)
