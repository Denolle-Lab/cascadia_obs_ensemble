"""S-mismatch test: tomoDD 1-iteration residuals at B4 and B5 positions of one B5 cluster (scratch copy)."""
import re, shutil, subprocess, sys
from pathlib import Path

cid = sys.argv[1]
SRC = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/clusters") / cid
W = Path("/wd1/mdenolle_data/tomo_v1/b5/b5_v0.3.1/diag/s_mismatch") / cid
for tag in ("b4", "b5"):
    d = W / tag
    if (d / "tomoDD-SE.res.ini").exists():
        continue
    (d / "Input_Files").mkdir(parents=True, exist_ok=True)
    (d / "Output_Files").mkdir(exist_ok=True)
    for f in ("MOD", "ak135.15.SKS", "layer-16.dat"):
        shutil.copy(SRC / f, d / f)
    for f in (SRC / "Input_Files").iterdir():
        shutil.copy(f, d / "Input_Files" / f.name)
    L = (SRC / "tomoDD-SE.inp").read_text().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("*  ISTART"))
    L[i + 1] = re.sub(r"(\d+)\s*$", "1", L[i + 1])                      # NSET 1
    j = next(k for k, l in enumerate(L) if l.startswith("* NITER"))
    L[j + 1] = re.sub(r"^\s*\d+", "  1", L[j + 1])                       # NITER 1, first set only
    del L[j + 2:j + 5]
    (d / "tomoDD-SE.inp").write_text("\n".join(L) + "\n")
    if tag == "b5":   # positions of tomoDD.reloc (tomoDD frame), origin times of event.dat kept
        rel = {}
        for l in (SRC / "Output_Files" / "tomoDD.reloc").read_text().splitlines():
            p = l.split()
            rel[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
        out = []
        for l in (d / "Input_Files" / "event.dat").read_text().splitlines():
            p = l.split()
            eid = int(p[9])
            if eid in rel:
                la, lo, z = rel[eid]
                out.append(f"{p[0]}  {p[1]} {la:10.6f} {lo:11.6f} {z:8.3f} {' '.join(p[5:9])} {eid:10d} 0")
        (d / "Input_Files" / "event.dat").write_text("\n".join(out) + "\n")
    exe = __import__("json").loads((SRC / "run.json").read_text())["bin"]
    with open(d / "stdout.txt", "w") as so:
        subprocess.run([exe, "tomoDD-SE.inp"], cwd=d, stdout=so, stderr=subprocess.STDOUT, timeout=3600)
    print(tag, "res.ini" if (d / "tomoDD-SE.res.ini").exists() else "MISSING", flush=True)
