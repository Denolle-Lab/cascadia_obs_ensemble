"""Build the gaia-skills version of the pick-timing skill (skill name correcting-pick-timing) from this repository.

The skill folder is self-contained (clients install one skill folder at a time):
  SKILL.md                 from .claude/skills/pick-timing/SKILL.md, renamed, links pointed at references/, the
                           "Get the code" section replaced by the bundled-code / pinned-install text
  references/*.md          copies of tools/pick_timing/docs/*.md
  scripts/pick_timing/     the package (python files only)
  scripts/run_pick_timing.py  launcher (python scripts/run_pick_timing.py <command> ...)
  scripts/example_config.yaml copy of configs/example_generic.yaml (runs on the synthetic example)
  scripts/tests/           the unit tests
  scripts/pyproject.toml   so that `pip install ./scripts` works
Then checks: frontmatter, description length, line count, links, private paths, the gaia security patterns.

  python tools/pick_timing/scripts/export_gaia_skill.py --dest DIR/correcting-pick-timing --commit <sha>
"""
from __future__ import annotations

import argparse
import re
import shutil
import stat
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                      # tools/pick_timing
REPO = PKG.parent.parent
NAME = "correcting-pick-timing"
REPO_URL = "https://github.com/Denolle-Lab/cascadia_obs_ensemble"

GET_CODE = """## 0. Get the code

The code is bundled in this skill's `scripts/` folder: package `pick_timing` {version}, exported from commit
`{commit}` of the cascadia_obs_ensemble repository (folder `tools/pick_timing`). It needs Python >= 3.9, numpy,
scipy, pandas >= 2.0, obspy >= 1.4 and pyyaml (tested with Python 3.10, numpy 1.26.4, scipy 1.15.2, pandas 2.3.3,
obspy 1.5.0, pyyaml 6.0.3; pyarrow only for parquet tables). Check that they are installed; ask before installing
anything. `SKILL_DIR` below is the folder that holds this SKILL.md.

```bash
python $SKILL_DIR/scripts/run_pick_timing.py --help
python -m unittest discover -s $SKILL_DIR/scripts/tests -t $SKILL_DIR/scripts    # 17 tests, ~15 s; all must pass
```
Or install the same code into the user's environment (ask first):
```bash
pip install "pick-timing @ git+{repo_url}@{commit}#subdirectory=tools/pick_timing"
```
Below, `python -m pick_timing` stands for either `python $SKILL_DIR/scripts/run_pick_timing.py` or the installed
module. To see the whole procedure on synthetic data first (12 stations, one drifting clock, one offset epoch, one
drifting MiniSEED day; the truth is in `example_data/truth_*.csv`):
```bash
python -m pick_timing make-example --out example_data
python -m pick_timing detect-clock -c $SKILL_DIR/scripts/example_config.yaml
python -m pick_timing report -c $SKILL_DIR/scripts/example_config.yaml
```

"""

LAUNCHER = '''#!/usr/bin/env python3
"""Run the bundled pick_timing package: python run_pick_timing.py <command> [options] (see --help)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_timing.cli import main  # noqa: E402

sys.exit(main())
'''

SECURITY = [r"allowed-tools", r"curl", r"wget", r"\| *(ba|z)?sh", r"sudo", r"rm -rf", r"https?://", r"<!--",
            r"base64", r"ignore (all|previous)"]
PRIVATE = [r"/wd1", r"/auto/", r"/home/", r"mdenolle", r"PNWstore_sqlite", r"@uw\.edu", r"washington\.edu"]


def build(dest: Path, commit: str):
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "references").mkdir(parents=True)
    (dest / "scripts").mkdir()
    from importlib.util import module_from_spec, spec_from_file_location
    sp = spec_from_file_location("pt_init", PKG / "pick_timing" / "__init__.py")
    mod = module_from_spec(sp)
    sp.loader.exec_module(mod)
    version = mod.__version__
    # SKILL.md
    src = (REPO / ".claude" / "skills" / "pick-timing" / "SKILL.md").read_text()
    s = src.replace("name: pick-timing", f"name: {NAME}", 1)
    s = s.replace("tools/pick_timing/docs/", "references/")
    s = s.replace("tools/pick_timing/configs/example_generic.yaml", "scripts/example_config.yaml")
    a = s.index("## 0. Get the code")
    b = s.index("## Ask a person first")
    s = s[:a] + GET_CODE.format(version=version, commit=commit, repo_url=REPO_URL) + s[b:]
    (dest / "SKILL.md").write_text(s)
    for f in sorted((PKG / "docs").glob("*.md")):
        shutil.copy(f, dest / "references" / f.name)
    shutil.copytree(PKG / "pick_timing", dest / "scripts" / "pick_timing",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(PKG / "tests", dest / "scripts" / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(PKG / "configs" / "example_generic.yaml", dest / "scripts" / "example_config.yaml")
    pp = (PKG / "pyproject.toml").read_text().replace('readme = "README.md"\n', "")
    (dest / "scripts" / "pyproject.toml").write_text(pp)
    ln = dest / "scripts" / "run_pick_timing.py"
    ln.write_text(LAUNCHER)
    for f in [ln] + sorted((dest / "scripts").rglob("*.py")):   # gaia-skills: bundled scripts are executable
        f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return version


def check(dest: Path) -> list[str]:
    msgs = []
    s = (dest / "SKILL.md").read_text()
    m = re.match(r"---\n(.*?)\n---\n", s, re.S)
    if not m:
        msgs.append("ERROR no frontmatter")
    else:
        import yaml
        fm = yaml.safe_load(m.group(1))
        if fm.get("name") != dest.name:
            msgs.append(f"ERROR name {fm.get('name')} != folder {dest.name}")
        n = len(fm.get("description", ""))
        msgs.append(f"description: {n} characters (100-1024)" + ("" if 100 <= n <= 1024 else "  ERROR"))
    nl = s.count("\n")
    msgs.append(f"SKILL.md: {nl} lines (< ~500)" + ("" if nl < 500 else "  ERROR"))
    for md in [dest / "SKILL.md"] + sorted((dest / "references").glob("*.md")):
        for link in re.findall(r"\]\(([^)]+)\)", md.read_text()):
            if link.startswith("http"):
                continue
            target = (md.parent / link).resolve()
            if not str(target).startswith(str(dest.resolve())) or not target.exists():
                msgs.append(f"ERROR link {link} in {md.name} leaves the skill folder or is missing")
    for f in sorted(dest.rglob("*")):
        if not f.is_file():
            continue
        t = f.read_text(errors="replace")
        rel = f.relative_to(dest)
        for pat in PRIVATE:
            for mm in re.finditer(pat, t):
                line = t.count("\n", 0, mm.start()) + 1
                msgs.append(f"ERROR private pattern {pat!r} in {rel}:{line}")
        for pat in SECURITY:
            for mm in re.finditer(pat, t):
                line = t.count("\n", 0, mm.start()) + 1
                msgs.append(f"review: {pat!r} in {rel}:{line}: {t.splitlines()[line - 1].strip()[:100]}")
        if re.search("[​-‏‪-‮⁠-⁤﻿]", t):
            msgs.append(f"ERROR invisible unicode in {rel}")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help=f"output skill folder (its name must be {NAME})")
    ap.add_argument("--commit", required=True, help="pushed commit of this repository the pinned install points to")
    a = ap.parse_args()
    dest = Path(a.dest)
    if dest.name != NAME:
        sys.exit(f"--dest must end in /{NAME}")
    v = build(dest, a.commit)
    msgs = check(dest)
    print(f"{dest}: pick_timing {v}, commit {a.commit}")
    for m in msgs:
        print("  " + m)
    return 1 if any(m.startswith("ERROR") for m in msgs) else 0


if __name__ == "__main__":
    sys.exit(main())
