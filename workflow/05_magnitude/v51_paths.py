"""Where the catalog v5.1 magnitude scripts find their inputs.

REPO is the checkout that holds this file (the main tree or a git worktree); tracked
inputs are read from it.

main_tree() is the main checkout. The inputs that git ignores (data/magnitude_v4/*.csv,
data/magnitude/, data/catalog_v4/*.csv, data/catalog_v4/report/,
data/catalog_v4/ridge_report/) and the pixi environment exist only there, not in a
worktree. It is found from the git common directory; the environment variable
CASCADIA_MAIN overrides it.

WD1 is the project data disk (default /wd1/mdenolle_data; the environment variable
CASCADIA_WD1 overrides it): catalog v5.1, the v4.1t picks, the station table and the
per-station-day amplitudes.

Each script also takes these paths as command-line options; the values here are the
defaults.
"""
from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WD1 = Path(os.environ.get("CASCADIA_WD1", "/wd1/mdenolle_data"))
V51_CATALOG = WD1 / "catalog_v5/v5.1/cascadia_catalog_v5.1.csv"
V41T_PICKS = WD1 / "catalog_v5/v5.1/Cascadia_picks_catalog_v4.1t.csv.gz"
STATIONS = WD1 / "picks_share/stations/stations_ver5_v6.csv"
AMPS = WD1 / "amps_v6_all"
REVIEW_SETS = WD1 / "catalog_v5/v5.1_review/work/sets"


@functools.lru_cache(maxsize=None)
def main_tree() -> Path:
    """The main checkout (parent of the git common directory), or REPO without git."""
    env = os.environ.get("CASCADIA_MAIN")
    if env:
        return Path(env)
    try:
        common = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return REPO
    return Path(common).parent


def main_data(*parts: str) -> Path:
    """A path under data/ of the main checkout."""
    return main_tree().joinpath("data", *parts)
