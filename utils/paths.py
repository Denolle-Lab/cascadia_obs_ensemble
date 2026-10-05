"""Repository paths, the same from any script or notebook.

    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from utils.paths import REPO, DATA

Scripts in workflow/NN_*/ can also keep the relative ``../../data``: every stage
directory sits two levels below the repository root.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
WORKFLOW = REPO / "workflow"
FIGURES = REPO / "figures"
PAPER_FIGURES = REPO / "paper" / "figures"
