"""Track B (D-B047): the Zenodo bundle velocity_models.zip, built around one product version (PI 2026-11-23: the
merged model is the Zenodo product).

Contents: README.md and read_velocity_model.py (tutorial), registry.csv, and for every registered version its
netCDF, Zarr store and provenance JSON; the NonLinLoc grids only for the product version (the earlier versions'
grids can be regenerated with b2_model_export.py; v0.1.0's were sampled at the nodes, not the cell centres).
Writes <store>/zenodo/velocity_models.zip and prints its md5 (upload with utils/zenodo_picks_draft.py upload).

  python b2_model_bundle.py PRODUCT_VERSION
"""
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from manifest import REPO, load_params  # noqa: E402

X = load_params()["tomodd"]["models"]
STORE = Path(X["store"]) / X["name"]


def main():
    product = sys.argv[1]
    reg = pd.read_csv(REPO / "data" / "tomo_v1" / "models" / "registry.csv", dtype={"version": str})
    assert (reg.version == product).any(), f"v{product} not registered"
    out = Path(X["store"]) / "zenodo" / "velocity_models.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.partial")
    top = "velocity_models"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(REPO / "data" / "tomo_v1" / "models" / "README.md", f"{top}/README.md")
        z.write(HERE / "read_velocity_model.py", f"{top}/read_velocity_model.py")
        z.write(STORE / "registry.csv", f"{top}/registry.csv")
        z.write(STORE / "registry.csv", f"{top}/{X['name']}/registry.csv")
        for v in reg.version:
            for f in (f"v{v}.nc", f"v{v}.provenance.json"):
                z.write(STORE / f, f"{top}/{X['name']}/{f}")
            for p in sorted((STORE / f"v{v}.zarr").rglob("*")):
                if p.is_file():
                    z.write(p, f"{top}/{X['name']}/{p.relative_to(STORE)}")
            if v == product and (STORE / f"v{v}_nll").exists():
                for p in sorted((STORE / f"v{v}_nll").iterdir()):
                    z.write(p, f"{top}/{X['name']}/v{v}_nll/{p.name}")
        z.writestr(f"{top}/PRODUCT.txt", f"Product version: {product} ({reg[reg.version == product].description.iloc[0]})\n"
                                         f"NonLinLoc grids included for v{product} only.\n")
    shutil.move(tmp, out)
    h = hashlib.md5()
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    print(out, out.stat().st_size, h.hexdigest())


if __name__ == "__main__":
    main()
