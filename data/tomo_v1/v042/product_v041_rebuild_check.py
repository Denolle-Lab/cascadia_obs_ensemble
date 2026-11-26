"""Check that the (refactored) b2_model_v041.build_nodes in tomo/ensemble-fix rebuilds the registered v0.4.1 node values
from merged_v041.nc (in memory, nothing written)."""
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, "/home/mdenolle/wt_ensfix/workflow/10_tomography")
import b2_model_v041 as M  # noqa: E402

ds, c, L, base = M.build_nodes()
ref = xr.open_dataset(M.STORE / "v0.4.1.nc")
out = {v: bool(np.array_equal(ds[v].values, ref[v].values, equal_nan=True)) for v in ref.data_vars}
attrs = sorted(k for k in set(ds.attrs) | set(ref.attrs) if str(ds.attrs.get(k)) != str(ref.attrs.get(k)))
res = dict(reference=str(M.STORE / "v0.4.1.nc"), variables_equal=out, all_equal=all(out.values()),
           attrs_differing=attrs, info=c["info"])
Path("/wd1/mdenolle_data/tomo_v1/v042/merge_check/product_v041_rebuild_check.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
