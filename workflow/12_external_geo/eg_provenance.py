"""Write /wd1/mdenolle_data/external_geo/PROVENANCE.csv (D-B114).

One row per raw file: dataset, version, DOI, URL, licence, download date, file, md5, full citation. Downloaded files
come from fetch_log.tsv (written by fetch.sh, dates from `date`); files extracted from an archive or read over HTTP
byte ranges from a remote zip (remote_zip.py) and the local inputs used as-is (GMRT tiles, Slab2) are added here with
their md5 computed now. Rerunning rewrites the file.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg_common as C  # noqa: E402

GHFDB_AUTHORS = ("Global Heat Flow Data Assessment Group; Fuchs, S.; Neumann, F.; Norden, B.; Balkan-Pazvantoglu, E.; "
                 "Elbarbary, S.; Petrunin, A.; Beardsmore, G.; Harris, R.; Negrete-Aranda, R.; Poort, J.; Verdoya, M.; "
                 "Liu, S.; Chambers, E.; Fuentes, K.; Sidigam, E. R.; Matiz-Leon, J. C.; Bencharef, M. H.; Mino, B. G.; "
                 "Khaled, M. S.; Verch, D.; Berger, L.; Chishti, S. F.; Dergunova, V.; Liebing, H.; Schulz, M.; "
                 "Schuppe, P.; Trepalova, Z.; Chiozzi, P.; Duque, M. R. A.; Forster, F.; Leveni, M.; Stal, T.")
META = {
    "ghfdb_r2024": dict(
        version="Release 2024, V. 2026.03 (published 2026-03-09)", doi="10.5880/fidgeo.2024.014",
        licence="CC BY 4.0",
        citation=(f"{GHFDB_AUTHORS} (2024): The Global Heat Flow Database: Release 2024. V. 2026.03. GFZ Data Services. "
                  "https://doi.org/10.5880/fidgeo.2024.014. Data description: Neumann, F., Norden, B., "
                  "Balkan-Pazvantoglu, E., Elbarbary, S., Petrunin, A., Elger, K., Jennings, S., Dergunova, V., Fuchs, "
                  "S. (2026). The 2024 release of the Global Heat Flow Database (GHFDB): quality assessment, metadata "
                  "standards, and a century of geothermal data. Earth System Science Data, 18, 4639-4668. "
                  "https://doi.org/10.5194/essd-18-4639-2026. Structure: Fuchs, S., et al. (2023). Tectonophysics, "
                  "863, 229976. https://doi.org/10.1016/j.tecto.2023.229976")),
    "agegrid_seton2020": dict(
        version="2020.1 (age GTS2012; misfit and confidence GeeK2007; misfit grid added 2022-06-30)",
        doi="10.1029/2020GC009214", licence="CC BY 4.0 (EarthByte agegrid License.txt)",
        citation=("Seton, M., Muller, R. D., Zahirovic, S., Williams, S., Wright, N. M., Cannon, J., Whittaker, J. M., "
                  "Matthews, K. J., McGirr, R. (2020). A global data set of present-day oceanic crustal age and "
                  "seafloor spreading parameters. Geochemistry, Geophysics, Geosystems, 21, e2020GC009214. "
                  "https://doi.org/10.1029/2020GC009214")),
    "globsed_v3": dict(
        version="GlobSed v3 (2019); Caltech Data mirror of the retired NOAA NCEI product",
        doi="10.1029/2018GC008115 (paper); 10.22002/k4070-ngc79 (data mirror)", licence="Public domain (mirror record)",
        citation=("Straume, E. O., Gaina, C., Medvedev, S., Hochmuth, K., Gohl, K., Whittaker, J. M., Abdul Fattah, R., "
                  "Doornenbal, J. C., Hopper, J. R. (2019). GlobSed: Updated total sediment thickness in the world's "
                  "oceans. Geochemistry, Geophysics, Geosystems, 20, 1756-1772. https://doi.org/10.1029/2018GC008115. "
                  "Data: Total Sediment Thickness for the World's Oceans and Marginal Seas, version 3, CaltechDATA, "
                  "https://doi.org/10.22002/k4070-ngc79 (archived from NOAA NCEI before its retirement on 2025-05-12)")),
    "gebco_2026": dict(
        version="GEBCO_2026 (published 2026-04-23); ice-surface elevation and TID, GeoTIFF tile n90 s0 w-180 e-90",
        doi="10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa",
        licence="Public domain (GEBCO terms of use); CEDA archive: Open Government Licence v3",
        citation=("GEBCO Bathymetric Compilation Group 2026 (2026). The GEBCO_2026 Grid - a continuous terrain model for "
                  "oceans and land at 15 arc-second intervals. NERC EDS British Oceanographic Data Centre NOC. "
                  "https://doi.org/10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa")),
    "crust1": dict(
        version="CRUST1.0 (1x1 deg; files dated 2014-08-27)", doi="none (EGU 2013 abstract)",
        licence="not stated on the distribution page (public download)",
        citation=("Laske, G., Masters, G., Ma, Z., Pasyanos, M. (2013). Update on CRUST1.0 - A 1-degree global model "
                  "of Earth's crust. Geophysical Research Abstracts, 15, EGU2013-2658. "
                  "https://igppweb.ucsd.edu/~gabi/crust1.html")),
    "vankeken_wilson_2023": dict(
        version="Zenodo v0.9.6 (2025-02-11); D80 and D80new Sepran runs, 03_British_Columbia (= North Cascadia) and "
                "04_Cascadia (= Central Cascadia)", doi="10.5281/zenodo.14854311 (concept 10.5281/zenodo.7843967)",
        licence="CC BY 4.0",
        citation=("van Keken, P. E., Wilson, C. R. (2025). An introductory review of thermal structure of subduction "
                  "zones: II & III (v0.9.6) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14854311. Papers: "
                  "Wilson, C. R., van Keken, P. E. (2023). An introductory review of the thermal structure of "
                  "subduction zones: II-numerical approach and validation. Progress in Earth and Planetary Science, "
                  "10, 68. https://doi.org/10.1186/s40645-023-00588-6; van Keken, P. E., Wilson, C. R. (2023). An "
                  "introductory review of the thermal structure of subduction zones: III-Comparison between models "
                  "and observations. Progress in Earth and Planetary Science, 10, 57. "
                  "https://doi.org/10.1186/s40645-023-00589-5; Syracuse, E. M., van Keken, P. E., Abers, G. A. (2010). "
                  "The global range of subduction zone thermal models. Physics of the Earth and Planetary Interiors, "
                  "183, 73-90. https://doi.org/10.1016/j.pepi.2010.02.004")),
    "gmrt_local": dict(
        version="GMRT Synthesis v4 (GridServer tiles, local copies from earlier sessions)", doi="10.1029/2008GC002332",
        licence="CC BY 4.0 (GMRT)",
        citation=("Ryan, W. B. F., Carbotte, S. M., Coplan, J. O., O'Hara, S., Melkonian, A., Arko, R., Weissel, R. A., "
                  "Ferrini, V., Goodwillie, A., Nitsche, F., Bonczkowski, J., Zemsky, R. (2009). Global Multi-"
                  "Resolution Topography synthesis. Geochemistry, Geophysics, Geosystems, 10, Q03014. "
                  "https://doi.org/10.1029/2008GC002332")),
    "slab2_local": dict(
        version="Slab2 cas_slab2_dep (local copy in the repository data/slab2)", doi="10.1126/science.aat4723; "
        "data 10.5066/F7PV6JNV", licence="USGS public domain",
        citation=("Hayes, G. P., Moore, G. L., Portner, D. E., Hearne, M., Flamme, H., Furtney, M., Smoczyk, G. M. "
                  "(2018). Slab2, a comprehensive subduction zone geometry model. Science, 362, 58-61. "
                  "https://doi.org/10.1126/science.aat4723")),
}
VKW_ZIP = "https://zenodo.org/api/records/14854311/files/vankeken_wilson_peps_2023.zip/content"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def row(ds, file, url, date, m):
    meta = META[ds]
    return dict(dataset=ds, version=meta["version"], doi=meta["doi"], url=url, licence=meta["licence"],
                download_date=date, file=str(file), md5=m, citation=meta["citation"])


def main():
    log = pd.read_csv(C.ROOT / "fetch_log.tsv", sep="\t", header=None,
                      names=["dataset", "file", "url", "date", "md5", "bytes"])
    rows = [row(r.dataset, C.ROOT / r.dataset / r.file, r.url, r.date, r.md5) for r in log.itertuples()]
    # files extracted locally from downloaded archives
    for ds, sub, src in (("ghfdb_r2024", "extract", "GHFBD-R2024_v.2026-03.zip"), ("globsed_v3", "extract", "GlobSed.zip")):
        for p in sorted((C.ROOT / ds / sub).glob("*")):
            d0 = log[(log.dataset == ds) & (log.file == src)].date.iloc[0]
            rows.append(row(ds, p, f"extracted from {src}", d0, md5(p)))
    # members read over HTTP byte ranges from the van Keken & Wilson zip (zip CRC-32 checked on extraction)
    for p in sorted(C.VKW.glob("*")):
        member = p.name.replace("__", "/")
        rows.append(row("vankeken_wilson_2023", p, f"{VKW_ZIP} :: {member}",
                        dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") + " local (range read)",
                        md5(p)))
    for p in sorted(C.GMRT_DIR.glob("gmrt_*.nc")):
        rows.append(row("gmrt_local", p, "https://www.gmrt.org/services/gridserverinfo.php (GridServer; box in file name)",
                        dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") + " (file mtime)",
                        md5(p)))
    p = Path("/home/mdenolle/cascadia_obs_ensemble/data/slab2/cas_slab2_dep.xyz")
    rows.append(row("slab2_local", p, "https://doi.org/10.5066/F7PV6JNV",
                    dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") + " (file mtime)", md5(p)))
    out = pd.DataFrame(rows)
    out.to_csv(C.ROOT / "PROVENANCE.csv", index=False)
    repo = Path(__file__).resolve().parents[2] / "data/external_geo"
    repo.mkdir(parents=True, exist_ok=True)
    out.to_csv(repo / "PROVENANCE.csv", index=False)
    C.log(f"wrote {C.ROOT / 'PROVENANCE.csv'} and the repository copy ({len(out)} rows)")


if __name__ == "__main__":
    main()
