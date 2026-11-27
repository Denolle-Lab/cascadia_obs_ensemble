#!/usr/bin/env bash
# Raw downloads for D-B114 (external geophysical data), as run on 2026-11-27. Each dataset goes in its own folder
# under /wd1/mdenolle_data/external_geo/; fetch.sh logs URL, date (from `date`), md5 and size to fetch_log.tsv.
# Then: eg_bathy.py -> eg_build.py -> eg_segments.py -> eg_provenance.py (pixi python, run with python -I).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); F="nice -n 10 $H/fetch.sh"
ROOT=${EXTGEO_ROOT:-/wd1/mdenolle_data/external_geo}

# 1. Heat flow: IHFC Global Heat Flow Database, Release 2024, version 2026.03 (GFZ Data Services)
G=https://datapub.gfz.de/download/10.5880.FIDGEO.2024.014-VEueRf
$F ghfdb_r2024 $G/GHFBD-R2024_v.2026-03.zip
$F ghfdb_r2024 $G/GHFDB-R2024_v.2026.03_data_description.pdf
$F ghfdb_r2024 https://dataservices.gfz.de/10.5880/fidgeo.2024.014/the-global-heat-flow-database-release-2024/metadata/datacite.json ghfdb_r2024_datacite.json
(cd "$ROOT/ghfdb_r2024" && mkdir -p extract && unzip -o -j GHFBD-R2024_v.2026-03.zip IHFC_2024_GHFDB_v.2026.03.txt -d extract)

# 2. Plate age and spreading parameters: Seton et al. (2020), EarthByte age grid 2020.1
E=https://www.earthbyte.org/webdav/ftp/earthbyte/agegrid
for f in age.2020.1.GTS2012.1m.nc age.2020.1.GTS2012.6m.nc age_misfit.2020.1.GeeK2007.6m.nc conf.2020.1.GeeK2007.6m.nc \
         full_rate.2020.1.GTS2012.6m.nc README.txt; do $F agegrid_seton2020 $E/2020/Grids/$f; done
$F agegrid_seton2020 $E/License.txt

# 3. Sediment thickness: GlobSed v3 (Straume et al. 2019), CaltechDATA mirror of the retired NOAA NCEI page
$F globsed_v3 https://data.caltech.edu/api/records/k4070-ngc79/files/GlobSed.zip/content GlobSed.zip
(cd "$ROOT/globsed_v3" && mkdir -p extract && unzip -o -j GlobSed.zip GlobSed/GlobSed_package3/GlobSed-v3.nc \
   GlobSed/GlobSed_package3/ArcGIS_ascii/GlobSed-v3.ASC.xml -d extract)

# 4. Bathymetry gaps outside GMRT: GEBCO_2026 (current release; supersedes GEBCO_2024), tile covering the area + TID
B=https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026
$F gebco_2026 $B/ice_surface_elevation/geotiff/gebco_2026_n90.0_s0.0_w-180.0_e-90.0_geotiff.tif
$F gebco_2026 $B/type_identifier_grid/geotiff/gebco_2026_tid_n90.0_s0.0_w-180.0_e-90.0_geotiff.tif
for f in 00README_catalogue_and_licence.txt 00readme.txt GEBCO_Grid_docmentation.pdf GEBCO_Grid_terms_of_use.pdf; do
  $F gebco_2026 $B/$f; done
# subset (derived, not raw): gdal_translate from the pixi environment
GT=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/gdal_translate
mkdir -p "$ROOT/derived/gebco_subset"
for f in gebco_2026 gebco_2026_tid; do
  $GT -q -projwin -134 53.5 -118 37.5 -of EHdr "$ROOT/gebco_2026/${f}_n90.0_s0.0_w-180.0_e-90.0_geotiff.tif" \
      "$ROOT/derived/gebco_subset/${f}_cascadia.bil"; done

# 5. Slab thermal models: van Keken & Wilson (2023) global suite (Syracuse et al. 2010 D80 + D80new), Zenodo v0.9.6
Z=https://zenodo.org/api/records/14854311/files
for f in Subduction_parameters.json Subduction_parameters.pdf Table2_Syracuse2010_updated.pdf \
         Table3_part_Syracuse2010_updated.pdf Figure_comp_D80_WW09_all.pdf; do $F vankeken_wilson_2023 $Z/$f/content $f; done
# the 3.4 GB archive is read over HTTP byte ranges; only the Cascadia members are written (run in two passes on
# 2026-11-27, 04_Cascadia then 03_British_Columbia; this single pattern covers both)
PY=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python
O="$ROOT/vankeken_wilson_2023/zip_members"
nice -n 10 $PY -I $H/remote_zip.py "$Z/vankeken_wilson_peps_2023.zip/content" get \
  '(README\.md$|Table[23].*xlsx$|D80(new)?/sepran/0[34]_(Cascadia|British_Columbia)/(T_040\.vtu|dtdx\.040|slabpaths\.040)$|D80new/TF/0[34]_(Cascadia|British_Columbia)/subduction_steadystate_linpicard_p2p1p2\.tfml\.run/Dc_80\.0/minres_2\.0/run_0/(slab_T\.tsv|surface_q\.tsv|subduction\.slab)$|D80new/TF/04_Cascadia/subduction_[Tq]\.png$)' "$O"

# 6. Crustal thickness (context): CRUST1.0
$F crust1 http://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz
$F crust1 https://igppweb.ucsd.edu/~gabi/crust1/crsthk.xyz.zip
$F crust1 https://igppweb.ucsd.edu/~gabi/crust1.html crust1_page.html
