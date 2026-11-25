# 2016 offshore events missing from catalog v4: repick, association, location

**Events.** The 8 December 2016 14:49:45.91 UTC Mw 6.6 on the Mendocino transform (ComCat `us20007z6r`),
its M 4.7 aftershock at 16:32:46 (`us20007z7b`, v4 event 172114 with 9 P / 0 S picks), and the Gorda plate
pair of 30 January 2016, M 5.1 at 00:28:41.96 (`us20004vq3`) and M 5.0 at 00:30:52.57 (`us20004vq9`).

**Why v4 missed them.** No `NC`/`BK` picks exist in 2016 (the 2016 northern California pick gap); the ver6
picks of that day near 14:50 come from the Axial and Hydrate Ridge cabled stations and a few `CC`/`NV`
stations 300-800 km away. GENIE associated four of them into v4 event **172113** (14:51:18, 4 P + 1 S,
located near Mount St. Helens): its P picks at `PB.B203`, `PB.B202`, `UW.PCFR` and `UW.GHW` are the Mw 6.6 Pn
arrivals at 708-797 km (0.6-1.9 s from iasp91), and its "S" at `PB.B941` is P coda. **172113 is spurious and
should be flagged.**

**Method** (scripts in `workflow/m66_2016/`, large files in `/wd1/mdenolle_data/m66_2016/`).
1. `m1_download.py`: 30-min windows from EarthScope and NCEDC FDSN for all stations within 4 deg
   (Mw 6.6 window: 279 stations returned, 193 `NC`/`BK`; `stations_requested_<window>.csv` lists requested vs
   returned). Every station is east of the events; the nearest is 156 km away.
2. `m2_pick.py`: ELEP ensemble (same models, semblance, 4-15 Hz, 100 Hz, 0.05 thresholds as
   `workflow/01_picking/elep_picker.py`); Z-only stations get Z copied to three inputs; HN-only stations are
   picked on HN (flagged).
3. `m4_associate.py`: per station, the highest-probability pick within P -3..+3 s + 1% of distance and
   S -5..+5 s + 4% of distance of the TauP iasp91 prediction from the ComCat hypocentre, stations to 400 km.
4. `m3_grids.sh`: travel-time grids from the Track A 3D model (read-only) cut +-500 km around each station
   (Track A's +-250 km grids do not reach these events), and iasp91 1D grids. The 3D model ends near
   38.6 N, so 30 southern stations have only 1D grids.
5. `m5_locate.py`: NonLinLoc, Track A settings (EDT_OT_WT, LOCGAU2), search box ComCat +-60 km; models 3D
   with iter1 station terms, 3D without terms, 1D iasp91; depth free (-5 to 45 km) and bounded 0-20 km.
6. `m6_outputs.py`: patch rows, picks in v4 format, figures.

**Picks.**

| Event | P | S | stations | distance range (km) |
|---|---|---|---|---|
| Mw 6.6 | 146 | 54 | 148 | 156-400 |
| M 4.7 aftershock | 105 | 57 | 108 | 171-400 |
| Gorda M 5.1 | 42 | 29 | 45 | 247-397 |
| Gorda M 5.0 | 11 | 1 | 12 | 294-398 (arrives in the coda of the M 5.1) |

**Locations** (`nll_locations.csv`; "REJECTED" = maximum on the grid boundary, here always the depth limit):

| event | model | status | lat | lon | depth | rms | nphs | gap | dist_to_comcat_km | dt_origin_s |
|---|---|---|---|---|---|---|---|---|---|---|
| mw66 | 3d_trackA_iter1 | REJECTED | 40.34 | -126.66 | 44.84 | 0.86 | 159 | 265.75 | 41.41 | -3.59 |
| mw66 | 3d_trackA_noterms | REJECTED | 40.34 | -126.65 | 44.84 | 0.88 | 159 | 265.57 | 40.58 | -3.49 |
| mw66 | 1d_iasp91 | LOCATED | 40.39 | -126.49 | 34.06 | 0.91 | 200 | 257.06 | 26.22 | -0.84 |
| mw66 | 3d_trackA_iter1_z0-20 | REJECTED | 40.33 | -126.66 | 19.94 | 0.88 | 159 | 265.79 | 41.59 | -3.74 |
| mw66 | 3d_trackA_noterms_z0-20 | REJECTED | 40.33 | -126.65 | 19.94 | 0.9 | 159 | 265.61 | 40.76 | -3.65 |
| mw66 | 1d_iasp91_z0-20 | REJECTED | 40.4 | -126.41 | 19.94 | 0.89 | 200 | 255.62 | 19.78 | -1.38 |
| m47 | 3d_trackA_iter1 | LOCATED | 40.31 | -126.46 | 44.53 | 0.87 | 145 | 264.71 | 15.16 | 2.06 |
| m47 | 3d_trackA_noterms | LOCATED | 40.31 | -126.45 | 41.72 | 0.91 | 145 | 264.5 | 14.47 | 2.19 |
| m47 | 1d_iasp91 | LOCATED | 40.37 | -126.34 | 35.16 | 0.88 | 162 | 255.95 | 6.73 | 4.19 |
| m47 | 3d_trackA_iter1_z0-20 | REJECTED | 40.3 | -126.46 | 19.94 | 0.9 | 145 | 264.91 | 16.09 | 1.9 |
| m47 | 3d_trackA_noterms_z0-20 | REJECTED | 40.31 | -126.45 | 19.94 | 0.93 | 145 | 264.64 | 15.14 | 2.02 |
| m47 | 1d_iasp91_z0-20 | REJECTED | 40.37 | -126.26 | 19.94 | 0.9 | 162 | 254.18 | 11.36 | 3.66 |
| gorda1 | 3d_trackA_iter1 | REJECTED | 40.85 | -127.35 | 44.69 | 0.77 | 71 | 286.27 | 13.81 | -2.11 |
| gorda1 | 3d_trackA_noterms | REJECTED | 40.86 | -127.33 | 44.69 | 0.79 | 71 | 285.98 | 11.77 | -1.96 |
| gorda1 | 1d_iasp91 | LOCATED | 40.84 | -127.25 | 35.31 | 0.67 | 71 | 284.86 | 9.44 | 0.05 |
| gorda1 | 3d_trackA_iter1_z0-20 | REJECTED | 40.82 | -127.34 | 19.88 | 0.78 | 71 | 286.26 | 15.52 | -2.2 |
| gorda1 | 3d_trackA_noterms_z0-20 | REJECTED | 40.83 | -127.33 | 19.88 | 0.81 | 71 | 286.04 | 13.93 | -2.07 |
| gorda1 | 1d_iasp91_z0-20 | REJECTED | 40.84 | -127.16 | 19.88 | 0.68 | 71 | 283.38 | 9.06 | -0.4 |
| gorda2 | 3d_trackA_iter1 | REJECTED | 41.04 | -127.21 | 44.38 | 0.46 | 12 | 309.02 | 14.83 | -0.35 |
| gorda2 | 3d_trackA_noterms | LOCATED | 41.06 | -127.11 | 2.5 | 0.37 | 12 | 307.81 | 22.77 | -0.64 |
| gorda2 | 1d_iasp91 | REJECTED | 41.06 | -127.15 | 44.38 | 0.4 | 12 | 308.16 | 20.2 | 2.13 |
| gorda2 | 3d_trackA_iter1_z0-20 | LOCATED | 41.14 | -126.9 | 3.25 | 0.5 | 12 | 304.5 | 42.39 | 2.09 |
| gorda2 | 3d_trackA_noterms_z0-20 | LOCATED | 41.14 | -126.9 | 3.25 | 0.45 | 12 | 304.5 | 42.39 | 2.01 |
| gorda2 | 1d_iasp91_z0-20 | REJECTED | 41.09 | -127.03 | 19.75 | 0.4 | 12 | 306.57 | 30.28 | 1.97 |

**Result.** The events are detected and well picked (`fig_m66_record_section.png`), but they cannot be
located better than ComCat with this geometry: all picks are Pn/Sn beyond 156 km, the azimuthal gap is
255-310 deg, and there is no station to the west. Depth is unresolved (every solution goes to the deepest
depth allowed, trading off against origin time), and epicentres move west of ComCat by 7-16 km (M 4.7),
9-15 km (M 5.1), 20-26 km (Mw 6.6, 1D) to 41 km (Mw 6.6, 3D) (`fig_m66_locations.png`).

**Patch** (does not modify catalog v4): `catalog_v4_patch_2016.csv` adds the Mw 6.6 and the Gorda pair as
IDs 900001-900003 with the **ComCat hypocentre and magnitude** and our pick counts (`spatial_sigma(m)` = distance
from our 1D epicentre to ComCat, `location_source` says so); `picks_v4_patch_2016.csv` holds the 445
associated picks in v4 format for these three and for v4 event 172114 (amplitudes not measured).

**Caveats.** Picks are matched to known origins, not blindly associated, so a wrong pick within the window
is possible (one P and one S per station, highest probability). HN-only and Z-only stations are picked
outside the standard channel rule. The 3D model south of 40 N offshore is the 1D young-ocean fill.
