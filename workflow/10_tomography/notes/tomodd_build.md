# tomoDD: provenance, build, example reproduction and efficiency audit

Date: 2026-11-22. Agent: Claude (Track B tomoDD build subagent), for D-B021 (tomoDD replaces TomoATT).
Everything large lives outside git under `/wd1/mdenolle_data/tomo_v1/`:
`src/download_*` (untouched downloads), `src/tomoDD/` (clean reference build), `src/tomoDD_opt/`
(flag and array-size variants), `runs_tomodd_example/` (example reruns), `runs_tomodd_bench/`
(Cascadia benchmark). No downloaded script or binary was executed; only code compiled here from source ran.

Short version: the only tomoDD that circulates is the USTC training package of **tomoDD-SE**
(Zhang & Thurber, spherical pseudo-bending version, WARF academic licence). Two independent channels
carry identical source; the build reproduces the shipped reference outputs byte for byte where those
outputs are consistent. tomoDD-SE ray-traces in spherical coordinates, so Earth curvature is not what
limits sub-region size. A Cascadia benchmark (Mendocino, 9,334 events, 228k absolute + 599k differential
times, 40 x 40 x 23 grid) runs 5 iterations in 9.9 min on one core with 2.5 GB peak memory after dropping
`-ffloat-store`, with output byte-identical to the as-shipped flags (17.1 min). The licence forbids
modifying the source, which rules out OpenMP and the algorithmic fixes below unless the authors agree;
running ensemble members as many serial jobs is the better strategy anyway.

---

## 1. Provenance

### 1.1 What exists

There is no public repository or release by the authors. Searches (GitHub code and repository search,
web search, Zenodo, the USTC/UW-Madison pages cited by the tutorial) found only the USTC training
packages ("USTC-tomoDD-SE-training2017" and "...training2020") and derivatives of them. The original
Cartesian tomoDD (2003) and tomoFDD are not public anywhere I could find; the manual in the package
refers to tomoFDD for large regions, and the shipped code is tomoDD-SE, which is the spherical one
(header: "tomoDD-SE: Double-difference tomography method for regional scale tomography. This version uses
the spherical pseudo-bending method").

| Copy | Source | Retrieved | Archive / commit | Content hash (all `*.f *.c` of `Src/tomoDD-SE`, sorted, sha256) | Notes |
|---|---|---|---|---|---|
| A (used) | CUHK seismology tutorial "Body Wave Tomography", https://cuseistut.readthedocs.io/en/latest/body_tomo/index.html, download link `_downloads/bddb170e1a6c5e0e4c9dea36044d36cc/USTC-tomoDD-SE-training2020.tar` | 2026-11-22 | tar 56,350,720 B, md5 `0c563be463073bffb9049ce702e12b1f`, sha256 `e618c6914d328c7dd9752bd105aa128fec051750144b8f1f17ea6353ee275fe1`; tar entries dated 2022-03-20, owner `xinrumu` | `771dfdd3a53623d8...` | Page thanks "Prof. Zhang and his team for sharing the TomoDD code". Contains Doc/ (tomoDD-SE manual, Zhang & Thurber 2003 BSSA and 2006 PAGEOPH PDFs), Example/, Scripts/, Src/ (tomoDD-SE, ph2dt). |
| B | GitHub `travisalongi/tomoDD-SE-training2020`, commit `7c4579a166ead4655f2d00e4529aaf6a8f3ce560` (2026-03-19, "upload tomoDD") | 2026-11-22 | git clone | `771dfdd3a53623d8...` (identical to A) | Third-party mirror. Source identical to A; differs only in `.o` files, compiled binaries, `.DS_Store`, one example `event.dat`. Probably a re-upload of the same tar, so not fully independent of A. |
| C | GitHub `CovMat/EBooks-of-Geophysics`, folder `USTC-tomoDD-SE-training2017/`, commit `87cd4587be179666c09b33d6a5eec91ff74c0097` | 2026-11-22 | sparse git checkout | `d06e0302759ba1e1...` | Independent earlier release (2017 training). Same code base; the 2020 copy adds: longitude wrap fixes in `RaySPDR2.f` ("added by lupan", 2019, needed for longitudes < 0, i.e. Cascadia), a `tomoDD-SE.res.ini` initial-residual output, per-iteration DWS output, a fixed `ttmder` argument list (2017 calls it with 6 arguments, the routine takes 7), log name `tomoDD-SE.log`, example array sizes. |
| D (not used) | GitHub `yhao-geo/tomoDD_SP_tech` (commit `41a9a66a`, 2025-08-19) | 2026-11-22 | clone | n/a | Teaching fork "tomoDD-SP" (S-P times). Same `lsqr.f`, modified `RaySPDR2.f` (138 changed lines) and main program; "parallel" in the folder name but no OpenMP/MPI in the code. No licence file. |

Other hits are not code: `Simon0413/tomoDD_tools` (Python helpers), `Luojiahong/tomoDD_TGR` (paper
data), phase-format converters. hypoDD (github.com/fwaldhauser/HypoDD, v2.1b) is the parent of the
location part only.

Version strings in the source: `starting tomoDD-SE (v1.0 - 01/2003)` (inherited banner, not
meaningful), header "All rights reserved by (c) Haijiang Zhang and Clifford Thurber, 2005-2015",
WARF notice "(c)2004-2009 Wisconsin Alumni Research Foundation" / "WARF 7/09"; manual dated
September 2015 (Zhang & Thurber, UW-Madison / USTC). The shipped binaries were built with
"GCC: (Ubuntu 5.4.0-6ubuntu1~16.04.11)" (from `strings`; not executed).

Per-file sha256 of the build tree: `/wd1/mdenolle_data/tomo_v1/src/tomoDD/SOURCE_SHA256.txt` (63 files).

### 1.2 Licence

`COPYRIGHT.txt` and the program header carry the WARF "Software License Agreement, Double-Difference
Seismic Tomography, for academic research use only" (US Copyright No. 6,813,639). Key terms:
non-commercial academic use only; licence is personal and non-transferable; **no modification, translation
or derivative works**; **no public display or posting** of the software; keep the copyright notices.
Consequences for us:
* Use for this project is allowed (academic research).
* Do not put the source or binaries in our git repo or on Zenodo (private draft included); cite
  Zhang & Thurber (2003, 2006) and state the version (USTC training 2020 package, content hash above).
* Editing array dimensions in `include/*.inc` is what the authors' manual tells users to do, so I treat
  it as permitted. Compiler flags are not modifications. Anything else (OpenMP directives, fixing the
  bugs listed in section 5) needs permission from Haijiang Zhang (zhang11@ustc.edu.cn, the address in the
  header) or WARF. I made no source edits other than the `.inc` sizes.

### 1.3 Trust verdict

Copy A (= B) is a third-party redistribution, not an author release, but it is traceable to the authors:
it is the USTC training package that the tutorial says Zhang's group shared, it carries the authors'
manual, papers and licence, and an independent older USTC package (C) has the same code apart from
documented, attributed fixes. The example outputs reproduce exactly (section 3). I recommend using A,
and asking Haijiang Zhang to confirm it is the current version (and to grant the modification
permission, section 6). If he sends a newer copy, re-run the checks in section 3 on it.

---

## 2. Build recipe

Compilers: system GCC 8.5.0 (`gfortran --version`: GNU Fortran (GCC) 8.5.0 20210514 (Red Hat 8.5.0-28)).
Host: 2 x Intel Xeon Gold 6448Y, 128 hardware threads, 250 GB RAM.

```bash
# clean copy of only the files the Makefiles list (no .o, no shipped binaries, no DARWIN .a)
SRC=/wd1/mdenolle_data/tomo_v1/src/download_cuseistut/USTC-tomoDD-SE-training2020/Src/tomoDD-SE
T=/wd1/mdenolle_data/tomo_v1/src/tomoDD/src; mkdir -p $T/include
cp -p $SRC/{tomoDD-SE.f,tomoDD-SE_syn.f,syn_time_new2.f,Makefile,Makefile_syn,COPYRIGHT.txt,geometry.h} $T/
cp -p $SRC/<the 53 .f/.c files in SRCS/CSRCS of Makefile> $T/
cp -p $SRC/include/{tomoFDD.inc,RaySPDR.inc,geocoord.inc,compat.h,f77types.h,matvec6.h} $T/include/
cd $T && make && make -f Makefile_syn        # -> tomoDD-SE, tomoDD-SE_syn
```

Notes:
* `geometry.h` is not listed in the Makefile but is needed by `transform_r_gfortran.c`.
* The Makefile has no dependency on the include files: after editing `include/*.inc` you must
  `make clean` first, or main is not rebuilt (I hit this).
* As-shipped flags: `FFLAGS = -O3 -ffixed-line-length-none -ffloat-store -W -m64 -mcmodel=medium`,
  `CFLAGS = -O3 -m64 -mcmodel=medium`. Builds with compiler warnings (unused variables, an `#endif` label), no errors.
* gfortran 13.4 (conda-forge, `pixi exec --spec gfortran=13 --spec gcc=13`) builds, and the forward code
  gives byte-identical synthetic times, but the inversion stops at the first iteration with a runtime
  "Missing comma between descriptors" in the FORMAT at `tomoDD-SE.f` line 2320 (`f7.3")"`). gfortran 8.5
  tolerates it. Use gfortran 8.5 (fixing the FORMAT is a source edit, see 1.2).
* Static-array limit: gfortran puts the main program's arrays in `.bss`; the link fails ("relocation
  truncated to fit") once they exceed 2 GB. The largest is `col(MAXND*MAXDATA)` (4 bytes each), so
  `MAXND*MAXDATA` must stay below ~3.5e8 (MAXDATA = 3e6 with MAXND = 100 links; MAXND = 200 does not).

---

## 3. Example reproduction

### 3.1 RealData-Inversion (Sichuan, 322 events, 13 of 60 stations, 22,140 data)

Run: `runs_tomodd_example/ref_O3/` with the shipped `MOD`, `tomoDD-SE.inp` and `Input_Files`.
10.3 s, 110 MB.

* `tomoDD-SE.log`: the shipped log is 1,261 lines (it stops in iteration 5 (10), a truncated, unflushed
  copy); our first 1,261 lines match it line for line in every number. The only differences are the
  "( x %)" change columns of the residual summary, which print an uninitialised/zero denominator
  (`Inf` here, `*******` or random values in the shipped log), and garbage bytes after timestamps.
* `Output_Files/tomoDD.loc` and `tomoDD.reloc.001.001` ... `.005` are byte-identical (md5) to the shipped ones.
* `Output_Files/tomoDD.vel`: the shipped file (2,064,719 bytes) is a byte-exact prefix of ours (truncated
  like the log).
* The shipped `tomoDD.reloc.001.006`-`.010`, `tomoDD.reloc`, `tomoDD.res`, `tomoDD.sta`, `Vp_model.dat`
  and `Vs_model.dat` come from a different run: 303 events at iteration 6 where the logged run has 194,
  18,944 residual lines vs the logged run's 8,581 data, and `Vp_model.dat` has 154 lines (a 12 x 14 x 11
  grid) while the shipped `MOD` is 29 x 31 x 19. Running with the shipped `MOD_old` (12 x 14 x 11) does not
  reproduce them either. They cannot be used as a reference.
* **The shipped example is broken as a tomography test.** Its `MOD` has longitudes in the latitude row
  (`yn = 28.7 104.36 ... 104.59 30.1`), so all 322 events are "outside the computation domain", every
  ray has zero velocity derivatives ("# of nonzero model derivatives before smoothing 0") and the run is a
  relocation only. The tutorial users apparently did not notice. With `MOD_old` the velocity inversion
  works (481,531 non-zero derivatives, absolute RMS 1.78 -> 0.40 s over 17 iteration passes including air-quake repeats).

Determinism: two -O3 runs, an -O0 build and an -O1 `-fcheck=all` build all give byte-identical outputs
(`MOD_old` case). The bounds-checked build runs clean on the `MOD_old` case; on the broken `MOD` it
stops at an out-of-bounds read in `intmap` called from `ttmder` (event outside grid), a symptom of the bad
input.

### 3.2 Forward synthetics (checkerboard example)

`tomoDD-SE_syn` on the shipped `checkerboard/Syn` inputs (1,409 events, 27,672 data, 12 x 14 x 11 model):
`absolute.syn` (502,440 B), `dt.syn` (1,717,522 B), `tomoDD.loc`, `tomoDD.vel` are **byte-identical** to
the shipped files (made in 2021 with GCC 5.4 on Ubuntu). This checks the pseudo-bending ray tracer and
travel-time output on 8 years and 2 compiler generations. The gfortran 13 build gives the same bytes.

Verdict: the build reproduces the package. The velocity-inversion path is checked only through
internal consistency (MOD_old runs, determinism, bounds check), because the package has no valid
reference output for it.

---

## 4. Audit: limits, memory, profile

### 4.1 Compile-time limits (`include/tomoFDD.inc`, `include/RaySPDR.inc`)

| Parameter | Shipped | Meaning (from code) | Checked at run time? | Proposed for Cascadia sub-regions |
|---|---|---|---|---|
| MAXEVE | 1650 | events | yes (stop) | 10000 |
| MAXSTA | 80 | stations in station file within DIST | yes | 300 |
| MAXOBS | 80 | **stations per event** (rays stored per event-station pair) | **no**: `add_sta` writes past the end silently | = MAXSTA (300), or >= max stations per event + margin |
| MAXDATA | 40000 | absolute + differential data (all phases) | yes | 3,000,000 |
| MAXND | 80 | mean non-zero model derivatives per datum (G has at most MAXND*MAXDATA non-zeros) | yes | 100 (link limit, section 2) |
| MAXNODE | 300 | non-zero velocity derivatives per ray | **no**: `tmp_vp_index(k,j,n+1)` overflows silently | 400 (verify with a `-fcheck=bounds` run per new grid) |
| maxnx, maxny, maxnz | 90, 80, 50 | grid nodes | yes | 90, 80, 50 (enough for 40 x 40 x 25) |
| mxpari, maxpar | 500,000 | inversion parameters | yes | unchanged |
| ixkms/iykms/izkms | 200,000 | index maps, (range)/bld | yes | unchanged; use bld = 0.01 (0.001 overflows izkms for a 300 km deep bottom node) |
| amcusp (fixed) | 1000 | air quakes per iteration | yes (stop) | n/a |
| station label | 7 chars | `character*7` | truncation | our names reach 8 chars: write numeric codes (S0001...) |

A `-fcheck=bounds` build (`tomoDD_opt/chk_big`) ran one Mendocino iteration with these sizes without
a bounds error and with output identical to the optimised build (132 s), so MAXOBS 100 and MAXNODE 400
are enough for that region and grid.

Memory per run, measured (Mendocino benchmark, MAXEVE 10,000, MAXOBS 100, MAXDATA 3e6, MAXND 100,
MAXNODE 400): **peak RSS 2.50 GB** (virtual 12.7 GB; Linux only backs touched pages). Main parts:
the per-ray derivative cache `tmp_vp/tmp_vs` (+ index arrays), 4 arrays of MAXOBS x MAXEVE x MAXNODE
(6.4 GB virtual here) of which only the planes up to the longest ray's node count are touched; the sparse
G (32.7 M non-zeros x ~16 B = 0.5 GB); and the MAXDATA-sized vectors. Because the cache scales with
MAXOBS x MAXEVE, set those two tight per sub-region (one build per region size class) rather than to
the global maximum. Extrapolation for the largest sub-region (Puget, 7,869 events, 314k picks, 109
stations per event max, ~1.5-2 M differential times): 3-5 GB per run (estimate, not measured).

### 4.2 Where the time goes

Example (`MOD_old`, gprof, -O3): ray tracing 81% (`vel3` 42%, `cellvel` 19%, `pbr` 17%, `vmodel` 2%,
`ttmder` 1%), LSQR `aprod` 11%, main program 5%; 31,484 rays in 9.9 s = 0.31 ms per ray.

Cascadia benchmark (Mendocino box 39.5-43.1 N, 127.5-122.5 W; 9,334 events, 153 stations of which 111
OBS, 228,454 absolute P+S, 599,440 catalogue differential times from <= 10 neighbours within 10 km and >= 8
common picks; 40 x 40 x 23 nodes, 60,648 Vp+Vs parameters, 17,027 above the DWS threshold; 2 weight sets,
5 iterations + 1 air-quake repeat; inputs built by `make_bench.py`/`make_mod.py` (scratch, benchmark
only, simplified weights; not production inputs). Fit: absolute RMS 0.75 -> 0.22 s, weighted 2.16 -> 0.20 s.

| Stage per iteration | as shipped (-O3 -ffloat-store) | -O3 without -ffloat-store |
|---|---|---|
| ray tracing + derivatives (228k rays) | 82-112 s | 52-70 s |
| reweighting + G assembly | 49 s | 33 s |
| LSQR | 25-51 s | 8-19 s |
| whole run (5 + 1 iterations) | **17.1 min** (1,019 s CPU) | **9.9 min** (593 s CPU) |

A sampling profile of the as-shipped run (40 stack samples over ~6 min) gives ray tracing 45%, LSQR
23%, main-program loops (G assembly) 28%. The G-assembly share comes from one loop pattern: for every
differential datum the code zeroes and scans two dense vectors of length npari (`temp1/temp2`, main
program lines ~1430-1525), i.e. O(ndt x npari) = 6e5 x 6e4 per iteration here, and the same dense
pattern per ray in `ttmder` and the derivative compaction. It grows with grid size times data count.

Rays are traced once per event-station pair per iteration (the `eve_sta` table), and differential data
reuse the cached derivatives; there is no recomputation per pair to remove.

### 4.3 Efficiency options, ranked by gain/risk

1. **Drop `-ffloat-store`** (flag only). Byte-identical outputs on the example (all files) and on the
   Cascadia benchmark (`tomoDD.reloc`, `Vp_model.dat`, `Vs_model.dat`, `tomoDD.res`, `tomoDD.vel`,
   `tomoDD.sta`); 1.72x faster (17.1 -> 9.9 min; example 19.6 -> 11.2 s). On x86-64 SSE math the flag only
   forces extra stores. -O2 without it: 12.7 s on the example, identical output. **Adopt.**
2. **Run ensemble members as independent serial jobs** (no code change). At 2-5 GB per run, 128 threads
   and 250 GB, 40-60 concurrent runs fit; 64 concurrent runs gave 34x the serial throughput (4.4).
   This makes per-run parallelism unnecessary for the 100-member ensemble.
3. **Do not use `-march=native`.** It enables FMA contraction, which changes results well beyond round-off
   (MOD_old example: median hypocentre shift 0.14 km horizontal and 0.23 km vertical, maximum 5.4 / 7.7 km;
   Vp nodes up to 0.31 km/s, Vs 0.14 km/s), and it was slower here (19-54 s vs 11-20 s). With
   `-ffp-contract=off` the output is identical again but the run took 47 s. The size of the FMA effect is
   itself a finding: the iteratively reweighted, DWS-thresholded inversion amplifies last-bit differences to
   ~0.1-0.3 km and ~0.1-0.3 km/s at poorly constrained nodes and events. Pin compiler, flags and machine
   for all ensemble members; numerical noise of this size sits inside the ensemble spread we will report.
4. **Size arrays per region** (`.inc` edit, allowed): tight MAXOBS and MAXEVE cut the derivative cache,
   the main memory term, roughly in proportion.
5. Source changes (need licence permission, not done): (a) replace the dense `temp1/temp2` scans with a
   sparse merge of the two cached derivative lists: removes most of the 28% G-assembly time and its
   O(ndt x npari) growth; (b) OpenMP over the event-station ray loop: the rays are independent (each writes
   its own `tmp_*(k,j)` slots) but `pbr`/`ttmder`/`vel3` share state through COMMON blocks (`/rpath/`,
   `/weight/`, `/coord/`, `dtm` in `/modinv/`) and large local arrays that gfortran makes static, so they
   need `THREADPRIVATE`/`-frecursive`; expected near-linear speed-up of the 45-80% ray share, low-moderate
   risk; (c) reorder `tmp_vp(MAXOBS,MAXEVE,MAXNODE)` to node-fastest to cut touched memory. With serial
   ensemble runs, (b) only matters for a single large exploratory run.

### 4.4 Many serial runs at once

Test: N copies of a 1-iteration Mendocino run (same inputs, `O3nofs_big` binary) started at once
(`runs_tomodd_bench/conc.sh`).

| N concurrent | mean wall per run | slowest | peak RSS per run | runs per hour |
|---|---|---|---|---|
| 1 | 102 s | 102 s | 2.44 GB | 35 |
| 32 | 125 s | 128 s | 2.03 GB | 900 |
| 64 | 169 s | 193 s | 2.01 GB | 1,190 |

Outputs of all concurrent runs are byte-identical to the single run. 32 jobs cost 1.2x per run, 64 jobs
1.65x (memory bandwidth / hyper-threads), so 64 at once gives ~34x the serial throughput and 32 at once
~26x. Memory, not cores, sets the limit once runs reach ~4 GB (60 x 4 GB = 240 GB). For 100 members x 10
iterations of the Mendocino-size problem: ~100 x 10 x 2 min x 1.65 / 64 = ~50 min wall.

---

## 5. Correctness and fitness for our problem

* **Geometry.** Ray tracing is spherical pseudo-bending (Um & Thurber 1987 bending in spherical coordinates,
  Earth radius 6371 km, no ellipticity: the 0.99664719 factor is commented out). Locations are updated
  in an exact 3-D Cartesian frame tangent at (wlat, wlon) (`transform_r_gfortran.c`, `sph2car`/`car2sph`),
  not a flat-earth projection. The model grid is in longitude, latitude, depth (km). So Earth curvature
  does not limit sub-region size; D-B021's ~400 km limit is now a memory/run-time choice, not a geometry one.
  Single precision coordinates give ~0.4 m resolution at 400 km from the centre (fine).
* **Longitudes < 0.** Handled only in the 2020 copy (lupan fixes in `cellvel`, `ttmder`, main). The 2017
  copy maps longitudes to 0-360 in `cellvel` and would put every Cascadia ray outside a grid given in
  negative longitudes. Benchmark with negative longitudes: no "outside the domain" warnings.
* **Grid edges silently drop derivatives.** `ttmder` stops accumulating a ray's velocity derivatives
  (`goto 1717`) at the first point within 0.1 deg of a lateral edge node or within 1.0 km of the top or
  bottom edge node. `cellvel` uses the 3-D model only inside the same 0.1 deg margin and above
  zn(nz) - 1 km, and a global layered model (ak135 + `layer-16.dat`) outside. Rules for our MOD: edge nodes
  far outside the data (I used 8 deg); top edge node at least 1 km above the highest station (highest
  station 2.19 km, so zn(1) <= -4 km; I used -6); bottom edge well below the deepest ray.
* **Station elevation and seafloor stations.** `station.dat` takes elevation in metres, + above sea level;
  the code converts to depth (km, + down). OBS at -4.46 km elevation sit at 4.46 km depth: correct. The
  ray geometry uses the true elevation, but `cellvel` clamps depth to >= 0, so land stations above sea level
  get the sea-level velocity along the part of the path above sea level. The MOD has no topography or water:
  offshore nodes shallower than the seafloor must hold rock velocities (as in Track A), otherwise rays to
  OBS interpolate water speeds. Events above `Air_dep` are "air quakes"; any air quake makes the whole
  iteration's update (locations and velocities) be discarded and the iteration repeated with that event
  down-weighted (benchmark: 1 repeat). Set `Air_dep` to the highest station elevation (about -2.2 km), not
  0, or offshore events near the seafloor and land events above sea level trigger repeats.
* **Weights.** `absolute.dat` and `dt.ct` take a per-pick weight in the 4th/5th column, **but only in
  [0, 3]**: values outside are dropped with a console warning ("Weight quality format not Correct"), not
  clipped. Our `w_bal` reaches 2.54 and multiplied by the D-B008 quality weight could exceed 3, so
  rescale to max <= 3 when writing files. Effective weight = a-priori pick weight x WTCTP/WTCTS (per
  iteration set) x WTCD for absolute data (absolute data are rows with id1 = id2), then, if WRCT > 0,
  a bi-cube taper (1 - (|r|/c)^3)^3 with c = WRCT x MAD of the residuals (separate MADs for absolute and
  differential data when WRCT >= 1; absolute seconds when < 1), and a distance taper for event separation
  if WDCT is set. So the a-priori weights multiply the dynamic reweighting; they are not replaced.
  Units bug in the WDCT taper: in the first iteration the event separation `dt_offs` is in km (from
  `getdata`) but the taper divides by `WDCT*1000`, so the taper is nearly inactive; from the end of
  iteration 1 `dt_offs` is recomputed in metres and the taper works as intended.
* **No station terms.** There are no station corrections or statics in tomoDD-SE (the `RaySPDR.inc`
  comments about them are simul2000 leftovers). OBS statics (D-B020) must be applied to the picks before
  writing `absolute.dat`; differential times cancel a constant static. Origin times are solved per event.
* **Hard-coded limits in the update.** Per iteration, hypocentre shifts are clipped to 2 km horizontally
  and 3 km vertically; velocity changes to 0.4 km/s (P) and 0.2 km/s (S); Vp is kept in [1.0, 8.7] and Vs in
  [0.8, 4.9] km/s. The MOD gives Vp/Vs, but the inversion updates Vs (slowness) directly. Smoothing rows
  are first differences between neighbouring nodes with the user weight, not divided by node spacing, so
  an uneven grid gets uneven smoothing.
* **Thresholds.** Trilinear weights < 0.05 are not counted in derivatives; DD derivative differences <=
  0.01 (km of ray length) are dropped; nodes with DWS below THRES (S: THRES/3) are fixed. These make the
  result depend on round-off at the margin (section 4.3, item 3).
* **pbr settings** (hard-coded in `RaySPDR2.f`): convergence 1e-5 relative, segment doubling stops when
  segments are < 10 km (`mins = 10`), at most 10,000 segments; derivative integration step is `stepl`
  (inp file; 1 km in the examples and benchmark). A 10 km minimum segment is coarse for paths of 20-50 km
  in a strongly heterogeneous shallow model; check travel-time accuracy against a finer reference (as the
  TomoATT plan proposed) before production. Changing `mins` is a source edit.
* **Minor code bugs seen**, harmless for us: `if(y.LT.0) b2 = ...` in `pbr` uses a stale `y` (irrelevant
  because longitudes are rotated into 0-180 first); `upz = r0` (undefined, = 0) in a branch reached only near
  the Earth's centre; the "%" change fields in the residual summary are garbage on the first pass.

---

## 6. Proposed decisions

* **D-B024 (proposal) tomoDD copy.** Use tomoDD-SE from the USTC training 2020 package (copy A,
  tar sha256 `e618c691...`, source hash `771dfdd3...`), built with gfortran 8.5. Keep source and
  binaries out of git and Zenodo (WARF licence). Write to Haijiang Zhang to (a) confirm it is the
  current version, (b) ask permission to modify for our use (sparse G assembly, OpenMP, FORMAT fix,
  bound checks).
* **Flags.** `-O3 -ffixed-line-length-none -W -m64 -mcmodel=medium` (as shipped minus `-ffloat-store`):
  1.72x faster, byte-identical. No `-march=native`. All ensemble members on the same binary and host type.
* **Array sizes** (one build per region class): MAXEVE 10000, MAXSTA 300, MAXOBS = max stations per event
  in the region + 20%, MAXDATA 3e6, MAXND 100, MAXNODE 400, grid maxima unchanged. Each new grid: one
  `-fcheck=bounds` run (MAXOBS, MAXNODE overflow silently otherwise).
* **Inputs.** Numeric station codes (<= 7 chars); weights rescaled to [0, 3]; OBS statics applied to the
  picks; MOD in lon/lat/depth with bld 0.01, far edge nodes, top edge <= -4 km, rock velocities at all
  nodes, Air_dep ~ -2.2 km.
* **Parallel strategy.** Run the ~100 Bayesian ensemble members per region as independent serial jobs
  (section 4.4 for the concurrency); no OpenMP. Estimated cost for a large sub-region (~2 M differential
  times, 10 iterations): 0.5-1 h and 3-5 GB per member when run alone; with 50-60 members at once
  (1.5-2x slowdown each) 100 members take ~2-4 h wall per region on this host (estimate from the
  benchmark, to be checked on real inputs).
* **Before production**: travel-time accuracy test of pbr (mins = 10 km) against a fine reference for
  OBS and land paths; checkerboard with `tomoDD-SE_syn`; quantify numerical-noise spread (e.g. two builds
  differing only in FMA) as a floor for the ensemble spread.
