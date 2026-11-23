# Literature match: Puget Lowland and offshore Washington fault surfaces (catalog v4, 2010–2016)

Input: `faults_puget_wa.csv` (25 PU surfaces, 5 WA surfaces). Strikes use the right-hand rule. "Strike diff" compares the surface's trend (strike mod 180) with the trend of the mapped trace. Dip directions are written out because several surfaces dip the opposite way from the fault they sit on.

CFM traces were checked directly in the local CRESCENT CFM file (`scratchpad/cfm/traces.geojson`). That file includes WGS (2025) faults that the `nearest_cfm` column does not list, namely Rattlesnake Mountain, Cherry Creek, Tokul Creek and Dabob Bay. So `nearest_cfm` is not always the closest fault. For example, PU019 sits on the Rattlesnake Mountain trace, but the column names SWIF (north).

## (a) Overview

**What the set contains.** All 30 surfaces are crustal. The deepest bottom edge is 28.6 km, and `slab2_minus_depth` ranges from +3.6 to +69 km. **None of the surfaces is intraslab.** The deep Juan de Fuca Wadati–Benioff seismicity under Puget Sound, which hosted the 1949, 1965 and 2001 Nisqually events at 50–60 km depth (Ichinose et al., 2004; McCrory et al., 2012), produced no surface in this group. The St. Helens Seismic Zone (Weaver & Smith, 1983) lies south of the Puget box, so it belongs to the onshore group. No megathrust surface was found offshore Washington. The only surface near the slab top is WA005, and it is steep.

**Strong matches.**
- PU011 matches the NNE-dipping seismic zone that Li et al. (2018) relocated beneath the eastern 30 km of the Leech River fault (LRF), which extends about 20 km east offshore. PU011 covers the same along-strike extent and dips 44° N. The CFM gives the LRF a 52° N dip, and the 2016 AGU abstract by the same group gave about 45°. Strike difference is about 9°.
- PU022 is the volcano-tectonic cluster beneath the Mount Rainier summit (Moran et al., 2000).
- PU019 lies on the Rattlesnake Mountain fault zone, at the SE end of the SWIF system (Sherrod et al., 2008), and runs to the 1996 M5.3 Duvall source area.

**Medium matches.**
- PU003 is the Western Rainier Seismic Zone (WRSZ). Stanley et al. (1996) describe it as a north-trending band west of the park, about 65 km long, with en echelon oblique faults.
- PU016 is at the SE end of the Olympia structure, an 80 km long, 315°-trending potential-field lineament (Clement et al., 2010). Strike agrees within about 5°.
- PU025 is on the SWIF main strand at Possession Sound (Johnson et al., 1996).
- PU018 and PU015 are E–W and north-dipping, along the inferred eastern continuation of the Tacoma fault. Johnson et al. (2004) describe the Tacoma fault as a north-dipping backthrust. Brocher et al. (2004) describe a north-dipping roof thrust of the Seattle uplift.
- PU012 and PU009 follow the western Seattle fault zone, the Hood Canal area and the Seattle–Saddle Mountain transfer zone (Blakely et al., 2002, 2009; Lamb et al., 2012).

**Recurring polarity mismatch.** Many Puget surfaces dip the opposite way from the mapped master faults:
- Seattle fault and Olympia structure: S-dipping in the CFM.
- SWIF: NE-dipping in the CFM.
- Devils Mountain fault: N-dipping in the CFM.

The well-documented exceptions are the north-dipping backthrusts: Toe Jam Hill (Nelson et al., 2003), Tacoma (Johnson et al., 2004) and the roof thrust (Brocher et al., 2004). Where a surface's dip is 20–35°, the input caveat applies: the plane may be fitting a seismogenic layer rather than a fault.

Van Wagoner et al. (2002) found that Puget Lowland earthquakes sit mostly in Crescent Formation basement, while the sedimentary basins and the eastern Olympic subduction complex are nearly aseismic. They also found that hypocenters are hard to tie to individual mapped faults. Several broad, low-dip surfaces here (PU001, PU005, PU002, PU013, PU010) are best read as this basement seismicity rather than as faults.

**Interpretation to test.** PU002 dips 27° E. Its sea-level projection, at about 123.25–123.35°W, falls near the eastern Olympic contact between the Crescent Formation and the Olympic subduction complex (Hurricane Ridge fault; Brandon et al., 1998). PU002 may therefore mark the western, east-dipping edge of seismogenic Crescent basement, not a Quaternary fault.

## (b) Match table

Abbreviations: SFZ = Seattle fault zone; RMFZ = Rattlesnake Mountain fault zone; WRSZ = Western Rainier Seismic Zone; DDMFZ = Darrington–Devils Mountain fault zone.

| surface_id | match | confidence | evidence | refs |
|---|---|---|---|---|
| PU001 | Diffuse mid-crustal Puget Lowland basement seismicity. SE end reaches the WRSZ | low | 145 km long, strike 326, dip 26° NE, 8–21 km, 1025 events, unstable. Trace runs from W of Rainier to Hood Canal and crosses the E–W Seattle and Tacoma faults at about 50°. 38 km above the slab, so not intraslab. Too long and low-dipping to be one fault | Van Wagoner 2002; Stanley 1996 |
| PU002 | East-dipping western edge of seismogenic Crescent basement (Crescent / Olympic subduction complex contact, Hurricane Ridge fault trend). Alternative: Hood Canal fault zone | low | Strike 354, dip 27° E, 16–23 km, under Kitsap/Hood Canal. Sea-level projection at 123.24–123.35°W, near the E Olympic contact. Hood Canal faults strike NNE, about 25° off | Van Wagoner 2002; Brandon 1998; USGS QFFD Hood Canal |
| PU003 | Western Rainier Seismic Zone | medium | Centroid 122.09°W, 46.75°N, inside the north-trending WRSZ band W of the park. Trend 027 vs N–S zone (about 27° off), consistent with en echelon oblique faults. 1–19 km depth fits the WRSZ. Dip 25° is probably a layer effect | Stanley 1996 |
| PU004 | No known counterpart, probably several clusters chained together | none | 103 km, strike 241, dip 78° NW, unstable. Runs diagonally across the Seattle basin from Monroe to Belfair, about 60° off the SWIF and about 30° off the SFZ. No ENE fault is mapped here | — |
| PU005 | Same as PU001 (parallel duplicate) | low | Strike 321, dip 26° NE, 7–23 km, 76 km. Parallel to PU001 and offset about 10 km | Van Wagoner 2002 |
| PU006 | DDMFZ, Darrington–Oso section | low | Centroid at the DMF latitude (48.30°N), near Oso and Darrington. Trend 311 vs CFM DMF 280 (about 30° off). Dips 34° SW vs CFM 60° N, near-vertical at the trench (strike 286). Stable | Personius 2014 |
| PU007 | SE end of the SWIF in Seattle basin basement | low | Strike 282, dip 39° NNE, 8–22 km, near Kirkland. About 30° from the SWIF (311); dip direction agrees with the NE-dipping CFM SWIF. About 15 km N of the SFZ | Sherrod 2008; Johnson 1996 |
| PU008 | DDMFZ / SWIF junction volume, geometry discordant | low | Near Fidalgo/Burlington, 8–25 km. Trend 148 vs DMF (280) and Strawberry Point (105) faults, about 45° off | Personius 2014; Johnson 1996 |
| PU009 | Hood Canal fault zone (northern reach) | low–medium | Steep (59° NW), 13–21 km. Trend 049 follows the NNE–NE northern Hood Canal, where USGS mapping shows NNE-striking Quaternary faults. Near the Dabob Bay fault (CFM). About 20–30° off canal-parallel faults | USGS QFFD Hood Canal; Lamb 2012 |
| PU010 | Seismogenic-layer artifact under the N Seattle basin | none/low | Dip 14°, 15–20 km, strike 286. Overlaps PU007 in map view | Van Wagoner 2002 |
| PU011 | Leech River fault: NNE-dipping relocated seismic zone | high | Top edge 123.21–123.63°W at 48.40°N spans the eastern LRF and its offshore extension to the E, the extent Li et al. describe. Strike 269 vs LRF 278 (about 9° off). Dips 44° N vs CFM 52° N and AGU 2016 about 45°. 14.5–26.6 km, where Li et al. describe a seismic zone about 10 km thick | Li 2018; Morell 2017 |
| PU012 | Western SFZ (Bremerton–Hood Canal), Seattle–Saddle Mountain transfer zone | medium | Top edge runs from Sinclair Inlet (47.57°N, at the SFZ front) WSW to the Hood Canal Great Bend. Trend 062/242, about 25–30° off the E–W SFZ, in the area where the SFZ bends W–SW to join the Saddle Mountain fault. Dips 60° NW, opposite to the master SFZ (S-dipping) | Blakely 2002; Blakely 2009; Lamb 2012 |
| PU013 | Duplicate of PU002 | low | Strike 335, dip 22° E, Kitsap, 14–21 km | as PU002 |
| PU014 | Tacoma fault (western/central part) | low–medium | Centroid on the CFM Tacoma trace (122.69°W, 47.37°N). Trend 052 vs E–W (about 38° off). Near-vertical, consistent with steep north dip from microseismicity (Brocher et al., 2001). 14–25 km | Brocher 2001; Johnson 2004; Sherrod 2004 |
| PU015 | Duplicate of PU018 (eastern Tacoma fault / roof thrust); Seattle fault (east) backthrust also possible | low–medium | Strike 285, dip 34° N, 4–19 km, Issaquah. Same dip polarity as the north-dipping backthrusts (Toe Jam Hill, roof thrust); the SFZ dips S | Nelson 2003; Brocher 2004 |
| PU016 | Olympia structure, SE end | medium | Top edge at 122.61–122.74°W, 46.74–46.84°N, within about 8 km of the CFM Olympia SE tip. Strike 314 vs structure trend about 315° (about 1–5° off). Dips 21° NE vs CFM 60° S, so the dip is unreliable | Clement 2010 |
| PU017 | Cherry Creek fault zone / Duvall area | low | N–S (strike 0.5), dip 47° E, 2–15 km, stable. About 10 km W and N of the Cherry Creek trace (CFM, az 024). Strike 24° off. About 25 km N of the 1996 Duvall hypocenter | PNSN (Duvall 1996, web); Sherrod 2008 |
| PU018 | Eastern continuation of the Tacoma fault / north-dipping roof thrust of the Seattle uplift | medium | Top edge E–W at 47.39–47.40°N from 121.77 to 122.08°W, along the eastward projection of the CFM Tacoma trace (47.36–47.38°N; mapped only W of 122.51°W). Strike diff about 5°. North dip agrees (34° vs 60°). Johnson 2004 describes an eastern section with variable geometry | Johnson 2004; Brocher 2004 |
| PU019 | Rattlesnake Mountain fault zone (SWIF SE extension); 1996 Duvall source region | high (location), medium (geometry) | Top edge from 121.87°W, 47.79°N to 121.74°W, 47.57°N, within 3 km of the CFM RMFZ trace. Trend 158 vs 148 (about 10° off). 0.4–12 km, consistent with Duvall aftershocks at 8–10 km. Dips 46° SW vs vertical in the CFM | Sherrod 2008; PNSN Duvall (web) |
| PU020 | Hood Canal–Discovery Bay fault zone (northern) | low–medium | Port Ludlow to Port Townsend, 7–20 km. Trend 022 matches the NNE-striking faults beneath northern Hood Canal | USGS QFFD Hood Canal |
| PU021 | Shallow LRF / Langford area. Check for quarry blasts | low | 0.4–9.7 km, dip 20°, strike 24, about 3.5 km from the LRF. Very shallow and stable, near urban and quarry areas W of Victoria. Run a day/night ratio test before interpreting | Li 2018 |
| PU022 | Mount Rainier edifice VT seismicity | high | Centroid 1.5 km from the summit (121.76°W, 46.85°N), 0–10 km. Matches the steady VT clusters beneath the edifice base | Moran 2000 |
| PU023 | Possible eastern continuation of the Strawberry Point / Utsalady Point (SWIF–DDMFZ) splays | low | Strike 116 (about 10° from Strawberry Point, az 105). About 25 km SE of the mapped east ends, near Snohomish/Lake Stevens. Dips 37° SSW, 4–19 km | Johnson 1996; Sherrod 2008 |
| PU024 | Seattle fault east tip / RMFZ junction, no ENE structure known | low/none | Strike 64, dip 48°, 1–15 km, near Hobart/Tiger Mountain. Roughly 30° off the SFZ and 85° off the RMFZ | — |
| PU025 | SWIF main strand (Possession Sound) | medium | Centroid about 5 km from the CFM SWIF trace (interpolated about 47.91°N at 122.37°W). Trend 107/287 vs 311 (about 24° off). Steep (88°) vs CFM 60° NE. 13–27 km | Johnson 1996; Sherrod 2008 |
| WA001 | Secondary structure within the LRF seismic zone / Saanich–Malahat area | low | Strike 334, dip 59° ENE, 8–27 km. South end touches the LRF; about 55° off the LRF trend. Li 2018 reports subvertical secondary structures | Li 2018 |
| WA002 | No known counterpart (W Vancouver Island forearc, Barkley Sound / Ucluelet) | none | Strike 132, dip 40° SW, 10–20 km, stable. 16 km above the slab, about 55 km from the Beaufort Range fault. Coast-parallel; no published active fault found | — |
| WA003 | Gulf Islands crustal cluster (Saltspring); no named fault | none/low | Strike 105, dip 44° SSW, 9–21 km. About 30 km N of the XEOLXELEK–Elk Lake fault, which has a similar trend (125) and dip direction (50° SW). Distance makes a link unlikely | Harrichhausen 2023 |
| WA004 | Duplicate of WA003 (same cluster, steep plane) | none/low | Strike 255, dip 90°, overlaps WA003 | — |
| WA005 | Near-slab-top seismicity off La Push (upper slab or interface region), not a crustal fault | low | `slab2_minus_depth` = +3.6 km, 16–29 km. Steep plane (strike 296, dip 82°), so not the megathrust plane itself. The Lake Creek–Boundary Creek fault is 58 km away | Hayes 2018; McCrory 2012 |

**Duplicates and parallel planes:**
- PU001 and PU005
- PU002 and PU013
- PU015 and PU018 (PU024 is adjacent)
- PU007 and PU010
- WA003 and WA004
- PU011, PU021 and WA001 form one LRF family at different depths and orientations.

Most PU surfaces are flagged `stable=False`, which is consistent with planes trading events.

**No counterpart:** PU004, PU024, WA002, and WA003/WA004 (no named fault).

## (c) Possible new structures or targets

1. **PU018/PU015: eastern Tacoma fault / roof thrust.** E–W, north-dipping seismicity at 47.39–47.50°N east of 122.1°W would extend the Tacoma fault backthrust about 40 km beyond its mapped east end. Johnson et al. (2004) inferred this eastern section only from structure. Test with relocations and focal mechanisms (PR #22 set).
2. **PU004: ENE-trending steep alignment across the Seattle basin.** No mapped equivalent. Check whether it survives in the Track A NonLinLoc relocations or splits into the Duvall, SWIF and SFZ-west clusters.
3. **PU023: WNW, SSW-dipping zone near Snohomish/Lake Stevens.** Could be an eastward continuation of the Strawberry Point / Utsalady Point faults that is not in the CFM.
4. **WA002: coast-parallel forearc plane off Barkley Sound.** No published active fault. It may be a lower-plate or forearc structure, or a location artifact at the network edge; check azimuthal gap.
5. **PU002/PU013: east-dipping (about 25°) plane under Kitsap.** If robust, it would image the western limit of seismogenic Crescent basement. This is a structural boundary, not necessarily a hazard fault.
6. **WA003/WA004: Gulf Islands mid-crustal cluster.** Not tied to any CFM fault.

## (d) References

Status codes:
- **CR-OK**: resolved via api.crossref.org/works/DOI; author, year and title confirmed.
- **DC-OK**: DataCite DOI resolved via doi.org (CSL JSON).
- **Web**: confirmed only via a publisher or agency web page or search snippet.

| Reference | DOI | Status |
|---|---|---|
| Blakely, R.J., Wells, R.E., Weaver, C.S., Johnson, S.Y. (2002). Location, structure, and seismicity of the Seattle fault zone, Washington: evidence from aeromagnetic anomalies, geologic mapping, and seismic-reflection data. GSA Bulletin 114, 169–177 | 10.1130/0016-7606(2002)114<0169:LSASOT>2.0.CO;2 | CR-OK |
| Blakely, R.J., et al. (2009). Saddle Mountain fault deformation zone, Olympic Peninsula, Washington: western boundary of the Seattle uplift. Geosphere 5 | 10.1130/GES00196.1 | CR-OK |
| Brandon, M.T., Roden-Tice, M.K., Garver, J.I. (1998). Late Cenozoic exhumation of the Cascadia accretionary wedge in the Olympic Mountains. GSA Bulletin 110, 985–1009 | 10.1130/0016-7606(1998)110<0985:LCEOTC>2.3.CO;2 | CR-OK |
| Brocher, T.M., et al. (2001). Upper crustal structure in Puget Lowland, Washington: results from the 1998 SHIPS. JGR 106, 13541–13564 | 10.1029/2001JB000154 | CR-OK. 10.1029/2000JB900292 is a different paper (Behn 2000) |
| Brocher, T.M., Blakely, R.J., Wells, R.E. (2004). Interpretation of the Seattle uplift, Washington, as a passive-roof duplex. BSSA 94 | 10.1785/012003190 | CR-OK (the DOI really has 9 digits) |
| Clement, C.R., et al. (2010). High-resolution seismic reflection imaging of growth folding and shallow faults beneath the southern Puget Lowland, Washington State. BSSA 100, 1710–1723 | 10.1785/0120080306 | CR-OK (Crossref search hit; title, year and pages match) |
| Hayes, G.P., et al. (2018). Slab2, a comprehensive subduction zone geometry model. Science 362, 58–61 | 10.1126/science.aat4723 | CR-OK |
| Harrichhausen, N., et al. (2023). Discovery of an active forearc fault in an urban region: Holocene rupture on the XEOLXELEK–Elk Lake fault, Victoria, BC. Tectonics 42, e2023TC008170 | 10.1029/2023TC008170 | CR-OK |
| Ichinose, G.A., Thio, H.K., Somerville, P.G. (2004). Rupture process and near-source shaking of the 1965 Seattle-Tacoma and 2001 Nisqually intraslab earthquakes. GRL 31 | 10.1029/2004GL019668 | CR-OK. The 2006 Ichinose paper named in the brief was not located, so the 2004 paper is cited instead |
| Johnson, S.Y., Potter, C.J., Armentrout, J.M., Miller, J.J., Finn, C., Weaver, C.S. (1996). The southern Whidbey Island fault: an active structure in the Puget Lowland, Washington. GSA Bulletin 108, 334–354 | 10.1130/0016-7606(1996)108<0334:TSWIFA>2.3.CO;2 | CR-OK |
| Johnson, S.Y., et al. (2004). Active shortening of the Cascadia forearc and implications for seismic hazards of the Puget Lowland. Tectonics 23, TC1011 | 10.1029/2003TC001507 | CR-OK |
| Lamb, A.P., et al. (2012). Western limits of the Seattle fault zone and its interaction with the Olympic Peninsula, Washington. Geosphere 8, 915–930 | 10.1130/GES00780.1 | CR-OK |
| Ledeczi, A., et al. (2024). Late Quaternary surface displacements on accretionary wedge splay faults in the Cascadia subduction zone. Seismica 2(4) | 10.26443/seismica.v2i4.1158 | CR-OK. Offshore WA splay faults are all west of 125.3°W; none of the 30 surfaces matches |
| Li, G., Liu, Y., Regalla, C., Morell, K.D. (2018). Seismicity relocation and fault structure near the Leech River fault zone, southern Vancouver Island. JGR 123, 2841–2855 | 10.1002/2017JB015021 | CR-OK |
| McCrory, P.A., Blair, J.L., Waldhauser, F., Oppenheimer, D.H. (2012). Juan de Fuca slab geometry and its relation to Wadati-Benioff zone seismicity. JGR 117, B09306 | 10.1029/2012JB009407 | CR-OK |
| Moran, S.C., Zimbelman, D.R., Malone, S.D. (2000). A model for the magmatic-hydrothermal system at Mount Rainier, Washington. Bull. Volcanol. 61, 425–436 | 10.1007/PL00008909 | CR-OK |
| Morell, K.D., et al. (2017). Quaternary rupture of a crustal fault beneath Victoria, British Columbia, Canada. GSA Today 27(3), 4–10 | 10.1130/GSATG291A.1 | CR-OK |
| Nelson, A.R., et al. (2003). Late Holocene earthquakes on the Toe Jam Hill fault, Seattle fault zone, Bainbridge Island, Washington. GSA Bulletin 115, 1388–1403 | 10.1130/B25262.1 | CR-OK |
| Personius, S.F., et al. (2014). Holocene earthquakes and right-lateral slip on the left-lateral Darrington–Devils Mountain fault zone. Geosphere 10, 1482–1500 | 10.1130/GES01067.1 | CR-OK |
| Pratt, T.L., et al. (1997). Seismic reflection images beneath Puget Sound: the Puget Lowland thrust sheet hypothesis. JGR 102, 27469–27489 | 10.1029/97JB01830 | CR-OK. 10.1029/97JB01962 does not resolve. Background reference only |
| Sherrod, B.L., Brocher, T.M., et al. (2004). Holocene fault scarps near Tacoma, Washington, USA. Geology 32, 9–12 | 10.1130/G19914.1 | CR-OK. 10.1130/G20914.1 does not resolve |
| Sherrod, B.L., et al. (2008). Finding concealed active faults: extending the southern Whidbey Island fault across the Puget Lowland. JGR 113, B05313 | 10.1029/2007JB005060 | CR-OK |
| Stanley, W.D., Johnson, S.Y., Qamar, A.I., Weaver, C.S., Williams, J.M. (1996). Tectonics and seismicity of the southern Washington Cascade Range. BSSA 86(1A), 1–18 | 10.1785/BSSA08601A0001 | CR-OK |
| Van Wagoner, T.M., et al. (2002). Crustal structure and relocated earthquakes in the Puget Lowland, Washington, from high-resolution seismic tomography. JGR 107(B12), 2381 | 10.1029/2001JB000710 | CR-OK |
| Weaver, C.S., Smith, S.W. (1983). Regional tectonic and earthquake hazard implications of a crustal fault zone in southwestern Washington. JGR 88, 10371–10383 | 10.1029/JB088iB12p10371 | CR-OK |
| USGS (2021, ver. 3.0 2023). Earthquake geology inputs for the U.S. NSHM 2023 (western U.S.) | 10.5066/P9AWINWZ | DC-OK |
| Watt, J.T., et al. (2026). Quaternary fault traces of the U.S. Cascadia subduction zone offshore northern California, Oregon, and Washington. USGS data release | 10.5066/P14HJMM9 | DC-OK. No offshore-WA surface lies on these traces; WA005 and WA002 are inshore of the mapped wedge faults |
| USGS Quaternary Fault and Fold Database, Hood Canal fault zone (report 552) | none | Web only (search snippet: "NNE-striking zone of faults beneath the northern part of the canal"). PDF text not extracted; QFFD report number not confirmed |
| PNSN, 1996 Duvall earthquake page (pnsn.org/education/pnw-earthquakes/notable/duvall) | none | Web only, not peer-reviewed |
| CRESCENT Community Fault Model, local `traces.geojson` (dips: Seattle 45° S, Tacoma 60° N, SWIF 60° NE, LRF 52° N, XEOLXELEK 50° NE) | — | Local file. The CFM XEOLXELEK dip direction (NE) conflicts with Harrichhausen 2023 (50° SW, blind) |

Not used or unverified: Kelsey et al. (SWIF paleoseismology), Mace & Keranen, Staisch et al., Preston et al., and Wada et al. were not needed for any match and were not checked. Stanley et al. on the WRSZ is cited only through the 1996 BSSA paper.
