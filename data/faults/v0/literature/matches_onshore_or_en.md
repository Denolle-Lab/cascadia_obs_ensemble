# Literature match of fault surfaces v0: onshore (outside Puget and Mendocino boxes), offshore Oregon, Endeavour–Explorer

Input: `faults_onshore_or_en.csv` (46 surfaces: EN001–EN004, ON001–ON040, OR001–OR002), built from Cascadia catalog v4 (2010–2016). Members: `pv4/data/faults/v0/fault_surfaces_v0_members.csv`.

## Checks run besides the literature

1. **Blast screen.** For each surface I computed the share of events in local daytime (08–17 PST) and on weekdays. Natural seismicity gives about 0.42 daytime and 0.71 weekday. Five surfaces look mostly like quarry or construction blasts:
   - ON040: 1.00 daytime, 1.00 weekday
   - ON029: 0.95 daytime, 0.92 weekday
   - ON012: 0.92 daytime, 0.89 weekday
   - ON037: 0.83 daytime
   - ON020: 0.75 daytime, 0.86 weekday

   ON023 (0.71 daytime, 0.89 weekday) is likely contaminated by blasts.
2. **ANSS ComCat cross-check.** I queried M ≥ 2.5 events from 2010–2016 inside the footprint of each surface (USGS FDSN event service). This ties the largest events to named ComCat/PNSN events.
3. **Strike check against CRESCENT CFM v1.0.** For each surface, I found the nearest CFM traces in map view (`scratchpad/cfm/traces.geojson`) and computed the strike difference ΔS, taken modulo 180°.

Two depth caveats:
- ComCat depths offshore Oregon run about 20 km deeper than ours. OR001 is at 6 km in v4 and 26–33 km in ComCat. This agrees with the known shallow bias of v4.
- Onshore, v4 and ComCat depths are similar to within a few km.

## (a) Overview

**St. Helens seismic zone (SHZ).**
- The SHZ is a 90–100 km NNW-trending zone of dextral strike-slip, with N–S preferred nodal planes (Weaver & Smith 1983).
- Mount St. Helens sits at a dextral offset in the zone, with local extension between the fault segments (Weaver et al. 1987).
- Stanley et al. (1996) add three relevant points:
  - The SHZ is near vertical close to the volcano but dips steeply west farther north. It may merge with a band of seismicity at about 18 km depth in the lower Siletzia crust.
  - An ENE band of seismicity at 12–20 km depth, north of the Cowlitz River, steps over from the SHZ to the western Rainier zone.
  - A NE-trending zone at about 18 km depth runs from the SHZ toward Portland.
- Stanley et al. (1996) also name the **Goat Rocks zone**, a discontinuous NNW alignment of epicenters running north-northwest from Goat Rocks.
- v4 resolves all of these features: ON001/ON003, ON002/ON005, ON004/ON008 and ON018.
- The v4 dips of 35–39° W on ON001/ON003 have the same sense as the "steep westward dip" in Stanley et al. (1996) but are much shallower. A broad zone of seismicity, plus the deep band it merges with, probably flattens the fitted planes.

**Mount Hood.**
- Swarms on the south flank are regarded as tectonic (Jones & Malone 2005).
- The N-trending Mount Hood fault zone, including the Twin Lakes and Multorpor segments, has been mapped from lidar (Madin et al. 2021).
- ON015 matches the Twin Lakes fault closely: 2.5 km away, ΔS = 4°, dip 55° against 50° in the CFM.

**Portland basin.**
- Yelin & Patton (1991) tied sparse seismicity at 15–20 km depth to the Portland Hills and Frontal fault zones.
- Blakely et al. (2000) and Liberty et al. (2003) map the NW-striking Portland Hills, Sylvan–Oatfield and East Bank faults.
- ON013 (Portland Hills north, ΔS = 4°) and ON031 (Portland Hills south / Bolton, ΔS = 4–5°) agree in strike. Both sit at 15–18 km depth, the same depth range Yelin & Patton (1991) reported.
- The Mount Angel fault (Thomas et al. 1996) and the Gales Creek fault (Horst et al. 2020) have no strong v4 counterpart.

**North Cascades foothills, Fraser Lowland and San Juan Islands.**
- Literature for this area:
  - Kelsey et al. (2012): Bellingham basin faults.
  - Sherrod et al. (2013): Boulder Creek–Canyon Creek fault.
  - Balfour et al. (2012): double-difference lineations in SW British Columbia and the San Juan Islands.
  - Greene et al. (2018): the active, steep, left-lateral Skipjack Island fault zone, striking WNW.
- ON011 (WNW, 60° dip, about 13 km depth) is the best San Juan candidate. ON009 and ON010 are parallel planes fitted to the same cluster.

**Offshore Oregon.**
- Goldfinger et al. (1992, 1997) mapped WNW left-lateral forearc faults (Daisy Bank, Wecoma, Alvin Canyon) and N–S anticlines (Stonewall Bank).
- Seismicity beneath the shelf near 44.3–44.6°N has been interpreted as plate-boundary thrusting, possibly above subducted seamounts (Tréhu et al. 2008, 2012, 2015; Williams et al. 2011).
- OR001 and OR002 fall in this cluster. No surface lies on Daisy Bank or Wecoma.

**Endeavour–Explorer.**
- The 2014 Mw 6.4 Nootka sequence ruptured a new fault in the subducted Explorer plate that strikes about 156° and is more than 55 km long (Hutchinson et al. 2020, 2023). EN003 (strike 344° = 164°) contains only April–May 2014 events, so it is this fault.
- EN004 lies 1.7 km from the CFM "Southern Nootka" trace, with ΔS = 8°. The primary faults of the Nootka fault zone strike NE (Hyndman et al. 1979; Rohr et al. 2018; Hutchinson et al. 2023).
- EN002 lies in the broad conjugate-fault deformation zone of the eastern Sovanco fracture zone (Braunmiller & Nábělek 2002).
- EN001 is a ridge-parallel structure at the southern end of the Endeavour segment, where the Cobb propagating rift overlaps it (Krauss et al. 2023).

**Targets with no v4 surface.** None of the following produced a surface, mostly because they were quiet in 2010–2016 or seismicity there fell below the clustering thresholds: Leech River fault (Morell et al. 2017, 2018), Klamath Falls 1993 (Braunmiller et al. 1995), the western Rainier zone itself, Three Sisters, Newberry, Crater Lake, Entiat/Chelan, Beaufort Range, Nanoose, Daisy Bank and Wecoma.

**Outside the named region.** The ON set also includes surfaces near Redding, California (ON030, ON040).

## (b) Match table

ΔS is the strike difference modulo 180°. "v4" means our catalog. Confidence levels are high, med (medium), low and none.

| surface_id | match | conf. | evidence | refs |
|---|---|---|---|---|
| EN001 | Southern end of the Endeavour segment / Cobb propagating-rift overlap; ridge-parallel normal faulting | low | 47.60°N, 129.23°W, south of the main Endeavour seismicity (48.0–48.3°N). Strike 025°, matching the JdF ridge trend of about 020°. Near vertical, 7 km long. Swarm Nov 2013–Jan 2014. v4 depth of 13 km is too deep for the ridge (true depth < 4 km). | Krauss et al. 2023, 2025 |
| EN002 | Eastern Sovanco fracture zone, conjugate-fault deformation zone | med | 48.93°N, 129.41°W. Strike 303° (WNW), matching the dominant NW–SE fault set of Braunmiller & Nábělek (2002). Dip 84°, 51 km long. ComCat Mw 3.7–4.7 events 2011–2015 (e.g. 2014-12-12 Mw 4.7). | Braunmiller & Nábělek 2002 |
| EN003 | **2014 Mw 6.4 Nootka sequence fault** inside the subducted Explorer plate | high | All 46 events fall in Apr–May 2014; mainshock is ComCat 2014-04-24 Mww 6.5. Strike 344° (= 164°) against the published 156°, ΔS ≈ 8°. Dip 80°. 13 km long, which is shorter than the published > 55 km because v4 events are thinned offshore. | Hutchinson et al. 2020, 2023 |
| EN004 | Nootka fault zone, southern primary fault (SNF), or a N–S conjugate fault inside the zone | med | 1.7 km from the CFM "Southern Nootka" trace (local azimuth 012°), ΔS = 8°. Dip 70°. Largest event 2013-12-15 Mw 4.6. Strike-slip regime (Hutchinson et al. 2023). | Hyndman et al. 1979; Braunmiller & Nábělek 2002; Rohr et al. 2018; Hutchinson et al. 2023 |
| ON001 | **St. Helens seismic zone (SHZ)**, main strand | high | Top edge runs 46.49→45.92°N at about 122.17°W, through the volcano. Strike 182° against the N–S preferred planes, ΔS ≈ 0–15°. 64 km long against the published 90–100 km. Includes ComCat 2011-02-14 M4.3 and 2013-08-23 M3.7. Dip 39° W against "near vertical / steep W": same dip sense, dip too low. | Weaver & Smith 1983; Weaver et al. 1987; Stanley et al. 1996 |
| ON002 | SHZ base / deep band at about 12–18 km in lower Siletzia | low | Dip 6.8°, unstable. Overlaps ON001/ON003 and trades events with them. The low-dip fit is probably an artifact of the deep band of seismicity. | Stanley et al. 1996 |
| ON003 | SHZ, a second parallel strand (duplicate of ON001) | high (as SHZ) | Strike 183°, dip 35° W, parallel to ON001 about 10 km west. Unstable. Includes ComCat 2010-11-16 M4.2 near Mossyrock. | Weaver & Smith 1983; Stanley et al. 1996 |
| ON004 | Southern continuation of the SHZ toward Portland / the NE-trending SHZ–Portland zone at about 18 km | med | 45.94°N, 122.55°W (Amboy–Yacolt), 11 km depth, strike 172°. Stanley et al. (1996) describe a NE-trending zone at about 18 km running from the SHZ to Portland and a possible stepover between the Portland Hills fault and the SHZ. Includes ComCat 2013-01-31 M3.7 at Amboy. Unstable. | Stanley et al. 1996; Yelin & Patton 1991 |
| ON005 | Northern end of the SHZ / ENE stepover to the western Rainier zone at 12–20 km | med | 46.50°N, 122.35°W at 18 km, between Mossyrock and Morton, north of the Cowlitz River. This matches the depth and position of the stepover band in Stanley et al. (1996). Dip 9.5°, unstable. Includes 2015-10-29 M3.2. | Stanley et al. 1996 |
| ON006 | Mount St. Helens dextral offset zone (local extension) / east-flank volcanic seismicity | low–med | Top edge starts at the crater (46.195°N, 122.199°W) and runs NE. Strike 032°, dip 46°. Events peak in Oct 2010 and Jun 2016; the latter is close in time to the 2016 recharge swarm. A NE trend fits extension between right-stepping SHZ segments. | Weaver et al. 1987; Moran 1994; Ulberg et al. 2020 |
| ON007 | none (Wind River / Indian Heaven area) | none | 45.95°N, 122.00°W, strike 298°, dip 67°. Nearest CFM fault (Gate Creek) is 34 km away, ΔS = 50°. I found no published structure. Candidate new structure. | — |
| ON008 | NE-trending SHZ–Portland zone at about 18 km (Siletzia block) | med | 45.98°N, 122.77°W (Kalama–Woodland) at 15 km. Matches the "mean depth about 18 km" zone. Strike 185°, dip 23°, unstable. Includes ComCat 2014-01-26 M2.9 at Deer Island, z = 20 km. | Stanley et al. 1996 |
| ON009 | Same cluster as ON011 (San Juan Islands, Spieden–Stuart Island area) | low | Strike 184°, dip 29°, unstable. Parallel-plane duplicate of ON011. | Balfour et al. 2012 |
| ON010 | Same cluster as ON011 | low | Strike 255°, dip 29°, unstable. Duplicate. | Balfour et al. 2012 |
| ON011 | Skipjack Island fault zone or a parallel strand; Balfour lineation | med | 48.63°N, 123.05°W at about 13 km. Strike 292° against the WNW strike of the Skipjack Island fault zone. Dip 60° against "near vertical". Trace runs about 12–15 km south of Skipjack Island. ΔS = 11° to the CFM XEOLXELEK–Elk Lake fault, 26 km away. ComCat 2012-07-04 M3.0 and 2013-06-21 M2.9 at z = 11–14 km. | Greene et al. 2018; Balfour et al. 2012; Barrie & Greene 2018 |
| ON012 | Quarry blasts (Sumas Mountain / Chilliwack area) | none | 92% daytime, 89% weekday events, 3 km depth. | — |
| ON013 | **Portland Hills fault (north) / Sylvan–Oatfield**, deep part | med–high | 4 km from the CFM Portland Hills (north) trace, ΔS = 4°, and 7.5 km from Sylvan–Oatfield, ΔS = 6°. Depth 11–20 km, matching Yelin & Patton's 15–20 km events. Dip 60° to the NE. ComCat 2012-11-19 M3.2 (z = 20) and 2012-12-24 M2.8 NW of North Portland. | Yelin & Patton 1991; Blakely et al. 2000; Liberty et al. 2003; Wells et al. 2020 |
| ON014 | **2015-07-04 Springfield earthquake** (ComCat M4.1; SLU Mw 3.75); no mapped Quaternary fault | med (event) / none (named fault) | 44.14°N, 122.88°W. Strike 138°, dip 68°. SLU moment-tensor nodal plane 325°/60°: strike within 7°, but that plane dips NE while ON014 dips SW. Nearest CFM fault is 58 km away. Candidate new structure. | SLU moment tensor (not peer-reviewed) |
| ON015 | **Mount Hood fault zone, Twin Lakes segment** | high | 2.5 km from the CFM Twin Lakes trace, ΔS = 4°. Dip 55° against 50° in the CFM. 23 km long N–S on the south flank. South-flank tectonic swarms (May 2016, Sep 2014). | Jones & Malone 2005; Madin et al. 2021 |
| ON016 | MSH east flank, overlapping ON006 | low | Strike 019°, dip 32°, unstable. Duplicate of the volcanic/offset zone at the volcano. | Weaver et al. 1987 |
| ON017 | SHZ segment south of the volcano | low–med | 46.12°N, 122.15°W, 5 km depth. Strike 146°, dip 83°, about 15–20° off the SHZ trend. Fits the southern segment in the dextral-offset model. Includes 2016-02-08 M2.9. | Weaver et al. 1987; Weaver & Smith 1983 |
| ON018 | **Goat Rocks zone (GRZ)**, NNW alignment north of Goat Rocks near Packwood | med | Top edge runs 46.72→46.52°N at 121.60°W. Strike 174° against the GRZ NNW trend. 4.9 km depth. ComCat 2011-04-06 M3.4, 11 km N of Packwood. Dip 18°, unstable and unreliable. | Stanley et al. 1996 |
| ON019 | Mount Hood south-flank swarm zone (Twin Lakes / Multorpor graben) | med | Small surface (9 km long). 6.7 km from Twin Lakes, ΔS = 17°. ComCat 2010-05-14 M3.0, 6 km N of Government Camp. Partial duplicate of ON015. | Jones & Malone 2005; Madin et al. 2021 |
| ON020 | Probable quarry blasts near Clatskanie (plus sparse events) | none | 75% daytime, 86% weekday events. ComCat has only one M ≥ 2.5 event, at 48 km depth. Gales Creek is 55 km away. | — |
| ON021 | Possible companion of ON014 (same strike, 10 km north) | low | Strike 138°, dip 44°. Natural timing (33% daytime). Candidate new structure. | — |
| ON022 | **Site of the 2025-02-21 Salmon Inlet earthquake (Mww 4.8, ComCat 49.693°N, 123.545°W)**; no mapped fault | none (literature) | Centroid about 4 km from the 2025 epicenter. Strike 006°, dip 78°, 11 km depth. Swarm in Feb 2015 including the ML 3.4 on 2015-02-15 (NRCan). The Sunshine Coast has low historical seismicity. Candidate new structure. | NRCan event page; SSA 2025 abstract (not peer-reviewed) |
| ON023 | Blast-contaminated; possibly an East Bank / Lacamas Lake–parallel structure | low | Strike 128°. Lacamas Lake fault is 11.5 km away with ΔS = 9°; Portland Hills (north) has ΔS = 12°. 71% daytime, 89% weekday events. | Blakely et al. 2000 |
| ON024 | none (Sweet Home–Lebanon foothills) | none | Strike 240° (060°). Turner–Mill Creek is 42 km away. Candidate new structure, low event count. | — |
| ON025 | Portland Hills / Sylvan–Oatfield, deep part (duplicate of ON013) | low–med | 0.1 km from the Sylvan–Oatfield trace, ΔS = 15°. 17 km depth. Dip 23°, unstable. Includes 2016-07-17 M3.1 at Beaverton (z = 27). | Yelin & Patton 1991; Blakely et al. 2000 |
| ON026 | none (Acme–Sudden Valley, south of the Bellingham basin) | low | Strike 069°, dip 89°, 12 km depth. Parallel to the Boulder Creek (Kendall) fault (ΔS = 6°) but 20 km south of it. 2016-09-07 M3.2. Candidate new structure, or a deep parallel to Boulder Creek–Canyon Creek. | Sherrod et al. 2013; Kelsey et al. 2012 |
| ON027 | Recurring swarms about 19 km N of Mount McLoughlin (east of Butte Falls) | low | Swarms in Aug 2011 and Mar 2013 (M3.1), matching the PNSN 2013 blog. Strike 062°. Sky Lakes fault is 11 km away with ΔS = 65°. No peer-reviewed study found. | PNSN blog (not peer-reviewed) |
| ON028 | Woodburn–Canby deep events; Canby / Mount Angel fault area | low | Strike 000°, dip 34°, 18 km depth. Canby fault ΔS = 25° (16 km away); Mount Angel fault ΔS = 49° (10.6 km away). ComCat 2016-10-04 M3.0 (z = 24) and 2012-09-08 M3.5 near Scotts Mills. | Thomas et al. 1996; Blakely et al. 2000 |
| ON029 | Quarry blasts (Pitt Meadows / Maple Ridge) | none | 95% daytime, 92% weekday events. | — |
| ON030 | Deep crustal seismicity beneath Redding / Keswick (outside the target region) | low | Strike 277°, dip 43°. ComCat z = 18–31 km, including 2015-05-06 M3.2 at Keswick. Bear Creek fault is 18 km away with ΔS = 29°. | — |
| ON031 | **Portland Hills fault (south) / Bolton fault**, deep part | med | 2 km from Portland Hills (south), ΔS = 4°; 4 km from Bolton, ΔS = 5°. Dip 84°, but depth 12–21 km lies well below the mapped fault. Small events (≤ M1.3). | Blakely et al. 2000; Liberty et al. 2003; Wells et al. 2020 |
| ON032 | West side of the northern SHZ (Mossyrock–Onalaska) | low | Strike 176°, dip 33°, 13 km depth. Parallel to the SHZ, 20 km west of it. Unstable. | Stanley et al. 1996 |
| ON033 | none (Carson / lower Wind River) | none | Strike 135°, dip 63°. Gate Creek is 20 km away with ΔS = 33°. ComCat 2014-11-19 M2.8, 3 km N of Carson. Candidate new structure. | — |
| ON034 | Abbotsford–Sumas; parallel to Boulder Creek (Kendall), ΔS = 3°, 23 km NW of it | low | Strike 248° (068°), dip 36°, 9 km depth. Natural timing. Includes 2015-07-05 M3.3 (7 km ENE of Sumas). No peer-reviewed fault in the Fraser Lowland at this position. | Kelsey et al. 2012; Balfour et al. 2012 |
| ON035 | none (Lebanon–Scio) | none | Strike 350°, dip 63°. Turner–Mill Creek is 26 km away with ΔS = 89°. Candidate new structure, low event count. | — |
| ON036 | none (Jervis Inlet / Powell River coast) | none | Strike 081°, dip 26°. 42 events, no ComCat M ≥ 2.5. Location uncertain at the network edge. | — |
| ON037 | Blasts / artifact (Eugene–Springfield) | none | 83% daytime events. Top edge at −13 km, dip 29 ± 20°. | — |
| ON038 | **Oct 2015 swarm at the Davis Lake volcanic complex** (SW of La Pine) | med (event) | 29 of 36 events fall in Oct 2015, matching the USGS/PNSN swarm report (ComCat M2.9 on 2015-10-24/26). Strike 304° (124°) cuts across the La Pine graben, which is 5 km away with ΔS = 74°. | USGS CVO news, PNSN blog (not peer-reviewed) |
| ON039 | Boulder Creek–Canyon Creek fault zone (Kendall), cross-strike | low | 3 km from the CFM Boulder Creek trace, but strike 196° (016°) is about 74° off the E–W fault. Probably a N–S structure crossing the fault, or a misfit. | Sherrod et al. 2013 |
| ON040 | Quarry blasts (Redding area) | none | 100% daytime, 100% weekday events. | — |
| OR001 | Shelf cluster west of Yachats/Waldport: plate-boundary / subducted-seamount seismicity; Alvin Canyon fault as alternative | med (cluster) / low (Alvin Canyon) | 44.35°N, 124.49°W. ComCat z = 26–33 km (v4 6 km, shallow-biased), including 2012-03-20 M3.7. Strike 272°; Alvin Canyon is 8 km away with ΔS = 20°. Dip 29° is close to the interface dip. | Tréhu et al. 2008, 2012, 2015; Williams et al. 2011; Goldfinger et al. 1997 |
| OR002 | Stonewall Bank anticline / shelf cluster off Newport | low–med | 10 km from the CFM Stonewall Anticline, ΔS = 11°. Steep (81°). ComCat 2012-04-18 M2.9, z = 17. An anticline is a fold, so the seismicity is probably on the underlying thrust or the interface. | Goldfinger et al. 1992; Tréhu et al. 2015 |

### Duplicates and parallel-plane families

| family | surfaces | note |
|---|---|---|
| SHZ | ON001, ON003, ON002, ON005, ON032 | ON001 is the preferred plane; ON003 is a parallel strand. ON002 and ON005 are low-dip fits to the deep band and stepover. |
| MSH volcano / offset | ON006, ON016, ON017 | |
| San Juan Islands | ON009, ON010, ON011 | One cluster of about 179 events; ON011 is preferred. |
| Mount Hood | ON015, ON019 | |
| Portland Hills deep | ON013, ON025 | |
| Springfield foothills | ON014, ON021 (possibly ON024, ON035) | |
| Fraser Lowland | ON012 (blasts), ON034 | |

## (c) Possible new structures (no published fault)

Ordered by robustness.

1. **ON022, Salmon Inlet (Sunshine Coast, BC).**
   - A steep N-striking zone at about 11 km depth.
   - It was active in 2015 and lies about 4 km from the epicenter of the 2025-02-21 Mww 4.8 earthquake, the largest crustal event in the area.
   - It has natural timing.
   - No mapped Quaternary fault exists here.
   - This is the strongest "new structure" case, and the 2015 v4 seismicity predates the 2025 event.
2. **ON014 (with ON021), Springfield–Mohawk foothills, Oregon.**
   - A NW-striking steep plane hosting the 2015 M4.1 Springfield earthquake.
   - Its strike agrees within 7° with the SLU NW nodal plane (325°/60°, dipping NE), but the dip direction differs: ON014 dips SW.
   - It is 50–70 km from any CFM fault.
3. **ON026 (and ON034), north Cascades foothills / Fraser Lowland.**
   - ENE-striking planes parallel to Boulder Creek–Canyon Creek (ΔS = 3–6°), offset 20–23 km from it.
   - Possibly deep parallel strands of the same transpressional system.
4. **ON007 and ON033, Wind River / Carson.**
   - NW-striking zones between the SHZ and Mount Hood, with no mapped Quaternary fault.
   - Could connect to the Gate Creek / Blue Ridge structures. I found no literature on them.
5. **ON024 and ON035, southern Willamette Valley foothills.** Low event counts; treat as tentative.
6. **ON036, Jervis Inlet.** At the edge of the network; tentative.

## (d) References (DOIs checked against Crossref unless noted)

| reference | DOI / link | status |
|---|---|---|
| Weaver, C.S., Smith, S.W. (1983). Regional tectonic and earthquake hazard implications of a crustal fault zone in southwestern Washington. JGR 88(B12), 10371–10383. | 10.1029/JB088iB12p10371 | verified (Crossref) |
| Weaver, C.S., Grant, W.C., Shemeta, J.E. (1987). Local crustal extension at Mount St. Helens, Washington. JGR 92(B10), 10170–10178. | 10.1029/JB092iB10p10170 | verified (Crossref) |
| Moran, S.C. (1994). Seismicity at Mount St. Helens, 1987–1992: Evidence for repressurization of an active magmatic system. JGR 99(B3), 4341–4354. | 10.1029/93JB02993 | verified (Crossref) |
| Stanley, W.D., Johnson, S.Y., Qamar, A.I., Weaver, C.S., Williams, J.M. (1996). Tectonics and seismicity of the southern Washington Cascade Range. BSSA 86(1A), 1–18. | 10.1785/BSSA08601A0001 | verified (Crossref; full text read) |
| Ulberg, C.W., et al. (2020). Local source Vp and Vs tomography in the Mount St. Helens region with the iMUSH broadband array. G-cubed 21, e2019GC008888. | 10.1029/2019GC008888 | verified (Crossref) |
| Kiser, E., et al. (2016). Magma reservoirs from the upper crust to the Moho ... beneath Mount St. Helens. Geology 44, 411–414. | 10.1130/G37591.1 | verified (Crossref); background |
| Bedrosian, P.A., et al. (2018). Crustal inheritance and a top-down control on arc magmatism at Mount St Helens. Nat. Geosci. 11, 865–870. | 10.1038/s41561-018-0217-2 | verified (Crossref); background |
| Jones, J., Malone, S.D. (2005). Mount Hood earthquake activity: Volcanic or tectonic origins? BSSA 95(3), 818–832. | 10.1785/0120040019 | verified (Crossref lists only the first author) |
| Madin, I.P., Streig, A.R., Bennett, S.E.K. (2021). The Mount Hood fault zone, active faulting at the crest of the dynamic Cascade Range, north-central Oregon, USA. GSA Field Guide 62, 49–71. | 10.1130/2021.0062(03) | verified (Crossref) |
| Yelin, T.S., Patton, H.J. (1991). Seismotectonics of the Portland, Oregon, region. BSSA 81(1), 109–130. | 10.1785/BSSA0810010109 | verified (Crossref) |
| Blakely, R.J., Wells, R.E., Tolan, T.L., Beeson, M.H., Tréhu, A.M., Liberty, L.M. (2000). New aeromagnetic data reveal large strike-slip(?) faults in the northern Willamette Valley, Oregon. GSA Bull. 112(8), 1225–1233. | 10.1130/0016-7606(2000)112<1225:NADRLS>2.0.CO;2 | verified (Crossref) |
| Liberty, L.M., Hemphill-Haley, M.A., Madin, I.P. (2003). The Portland Hills Fault: uncovering a hidden fault in Portland, Oregon using high-resolution geophysical methods. Tectonophysics 368, 89–103. | 10.1016/S0040-1951(03)00152-5 | verified (Crossref) |
| Wells, R.E., et al. (2020). Geologic map of the greater Portland metropolitan area and surrounding region, Oregon and Washington. USGS SIM 3443. | 10.3133/sim3443 | verified (Crossref) |
| Thomas, G.C., Crosson, R.S., Carver, D.L., Yelin, T.S. (1996). The 25 March 1993 Scotts Mills, Oregon, earthquake and aftershock sequence: ... the Mount Angel fault. BSSA 86(4), 925–935. | 10.1785/BSSA0860040925 | verified (Crossref) |
| Horst, A.E., Streig, A.R., Wells, R.E., Bershaw, J. (2020/2021). Multiple Holocene earthquakes on the Gales Creek fault, northwest Oregon fore-arc. BSSA 111(1), 476–489. | 10.1785/0120190291 | verified (Crossref). This is the Gales Creek paper; I found no "Wells et al. 2020 Tectonics" Gales Creek paper. |
| Kelsey, H.M., Sherrod, B.L., Blakely, R.J., Haugerud, R.A. (2012). Holocene faulting in the Bellingham forearc basin. JGR 117, B03409. | 10.1029/2011JB008816 | verified (Crossref) |
| Sherrod, B.L., et al. (2013). Holocene tectonics and fault reactivation in the foothills of the north Cascade Mountains, Washington. Geosphere 9(4), 827. | 10.1130/GES00880.1 | partly verified: Crossref confirms the DOI, first author, journal, volume and page, but its record has no title. Title from memory. |
| Balfour, N.J., Cassidy, J.F., Dosso, S.E. (2012). Identifying active structures using double-difference earthquake relocations in southwest British Columbia and the San Juan Islands, Washington. BSSA 102(2), 639–649. | 10.1785/0120110056 | verified (Crossref) |
| Balfour, N.J., Cassidy, J.F., Dosso, S.E., Mazzotti, S. (2011). Mapping crustal stress and strain in southwest British Columbia. JGR 116, B03314. | 10.1029/2010JB008003 | verified (Crossref); background |
| Greene, H.G., Barrie, J.V., Todd, B.J. (2018). The Skipjack Island fault zone: An active transcurrent structure within the upper plate of the Cascadia subduction complex. Sediment. Geol. 378, 61–79. | 10.1016/j.sedgeo.2018.05.005 | verified (Crossref) |
| Barrie, J.V., Greene, H.G. (2018). The Devils Mountain Fault zone: An active Cascadia upper plate zone of deformation. Sediment. Geol. 364, 228–241. | 10.1016/j.sedgeo.2017.12.018 | verified (Crossref) |
| Morell, K.D., et al. (2017). Quaternary rupture of a crustal fault beneath Victoria, British Columbia, Canada. GSA Today 27(3), 4–10. | 10.1130/GSATG291A.1 | verified (Crossref); no v4 match |
| Morell, K.D., et al. (2018). Holocene surface rupture history of an active forearc fault redefines seismic hazard in southwestern British Columbia. GRL 45. | 10.1029/2018GL078711 | verified (Crossref); no v4 match |
| Goldfinger, C., Kulm, L.D., Yeats, R.S., Appelgate, B., MacKay, M.E., Moore, G.F. (1992). Transverse structural trends along the Oregon convergent margin. Geology 20, 141. | 10.1130/0091-7613(1992)020<0141:TSTATO>2.3.CO;2 | verified (Crossref) |
| Goldfinger, C., Kulm, L.D., Yeats, R.S., McNeill, L., Hummon, C. (1997). Oblique strike-slip faulting of the central Cascadia submarine forearc. JGR 102(B4), 8217–8243. | 10.1029/96JB02655 | verified (Crossref) |
| Tréhu, A.M., Braunmiller, J., Nábělek, J.L. (2008). Probable low-angle thrust earthquakes on the Juan de Fuca–North America plate boundary. Geology 36, 127. | 10.1130/G24145A.1 | verified (Crossref) |
| Williams, M.C., Tréhu, A.M., Braunmiller, J. (2011). Seismicity at the Cascadia plate boundary beneath the Oregon continental shelf. BSSA 101(3), 940–950. | 10.1785/0120100198 | verified (Crossref) |
| Tréhu, A.M., Blakely, R.J., Williams, M.C. (2012). Subducted seamounts and recent earthquakes beneath the central Cascadia forearc. Geology 40, 103–106. | 10.1130/G32460.1 | verified (Crossref) |
| Tréhu, A.M., Braunmiller, J., Davis, E. (2015). Seismicity of the central Cascadia continental margin near 44.5°N: A decadal view. SRL 86(3), 819–829. | 10.1785/0220140207 | verified (Crossref); abstract not read |
| Hyndman, R.D., Riddihough, R.P., Herzer, R. (1979). The Nootka Fault Zone – a new plate boundary off western Canada. GJI 58, 667–683. | 10.1111/j.1365-246X.1979.tb04801.x | verified (Crossref) |
| Braunmiller, J., Nábělek, J. (2002). Seismotectonics of the Explorer region. JGR 107(B10). | 10.1029/2001JB000220 | verified (Crossref) |
| Rohr, K.M.M., Furlong, K.P., Riedel, M. (2018). Initiation of strike-slip faults, serpentinization, and methane: The Nootka fault zone. G-cubed 19, 4290–4312. | 10.1029/2018GC007851 | verified (Crossref) |
| Hutchinson, J., et al. (2020). Significant geometric variation of the subducted plate beneath the northernmost Cascadia subduction zone ... 2014 M 6.4 earthquake sequence. EPSL 551, 116569. | 10.1016/j.epsl.2020.116569 | verified (Crossref) |
| Hutchinson, J., et al. (2023). Tectonic evolution of the Nootka fault zone and deformation of the shallow subducted Explorer plate. Sci. Rep. 13, 7873. | 10.1038/s41598-023-33310-z | verified (Crossref; full text read: Nootka Sequence fault strike about 156°, > 55 km) |
| Krauss, Z., et al. (2023). A long-term earthquake catalog for the Endeavour segment ... propagating rifts. JGR 128, e2022JB025662. | 10.1029/2022JB025662 | verified (Crossref; abstract read) |
| Krauss, Z., et al. (2025). A single-station earthquake catalog for the Endeavour segment of the Juan de Fuca Ridge (2011–2016). SRL. | 10.1785/0220240427 | verified (Crossref) |
| Braunmiller, J., Nábělek, J., Leitner, B., Qamar, A. (1995). The 1993 Klamath Falls, Oregon, earthquake sequence: Source mechanisms from regional data. GRL 22, 105–108. | 10.1029/94GL02844 | verified (Crossref); no v4 match |
| Stone, I., Vidale, J.E., Han, S., Roland, E. (2018). Catalog of offshore seismicity in Cascadia. JGR 123, 641–652. | 10.1002/2017JB014966 | verified (Crossref); background |
| Hatem, A.E., et al. (2021, ver. 3.0 2023). Earthquake geology inputs for the U.S. NSHM 2023 (western U.S.). USGS data release. | 10.5066/P9AWINWZ | verified (DataCite) |
| Watt, J.T. (2026). Quaternary fault traces of the U.S. Cascadia subduction zone offshore northern California, Oregon, and Washington. USGS data release. | 10.5066/P14HJMM9 | verified (DataCite); not inspected |
| Fildes, R., Amos, C., et al. (2025). CRESCENT CFM: A 3-D Community Fault Model for the Cascadia Subduction Zone. GSA Abstr. Prog. | 10.1130/abs/2025AM-7201 | verified (Crossref; abstract only). CFM v1.0 traces used locally. |

### Not peer-reviewed (used only for event identification)

- Springfield 2015-07-04 moment tensor: SLU, https://www.eas.slu.edu/eqc/eqc_mt/MECH.NA/MECH.NA.2015/20150704154216/HTML.REG/index.html (read: 90/45/45 and 325/60/125, Mw 3.75, 4 km).
- USGS CVO news "Seismic stations pick up earthquake swarm near Gilchrist, Oregon" (Oct 2015): https://www.usgs.gov/news/seismic-stations-pick-earthquake-swarm-near-gilchrist-oregon
- PNSN blog "Small swarm near Mount McLoughlin last night" (2013-03-24): https://www.pnsn.org/blog/2013/03/24/small-swarm-near-mount-mcloughlin-last-night
- NRCan event 2015-02-15 ML 3.4 north of Sechelt: https://seismescanada.rncan.gc.ca/recent/2015/20150215.0412/index-en.php
- ComCat 2025-02-21 Mww 4.8 at Salmon Inlet: 49.693°N, 123.545°W.

### Unverifiable or not found

- Qamar & Zollweg (1990), on the 1990 Deming earthquakes (SRL): no Crossref record found; not cited in the table.
- "Wells et al. 2020, Tectonics" on the Gales Creek fault: not found. The peer-reviewed Gales Creek paper is Horst et al. (2020/2021, BSSA).
- "Kreemer et al." on Explorer: not found. Braunmiller & Nábělek (2002) is used instead.
- "Jensen et al." and "Lisowski" on St. Helens seismicity: not searched further; no SHZ structure depends on them.
