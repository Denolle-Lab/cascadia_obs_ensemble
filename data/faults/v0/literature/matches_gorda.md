# Gorda plate and Gorda Ridge: literature match for fault surfaces GO001–GO020

Input: `faults_gorda.csv` (20 planar surfaces from Cascadia catalog v4).
Supporting checks done for this note (scratch scripts only, no project files edited):

- **Event timing per surface.** Catalog v4 events within ±3 km of each plane and inside its rectangle. In this region the catalog effectively covers 2010–2015 (2016 has only 189 events), and most events fall between 2012 and 2015, when the Cascadia Initiative OBS array was deployed.
- **Moment tensors.** ComCat moment tensors (`data/focal/comcat_mt.csv`) and ComCat M≥5 events from 2000 to 2026.
- **Ridge axis.** Picked from GMRT bathymetry profiles (`data/gmrt/gmrt_-128.8_-123.8_40.0_43.2_med.nc`). Axial deep at about 127.47°W for 40.9–41.2°N, 127.40°W for 41.5–41.6°N, 127.19°W for 41.7–41.9°N, 127.00°W at 42.4°N, 126.88°W at 42.5°N, 126.80°W at 42.6°N and about 126.6°W for 42.9–43.0°N.

Location note: on the ridge, the v4 surfaces sit on the bathymetric axial valley. Teleseismic ComCat epicentres of ridge earthquakes plot about 10–20 km east of the axis (for example the 2008 M5.7, 2014 M5.5 and 2011 M5.2 events). Map positions in v4 therefore look better than ComCat's offshore. Absolute depths do not. Plane depths of 20–35 km near the Mendocino transform are far deeper than expected for 0–5 Myr lithosphere, so treat them with caution. Ren et al. (2025, EGU abstract) report OBS-located events down to about 40 km along the Mendocino transform, so deep seismicity there is not unique to v4.

---

## (a) Known structures

**1. Gorda Ridge (spreading centre, intermediate rate, about 55 mm/yr full rate).**
- Older work splits the ridge into three segments. More recent work, based on multibeam axis offsets, uses five; from north to south these are North Gorda, Jackson, Central, Phoenix and Escanaba (Chadwick et al., 1998, as cited by Clague et al., 2020).
- **North Gorda segment.** About 65 km long, shallowest near 42.75°N. Site of the February–March 1996 dike intrusion and eruption, which SOSUS T-phases detected (Fox & Dziak, 1998; Chadwick et al., 1998; Clague et al., 2020).
- **Jackson segment.** Lies south of North Gorda. NOAA PMEL detected a volcanic swarm there in 2001 (web archive, not peer reviewed). The President Jackson Seamounts, a 65 km volcanic chain starting near 42°20′N, sit off-axis to the west (Davis & Clague, 2000; Stoddard, 1989).
- **Escanaba Trough.** Sediment-filled southernmost segment. It runs about 130 km north from the Mendocino ridge–transform intersection (RTI; 40°25′N, 127°30′W) to a right-lateral axis offset near 41°35′N. Its volcanic centres are SESCA (40°45′N) and NESCA (41°00′N) (Morton et al., 1987 USGS OFR; Zierenberg et al., 1993).
- **Faulting style.** ComCat mechanisms on the axis show normal faulting with N–S to NNE strikes.

**2. Gorda plate interior (Gorda Deformation Zone).**
- The plate deforms internally under N–S compression from the Pacific plate across the Mendocino transform (Silver, 1971; Riddihough, 1980; Wilson, 1986, 1989; Stoddard, 1987, 1991).
- The accepted mechanism is left-lateral slip on NE-striking faults. These are spreading-fabric normal faults reactivated as strike-slip faults by flexural slip (Stoddard, 1987; Smith et al., 1993; Gulick et al., 2001; Chaytor et al., 2004).
- Dziak et al. (2001) mapped this fabric with multibeam bathymetry. Chaytor et al. (2004) used multibeam bathymetry and T-phase seismicity. Secondary citations report that they found a second generation of strike-slip faults separating the northern and southern plate, and drag folding next to pseudofaults.
- Hydroacoustic monitoring found a NNE-trending band of microseismicity through the centre of the plate in 1991–92 (Fox & Dziak, 1999).
- Rollins & Stein (2010) count 20 M≥5.9 earthquakes since 1976 on four fault orientations. These include NE left-lateral faults, NW right-lateral faults (for example July 1991), the Mendocino transform and thrusts.
- Velasco et al. (1994) used rupture directivity to show that the 17 August 1991 Mw 7.1 event (41.68°N, 125.86°W) ruptured a SW-striking fault, while the 13 July 1991 Mw 6.8 event (42.18°N, 125.64°W) ruptured a SE-striking one. Both orientations are active.
- Ren et al. (2025, EGU abstract; independent PickBlue OBS catalog for 2012–2015) find that intraplate microseismicity follows spreading fabric and magnetic anomalies. Its northern limit lines up with pseudofault traces between 42°N and 42.5°N.
- Ismat et al. (2022) describe the plate as a ductile deformation zone that shortens N–S and rotates clockwise.
- Recent large events inside this region:
  - 2005-06-15 Mw 7.2 (41.29°N, 125.95°W). Nodal planes strike 317° and about 47°; it is interpreted as a NE left-lateral rupture, and its first-day aftershocks extend SW to about 40.8°N, 126.5°W.
  - 2005-06-17 Mw 6.6 (40.77°N, 126.57°W). Nodal plane strike 293°, so the conjugate strikes about 023°.
  - 2006 M5.0–5.1 events.
  - 2021-06-04 M5.9 doublet (42.27°N, 126.25°W; strike about 052°).
  - 2025-09-09 M5.7 (42.36°N, 126.56°W; strike 055°).
- The 1980 Mw 7.2 Trinidad/Eureka, 2010 M6.5, 2014 M6.8 and 2021/2022 events are inside the excluded triple-junction box. Guo et al. (2024) cover the 2022 event and Hsu et al. (2026) cover the triple-junction region.

**3. Mendocino transform (right-lateral, Pacific–Gorda), from the RTI at about 127.5°W eastward.**
- Strike-slip mechanisms strike 265°–275° (or 087°–095°): 1996 M5.5, 1997 M5.4, 2008 M5.9, 2009, 2015-04-24 M5.4, 2015-04-10 M4.7 and 2016-12-08 Mw 6.6 (40.45°N, 126.19°W; strike 95°). The 2018 M5.8 has nodal planes at 271° and 181°.
- Sverdrup (1987) relocated earthquakes near the RTI.
- Hsu et al. (2026) find background rate and aftershock productivity on the transform decrease westward with distance from the triple junction.
- Chen & McGuire (2016) measured source parameters with the 2012–2013 Cascadia Initiative OBS array. A 2014 AGU abstract by the same authors reports lower stress drops near the plate boundary than inside the Gorda plate; the 2016 paper itself was not read.

**4. Margin near Brookings.** Stone et al. (2018) note abundant offshore OBS seismicity south of 43°N but name no structure off Brookings.

---

## (b) Match table

Strike Δ is the angle between the surface strike and the reference strike, using the 180° equivalence for vertical faults or the mechanism nodal plane named in the evidence. "Unstable" is the CSV `stable=False` flag.

| surface_id | best match | confidence | evidence | refs |
|---|---|---|---|---|
| GO001 | Mendocino transform, segment at 125.7–126.5°W that ruptured in the **2016-12-08 Mw 6.6** event | high | Strike 94°, dip 78°, length 61 km, 282 events, stable. Surface at 40.43–40.47°N follows the transform. The 2016 Mw 6.6 (strike 95°, Δ1°) and 2015-04-10 M4.7 (strike 87°) mechanisms lie on it. Events are 2012–2015 with a peak April–July 2015, so this is activity **before the 2016 Mw 6.6**, not its aftershocks (the catalog ends before Dec 2016). Right-lateral slip. Depth 16–29 km matches Ren et al.'s deep transform seismicity. | Silver 1971; Rollins & Stein 2010; Hsu et al. 2026; Ren et al. 2025 |
| GO002 | Escanaba Trough axis, northern half (NESCA area); ridge-axis normal faults | high | Trace 40.97–41.54°N at 127.39–127.49°W, matching the bathymetric axial deep (about 127.45°W). Strike 187° (about 007°) matches the trough trend. Normal mechanisms nearby: 2016-01-30 M5.1/M5.0 (strikes 352° and 359°, Δ8–15°) and 2004-06-22 (strikes 13–15°). Length 63 km is about half the 130 km segment. Swarms in May 2013 and August 2015. | Morton et al. 1987; Zierenberg et al. 1993; Chadwick et al. 1998 |
| GO003 | Ridge axis north of the 41°35′N offset (Central segment, or possibly Phoenix; segment name not verified); axial normal faults | high | Trace 41.74–42.04°N at 127.11–127.17°W, on the axial deep at 127.19°W. Strike 8° vs. normal mechanisms 2014-03-13 M5.5 (13°, Δ5°), 2013-08-19 (10°), 2010-10-13 and 2011-06-18 (169–177°). Catalog maximum is M4.7 on 2014-03-13, the same day as that M5.5. Swarm February 2013 (63 events). | Chadwick et al. 1998; Clague et al. 2020 |
| GO004 | Mendocino transform, westernmost active segment just east of the Escanaba RTI | high | Trace 127.15–127.49°W at 40.45–40.47°N, strike 94° (vertical). Hosts the **2015-04-24 M5.4** (ComCat 40.456°N, 127.151°W; strike 265°, vertical, rake 167°, Δ9°); catalog maximum M5.6 on 2015-04-24, with an aftershock burst April–May 2015. 2008-11-28 M5.9 (strike 270°) is just east. Unstable, trading events with GO006 and GO019. | Sverdrup 1987; Rollins & Stein 2010 |
| GO005 | Southern Escanaba Trough axial faults, where the axis bends SE into the RTI (SESCA area) | medium | Trace 40.47–40.79°N at 127.34–127.53°W, strike 155° (= 335°). Follows the axial deep as it curves (127.55°W at 40.7°N, 127.32°W at 40.5°N). Normal mechanisms 2022-12-10 M4.9 (339°, Δ4°), 2019-05-05 M5.0 (179°), 2021-01-27 (339°) and 2020-05-18 M5.0/5.3 (173–182°). Their mechanism dips are 23–33°, against 76° here; dip is poorly constrained in v4. | Morton et al. 1987; Sverdrup 1987 |
| GO006 | Same as GO004 (Mendocino transform near the RTI) | medium (location), low (geometry) | Strike 262°, Δ12° from GO004. Dip 46° does not fit a vertical transform. Unstable. Peak April 2015 (M5.4 aftershocks). Treat as a duplicate of GO004 with a dip artefact. | as GO004 |
| GO007 | Duplicate of GO005 | medium | Strike 148°, same location, unstable. | as GO005 |
| GO008 | Ridge axis at 42.28–42.52°N: Jackson segment, possibly its junction with southern North Gorda; axial normal faults next to the President Jackson Seamounts | high | Trace 126.93–127.09°W, on the axial deep at 127.00°W (42.4°N) and 126.88°W (42.5°N). Strike 26° vs. normal mechanisms 2008-03-15 M5.7 (23°, Δ3°), 2024-03-27 M5.6 (7°) and 2007-01-31 (24–35°). The seamount chain starts near 42°20′N west of the axis. PMEL reported a Jackson-segment swarm in 2001. | Davis & Clague 2000; Stoddard 1989; Chadwick et al. 1998; Clague et al. 2020 |
| GO009 | Duplicate of GO005 | medium | Strike 160°, unstable, peak October 2012. | as GO005 |
| GO010 | Persistent intraplate source near 41.85°N, 126.1°W; WNW/NW strike-slip zone (right-lateral on the WNW plane, or the conjugate NE left-lateral plane) | medium | Strike 124°, dip 50°, unstable (shares events with GO011 and GO017). Repeated ComCat mechanisms with NW-striking nodal planes: 2014-10-03 M4.3 (117°), 2016-09-25 M4.9 (147°), 2019-05-11 (138°), 2020-06-17 (129°) and 2026-06-03 M5.7 (320°). The 2006-03-25 M5.1 (41.75°N) has a NE plane (226°). No published fault trace here, but NW right-lateral faults are one of the Gorda orientations (July 1991, Velasco et al.; Rollins & Stein). | Velasco et al. 1994; Rollins & Stein 2010; Smith et al. 1993 |
| GO011 | Same zone as GO010; best defined by the **2014-10-03 M4.3** | medium | Strike 114°, dip 70°. Catalog maximum M4.1 on **2014-10-03**, the ComCat M4.3 event (41.90°N, 126.18°W; strike 117°, dip 79°, rake 162°, Δ3°, right-lateral). Burst August–October 2014. | as GO010 |
| GO012 | Non-transform offset between Escanaba Trough and the segment to the north (axis jog near 41°35′N) | medium | Trace (41.53°N, 127.44°W) to (41.66°N, 127.31°W) connects the Escanaba axis (127.40°W) to the next axis (127.19°W). Strike 37° is oblique to both, which fits a transfer or oblique fault in the overlap zone. 18 km long. Burst May–June 2015. The 41°35′N right-lateral offset is documented (Morton et al. 1987). No mapped fault at this scale. | Morton et al. 1987 |
| GO013 | Southern Gorda plate NNE left-lateral fault; source zone of the **2005-06-17 Mw 6.6** | medium-high | Trace 40.55–40.91°N at 126.55–126.76°W, strike 203° (= 023°), dip 78°, 43 km, stable. The 2005-06-17 Mw 6.6 (40.77°N, 126.57°W) has nodal planes at 293° and about 023° (Δ0°). 2005-06-15 aftershocks at 40.83–40.91°N, 126.37–126.49°W (strike 207°) and 2006-03-22 M5.0 (strike 18°) are on the same trend. 2001 M5.3 and 2017 M5.7 (strikes 353–354°) are just to the south. Fits the NE/NNE left-lateral fault family. | Smith et al. 1993; Gulick et al. 2001; Chaytor et al. 2004; Rollins & Stein 2010 |
| GO014 | Central Gorda plate NE left-lateral fault (family that produced the Aug 1991 Mw 7.1) | medium | Trace 41.76–41.97°N at 125.62–125.75°W, strike 205° (= 025°), stable. Mechanisms 2017-08-10 M4.8 (35°, Δ10°, left-lateral) at 41.92°N, 125.67°W and 2019-10-28 M5.2 (50°) nearby. The Aug 1991 Mw 7.1 (41.68°N, 125.86°W) ruptured a SW-striking fault (Velasco et al.). | Velasco et al. 1994; Smith et al. 1993; Chaytor et al. 2004 |
| GO015 | Rupture fault of the **2005-06-15 Mw 7.2** (NE-striking left-lateral) | medium-high | Trace (41.35°N, 126.05°W) to (41.08°N, 126.42°W), strike 226° (= 046°), dip 82°, 43 km, stable. Runs SW from the Mw 7.2 epicentre (41.29°N, 125.95°W) along its first-day aftershock trend. Nodal plane about 047° (Δ1°). 2010–2015 events are late aftershocks or background, not the mainshock sequence. | Rollins & Stein 2010; Chaytor et al. 2004 |
| GO016 | Duplicate of GO002 (Escanaba axis, shallow part) | high (as duplicate) | Strike 191°, 41.12–41.33°N, depth 0–15 km, unstable. Swarm December 2014 near the 2014-12-19 M4.6 (oblique, strike 45°). | as GO002 |
| GO017 | NE conjugate within the GO010/GO011 zone | low-medium | Strike 34°, unstable. A NE left-lateral plane conjugate to the WNW planes; the 2006-03-25 M5.1 (226°) supports it. Probably part of the same conjugate or diffuse zone. | Smith et al. 1993; Chaytor et al. 2004 |
| GO018 | NE fault starting at the 2005 Mw 7.2 epicentre and running toward the Aug 1991 Mw 7.1 epicentre | low | Trace 41.33–41.70°N at 125.78–125.93°W, strike 17°, "dipping zone" (dip 38°, unstable), 37 events, Mmax 2.6. SW end at the 2005 epicentre, NE end near the 1991 epicentre. Strike Δ about 25–30° from the 2005 (047°) and 1991 (043°) planes. The low dip is probably an artefact. Could be the NE continuation of the 2005 rupture or a separate NNE fault. | Velasco et al. 1994; Rollins & Stein 2010 |
| GO019 | Duplicate of GO004 (Mendocino transform near the RTI, shallow part) | high (as duplicate) | Strike 277° (Δ3° from GO004), depth 0–8 km, unstable, peak April 2015. | as GO004 |
| GO020 | **No published counterpart.** Swarm about 10–15 km offshore Brookings, Oregon, near the Gorda slab | none | 124.61–124.77°W at 42.18–42.20°N, strike 282°, near-vertical, 14 km, depth about 1–21 km. Slab2 minus depth = −1.6 km, i.e. within about 2 km of the Slab2 interface (sign convention not checked here). 38 of 46 events are from 2012, peaking in August. ANSS also recorded the swarm: about 10 events M2–3 on 1–2 Aug 2012 and more through Nov 2012 (M3.1 on 2012-10-16), mostly with fixed depths near −0.9 km. Web and Crossref searches found no paper. | Stone et al. 2018 (regional context only) |

Summary of duplicates:
- GO002 and GO016: Escanaba axis.
- GO004, GO006 and GO019: Mendocino transform near the RTI.
- GO005, GO007 and GO009: southern Escanaba axis bend.
- GO010, GO011 and GO017: conjugate zone near 41.85°N, 126.1°W.

The 20 surfaces therefore reduce to about **11 independent structures**.

---

## (c) Possible new or under-documented structures

1. **WNW right-lateral source zone near 41.85°N, 126.1°W (GO010, GO011, GO017).**
   - The seismicity planes strike 114–124°, which favours the WNW plane rather than the NE left-lateral faults that dominate the Gorda models.
   - The site has produced M4–5.7 strike-slip events repeatedly: 2006, 2014, 2016, 2019, 2020 and 2026 (M5.7).
   - I found no named fault for it. It may be a member of the NW-trending right-lateral set (the 1991 July type), or a conjugate cross-fault in a fabric-parallel system.
   - Relocating the 2026 M5.7 sequence would test this.
2. **Brookings offshore swarm, 2012 (GO020).**
   - Near-slab-interface swarm on the inner margin, recorded by ANSS and v4. I found no publication.
   - Worth examining as possible slab or interface seismicity, or as a fluid-driven burst.
3. **Oblique transfer fault at the Escanaba–Central offset (GO012), about 41.6°N.** The offset itself is documented; a seismically active oblique fault crossing it is not.
4. **SE-bending southern Escanaba axis at the RTI (GO005, GO007, GO009).**
   - Normal faults striking about 335° rather than N–S at 40.5–40.8°N.
   - The RTI geometry is mapped (Sverdrup, 1987, relocations; USGS sidescan sonar), but no paper ties a NNW-trending active fault zone there to the seismicity.
5. **Deep (20–35 km) seismicity at the RTI and on the transform (GO001, GO004–GO009).** Independently noted by Ren et al. (2025, EGU abstract: mantle shear zones along the Mendocino transform). It needs relocation before interpretation because of v4's absolute depth bias.

Pseudofaults and propagator scars: no v4 surface follows a pseudofault trace. The 42–42.5°N pseudofault limit of Ren et al. (2025) has seismicity in ComCat (2021 M5.9 doublet, 2025 M5.7, both NE-striking), but those events fall after the catalog period.

---

## (d) References

Verification: "Crossref" means the title, authors, year and venue were confirmed through `api.crossref.org/works/{DOI}` on 2026-11-22.

| Reference | DOI | Status |
|---|---|---|
| Chadwick, W.W., Embley, R.W., Shank, T.M. (1998). The 1996 Gorda Ridge eruption: geologic mapping, sidescan sonar, and SeaBeam comparison results. Deep-Sea Res. II 45, 2547–2569 | 10.1016/s0967-0645(98)00083-6 | Crossref |
| Chaytor, J.D., Goldfinger, C., Dziak, R.P., Fox, C.G. (2004). Active deformation of the Gorda plate: constraining deformation models with new geophysical data. Geology 32, 353–356 | 10.1130/g20178.2 | Crossref (no abstract available; content taken from secondary citations) |
| Chen, X., McGuire, J.J. (2016). Measuring earthquake source parameters in the Mendocino triple junction region using a dense OBS array: implications for fault strength variations. EPSL 453, 276–287 | 10.1016/j.epsl.2016.08.022 | Crossref (abstract not read; content from a 2014 AGU abstract) |
| Clague, D.A., Paduan, J.B., Caress, D.W., McClain, J., Zierenberg, R.A. (2020). Lava flows erupted in 1996 on North Gorda Ridge segment and the geology of the nearby Sea Cliff hydrothermal vent field from 1-m resolution AUV mapping. Front. Mar. Sci. 7 | 10.3389/fmars.2020.00027 | Crossref plus publisher page |
| Davis, A.S., Clague, D.A. (2000). President Jackson Seamounts, northern Gorda Ridge: tectonomagmatic relationship between on- and off-axis volcanism. JGR 105, 27939–27956 | 10.1029/2000jb900291 | Crossref |
| Dziak, R.P., Fox, C.G., Bobbitt, A.M., Goldfinger, C. (2001). Bathymetric map of the Gorda plate: structural and geomorphological processes inferred from multibeam surveys. Mar. Geophys. Res. 22, 235–250 | 10.1023/a:1014606407111 | Crossref |
| Fox, C.G., Dziak, R.P. (1998). Hydroacoustic detection of volcanic activity on the Gorda Ridge, February–March 1996. Deep-Sea Res. II 45, 2513–2530 | 10.1016/s0967-0645(98)00081-2 | Crossref |
| Fox, C.G., Dziak, R.P. (1999). Internal deformation of the Gorda plate observed by hydroacoustic monitoring. JGR 104, 17603–17615 | 10.1029/1999jb900104 | Crossref |
| Gulick, S.P.S., Meltzer, A.S., Henstock, T.J., Levander, A. (2001). Internal deformation of the southern Gorda plate: fragmentation of a weak plate near the Mendocino triple junction. Geology 29, 691–694 | 10.1130/0091-7613(2001)029<0691:IDOTSG>2.0.CO;2 | Crossref |
| Guo, H., Atterholt, J., McGuire, J.J., Thurber, C. (2024 online; vol. 96 is 2025). Evidence for low effective stress within the crust of the subducted Gorda plate from the 2022 December Mw 6.4 Ferndale earthquake sequence. SRL 96, 1504–1520 | 10.1785/0220240078 | Crossref (event is outside this region; context only) |
| Henstock, T.J., Levander, A. (2003). Structure and seismotectonics of the Mendocino triple junction, California. JGR 108 | 10.1029/2001jb000902 | Crossref (context) |
| Hsu, Y.-F., Martínez-Garzón, P., Trugman, D.T., Ben-Zion, Y. (2026). Earthquake clustering and stress inversion in the Mendocino Triple Junction region. GJI 245 | 10.1093/gji/ggag089 | Crossref (title, authors and summary checked; first-author initials not checked) |
| Ismat, Z., Putera, H., Patzkowsky, S. (2022). Internal deformation of the Gorda plate and its tectonic significance within the Cascadia subduction zone. J. Struct. Geol. 161, 104643 | 10.1016/j.jsg.2022.104643 | Crossref |
| Morton, J.L., Normark, W.R., Ross, S.L., et al. (1987). Preliminary report, cruises L1-86-NC and L2-86-NC, Escanaba Trough, Gorda Ridge. USGS OFR 87-375A | 10.3133/ofr87375a | Crossref (USGS report, not peer reviewed); source of the 40°25′N–41°35′N Escanaba extent |
| Ren, Y., Lange, D., Grevemeyer, I. (2025). Intraplate deformation and mantle shear zones in the Gorda plate driven by the northward migration of the Mendocino triple junction. EGU25-17345 | 10.5194/egusphere-egu25-17345 | Crossref (conference abstract, not peer reviewed; no journal version found) |
| Riddihough, R.P. (1980). Gorda plate motions from magnetic anomaly analysis. EPSL 51, 163–170 | 10.1016/0012-821x(80)90263-0 | Crossref |
| Rollins, J.C., Stein, R.S. (2010). Coulomb stress interactions among M≥5.9 earthquakes in the Gorda deformation zone and on the Mendocino Fault Zone, Cascadia subduction zone, and northern San Andreas Fault. JGR 115 | 10.1029/2009jb007117 | Crossref |
| Silver, E.A. (1971). Tectonics of the Mendocino triple junction. GSA Bull. 82, 2965–2978 | 10.1130/0016-7606(1971)82[2965:TOTMTJ]2.0.CO;2 | Crossref |
| Smith, S.W., Knapp, J.S., McPherson, R.C. (1993). Seismicity of the Gorda plate, structure of the continental margin, and an eastward jump of the Mendocino triple junction. JGR 98, 8153–8171 | 10.1029/93jb00026 | Crossref. The prompt's "Smith et al. 1993" is this paper; its authors are Smith, Knapp and McPherson. |
| Stoddard, P.R. (1987). A kinematic model for the evolution of the Gorda plate. JGR 92, 11524–11532 | 10.1029/jb092ib11p11524 | Crossref |
| Stoddard, P.R. (1989). Kinematic models of the evolution of the Gorda rise and President Jackson Seamount chain. AGU Geophys. Monogr. 57, 31–37 | 10.1029/gm057p0031 | Crossref |
| Stoddard, P.R. (1991). A comparison of brittle deformation models for the Gorda plate. Tectonophysics 187, 205–214 | 10.1016/0040-1951(91)90420-w | Crossref |
| Stone, I., Vidale, J.E., Han, S., Roland, E. (2018). Catalog of offshore seismicity in Cascadia. JGR 123, 641–652 | 10.1002/2017jb014966 | Crossref |
| Sverdrup, K.A. (1987). Multiple-event relocation of earthquakes near the Gorda Rise–Mendocino Fracture Zone intersection. GRL 14, 347–350 | 10.1029/gl014i004p00347 | Crossref |
| Velasco, A.A., Ammon, C.J., Lay, T. (1994). Recent large earthquakes near Cape Mendocino and in the Gorda plate. JGR 99, 711–728 | 10.1029/93jb02390 | Crossref |
| Wilson, D.S. (1986). A kinematic model for the Gorda Deformation Zone as a diffuse southern boundary of the Juan de Fuca plate. JGR 91, 10259–10269 | 10.1029/jb091ib10p10259 | Crossref |
| Wilson, D.S. (1989). Deformation of the so-called Gorda plate. JGR 94, 3065–3075 | 10.1029/jb094ib03p03065 | Crossref |
| Zierenberg, R.A., et al. (1993). Genesis of massive sulfide deposits on a sediment-covered spreading center, Escanaba Trough, southern Gorda Ridge. Econ. Geol. 88, 2069 | 10.2113/gsecongeo.88.8.2069 | Crossref (title, venue and first author only) |

Not found or not verified:
- **"Gulick et al. 2001", "Chaytor et al. on Gorda Ridge", "Clague et al. on Gorda Ridge volcanism".** Gulick 2001 is listed above. I found no separate Chaytor paper on Gorda Ridge segmentation; Clague's Gorda work is represented by Clague et al. 2020.
- **"Nabelek/Braunmiller Gorda focal mechanisms".** No Gorda-specific paper found. The Oregon State regional moment tensor catalog exists, but no DOI was found. Braunmiller & Nabelek (2008, 10.1029/2007jb005213) covers the Blanco transform, not the Gorda. Unverified for Gorda.
- **"Ghorbani" (Cascadia Initiative OBS Gorda study).** Not found in Crossref. Unverified, possibly a misattribution. Relevant OBS studies are Stone et al. 2018, Chen & McGuire 2016 and Ren et al. 2025 (abstract).
- **"Guo et al. 2024".** The match is the SRL paper on the 2022 Ferndale sequence above, which is outside this region. No 2024 Guo paper on the offshore Gorda plate was found.
- **Finite-fault or aftershock papers for the 2005 Mw 7.2, 2014 Mw 6.8 and 2016 Mw 6.6.** No dedicated peer-reviewed paper verified. Event parameters here come from USGS ComCat (catalog, no DOI).
- **NOAA PMEL pages (1996, 2001 Jackson, 2008 North Gorda swarms).** Web archive only, not peer reviewed.
