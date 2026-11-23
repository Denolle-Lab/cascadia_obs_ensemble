# Existing 3D fault/interface models and seismicity-based fault imaging (agent report, 2026-11-21)

## Existing models
- CRESCENT Community Fault Model v1.0: 329 upper-plate crustal fault sections MTJ→N Vancouver Island + a few lower-plate offshore (Nootka, Mendocino). Trace + 3D triangular mesh + metadata (dip, rake, depth limits, activity 1–3, 3D-constraint source). Status: "under review, not yet published"; data DOI TBD; viewer DOI 10.5281/zenodo.15021860 (Bahavar et al. 2025). Files public, CC BY 4.0: https://github.com/cascadiaquakes/CRESCENT-CFM (last push 2026-11-02); viewer https://cfm.cascadiaquakes.org; docs https://cascadiaquakes.github.io/cfm-book/. Formats: GeoJSON/GPKG/KML traces; meshes GeoJSON/GPKG at full, 5 km, 1 km, low res; tsurf. 3D geometry source: 228 constant-dip extrapolation, 80 trace + 1–2 MCS lines, 21 from SCEC CFM7, 0 from hypocenters. E.g. Seattle (middle) 45° to 35 km; Tacoma 60° to 35 km; Mendocino and S Nootka vertical to 10 km. No Blanco, no Gorda intraplate faults, no JdF ridge/Axial/Endeavour. Repo also has interface meshes: CASIE21 (Carbotte 2024), Slab2, McCrory 2012, Graham 2018 ("uncertain provenance"). Plesch FY24 seed-grant report on merged megathrust t-surf (no released file).
- USGS Qfaults: traces only. 10.5066/P9BCVRCK.
- USGS NSHM23 Fault Sections (v3.0, Dec 2023): traces + dip/rake/depths, no meshes. 10.5066/P9AWINWZ; Hatem et al. 2022 Sci Data 9 506 10.1038/s41597-022-01609-7. Includes Mendocino, not Blanco.
- Watt (2026) USGS offshore Cascadia Quaternary fault traces 10.5066/P14HJMM9 (traces + dip points). Ledeczi, Lucas, Tobin, Watt, Miller 2024 Seismica 2(4) 10.26443/seismica.v2i4.1158.
- WA DNR Earthquakes and Faults v6.0 (Apr 2025): traces. Oregon: Personius et al. 2003 USGS OFR 03-095 traces; no current standalone DOGAMI fault database found.
- SCEC CFM template: Plesch et al. 2007 BSSA 97 1793–1802 10.1785/0120050211.
- Slab2: Hayes et al. 2018 Science 362 58–61 10.1126/science.aat4723; data 10.5066/F7PV6JNV.
- McCrory et al. 2012 JGR 117 B09306 10.1029/2012JB009407; USGS DS 633 (partly seismicity-constrained).
- CASIE21: Carbotte et al. 2024 Sci Adv 10 eadl3198 10.1126/sciadv.adl3198 (mesh in CRESCENT-CFM repo).
- Receiver functions: Bloch, Bostock & Audet 2023 G3 24 e2023GC011088 10.1029/2023GC011088 (files not located).
- Blanco, Gorda intraplate, JdF ridge, Endeavour, Axial ring faults: no community 3D model or vetted traces.

## Methods (fault surfaces from catalogs)
1. Ouillon, Ducorbier & Sornette 2008 JGR 113 B01306 10.1029/2007JB005032 — anisotropic clustering of planes.
2. Wang, Ouillon, Woessner, Sornette & Husen 2013 JGR 118 5956–5975 10.1002/2013JB010164 — with location uncertainty.
3. Kamer, Ouillon & Sornette 2020 NHESS 20 3611–3625 10.5194/nhess-20-3611-2020 — agglomerative.
4. Riesner, Durand-Riard, Hubbard, Plesch & Shaw 2017 SRL 88 831–839 10.1785/0220160192 — implicit modeling with traces, nodal planes, hypocenters.
5. Plesch, Shaw, Ross & Hauksson 2020 BSSA 110 1818–1831 10.1785/0120200053 — Ridgecrest for SCEC CFM.
6. Ross, Cochran, Trugman & Smith 2020 Science 368 1357–1361 10.1126/science.abb0779 — Cahuilla swarm (ML catalog).
7. Truttmann, Diehl & Herwegh 2023 JGR 128 e2023JB026352 10.1029/2023JB026352 — NN + PCA.
8. Piegari, Camanni, Mercurio & Marzocchi 2024 ESS 11 e2023EA003267 10.1029/2023EA003267 — DBSCAN/OPTICS + PCA.
9. Alongi, Elliott, Skoumal, Shelly & Hatem 2025 SRL 97 1174–1190 10.1785/0220250126 — SURF (USGS Python): clustering, orientation merging, SVR nonplanar quad meshes. Most usable tool.
Context: Tan et al. 2021 Seismic Record 1 11–19 10.1785/0320210001; Skoumal, Kaven & Walter 2019 SRL 10.1785/0220190045. No peer-reviewed HDBSCAN+PCA fault-surface paper.

## Seismicity-based imaging by region
- Blanco: Braunmiller & Nábělek 2008; Kuna 2019; Ren et al. 2023; Lange et al. (agent saw arXiv 2503.17148 preprint; the Blanco agent found the 2026 JGR version 10.1029/2025JB031584).
- Mendocino/Gorda: Yoon & Shelly 2024; Guo, Atterholt, McGuire & Thurber 2024 SRL 96 1504–1520 10.1785/0220240078; Hsu, Martínez-Garzón, Trugman & Ben-Zion 2026 GJI 245 ggag089 10.1093/gji/ggag089 (clustering + stress inversion); Chaytor 2004 (DOI suffix .2 per Crossref).
- Puget/Vancouver Island: Blakely 2002; Van Wagoner et al. 2002 JGR 107(B12) 10.1029/2001JB000710; Li, Liu, Regalla & Morell 2018 JGR 123 2841–2855 10.1002/2017JB015021 (Leech River). No seismicity-imaged SWIF/Tacoma geometry found.
- Axial: Wilcock 2016; Waldhauser 2020 (~100,000 events, <200 m wide outward-dipping ring faults, figure-8); Wang 2024.
- Endeavour: Krauss 2023, 2025; MGDS catalog https://www.marine-geo.org/tools/files/30498. No fault-surface models.

## Recommendation (agent)
Best pilot: Mendocino/Gorda (most active; land + OBS; CFM has only a vertical Mendocino to 10 km and no Gorda faults; independent catalogs Yoon & Shelly 2024, Hsu 2026; SURF). Second: Blanco (absent from all fault models; independent ML catalogs). Lower: Axial/Endeavour (denser cabled catalogs exist), Puget (diffuse).
Adds: first hypocenter-constrained 3D surfaces for offshore plate-boundary/intraplate faults (0% of CFM geometries come from hypocenters); test of Slab2/McCrory/CASIE21 where they disagree. Needs DD relocation + per-event uncertainty before plane fitting.
