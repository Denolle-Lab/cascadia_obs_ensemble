# Catalog v4: changes from the files as received

Source: `Cascadia_relocated_file_ver_1.csv`, `Cascadia_picks_ver_1.csv` (I. McBrearty, received 2026-11-21).
Built with `utils/fix_catalog_v4.py`.

- 21 events and their 339 picks moved one day later (origin time and pick times; travel times unchanged). They were dated one day early: origin a few seconds before midnight, picks just after it. After the shift every one of their picks matches a ver6 pick exactly.
  Event IDs: 2877, 18916, 26927, 28027, 44216, 45770, 54911, 71159, 74839, 78814, 79142, 79237, 82525, 82760, 82950, 83004, 106164, 106234, 117862, 149294, 171280.

Nothing else is changed. The format is still that of the files as received; the conversion to the origins/arrivals/picks standard of the v3 record comes later.

## v4.1 (2026-11-22): 2016 offshore events

Files in `fixes/m66_2016/` (README there). The 2016 Mw 6.6 Mendocino earthquake and three other M 4.7-5.1 offshore events of 2016 were missing or poorly picked in v4, because ver6 has no NC/BK picks in 2016. Their waveforms were downloaded again, repicked with the same ELEP ensemble and associated to the ComCat origins. Our NLLoc epicentres are biased west (no station west of the events, nearest at 156 km) and their depths are unresolved, so the added rows carry the ComCat hypocentre and magnitude.

- 3 events added (283 picks; `raw_ID`, `detection_value` and `amplitude(mm)` are nan; `spatial_sigma(m)` is the distance from our NLLoc 1D epicentre to ComCat):
  - 900001: 2016-12-08T14:49:45, 40.4535, -126.1936, 8.45 km, M 6.6, 146 P + 54 S; ComCat us20007z6r (Mww 6.6); NLLoc 1D epicentre 26.2 km away
  - 900002: 2016-01-30T00:28:41, 40.9167, -127.2074, 10.0 km, M 5.1, 42 P + 29 S; ComCat us20004vq3 (Mww 5.1); NLLoc 1D epicentre 9.4 km away
  - 900003: 2016-01-30T00:30:52, 41.0363, -127.3842, 10.0 km, M 5.0, 11 P + 1 S; ComCat us20004vq9 (Mww 5.0); NLLoc 1D epicentre 20.2 km away
- Event 172114 (M 4.7 aftershock, us20007z7b) keeps its v4 hypocentre; repicks added on station-phases it lacked (172114: +103 P, +57 S), travel times from the v4 origin time.
- Event 172113 and its 5 picks removed (1 event): its P picks are the Pn of the Mw 6.6 at 708-797 km, within 2 s of iasp91.
- `magnitude_v4/cascadia_catalog_M_routeA.csv`: 172113 removed, the 3 new events added as Mw (ComCat Mww).

The added picks are not ver6 picks (no `pick_id`, no Route A amplitude).
