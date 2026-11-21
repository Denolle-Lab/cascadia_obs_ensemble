# Catalog v4: changes from the files as received

Source: `Cascadia_relocated_file_ver_1.csv`, `Cascadia_picks_ver_1.csv` (I. McBrearty, received 2026-11-21).
Built with `utils/fix_catalog_v4.py`.

- 21 events and their 339 picks moved one day later (origin time and pick times; travel times unchanged). They were dated one day early: origin a few seconds before midnight, picks just after it. After the shift every one of their picks matches a ver6 pick exactly.
  Event IDs: 2877, 18916, 26927, 28027, 44216, 45770, 54911, 71159, 74839, 78814, 79142, 79237, 82525, 82760, 82950, 83004, 106164, 106234, 117862, 149294, 171280.

Nothing else is changed. The format is still that of the files as received; the conversion to the origins/arrivals/picks standard of the v3 record comes later.
