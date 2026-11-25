# Station-term test sets (b4_v6_termtest_sets.py)

| set | rule | events | pairs | lat | lon | depth median (km) | radius (km) | tiers | families |
|---|---|---|---|---|---|---|---|---|---|
| puget | Puget/Georgia density peak (8 km count 1025 at 47.771, -122.832); 1000 region events nearest the median epicentre of the 1145 events within 10 km | 1,000 | 2,423 | 47.768 | -122.850 | 17.48 | 7.7 | {'A': 176, 'B': 390, 'C': 362, 'D': 72} | {-1: 711, 1027: 201, 2048: 46, 1578: 13} |
| mtj | ZBZ family 1536 (682 located events): the 600 nearest its median epicentre | 600 | 521 | 40.304 | -124.544 | 22.52 | 17.1 | {'A': 256, 'B': 147, 'C': 92, 'D': 105} | {1536: 600} |
| blancow | ZBZ family 427 (1447 located events): the 1200 nearest its median epicentre | 1,200 | 3,099 | 44.159 | -129.453 | 8.05 | 4.7 | {'B': 30, 'C': 1073, 'D': 97} | {427: 1200} |

Events: ID, v5.1 origin time and hypocentre (km bsl), region, tier. Pairs: B5 prep catalog-dt pairs among them (10 nearest neighbours within 10 km, >= 8 common station-phases). No event is in the D-B078 sets (axial, blanco, msh, jdf). Family -1 = background (no ZBZ family).
