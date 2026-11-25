comcat_nonquake_2010-2016.csv: every ComCat (USGS FDSN event service) event that is not an earthquake (eventtype
explosion, quarry blast, other event, sonic boom, ...), 2010-01-01 to 2017-01-01, latitude 39-51, longitude -131 to
-120; columns type, t (origin time, Unix seconds UTC), lat, lon. Queried 2026-11-24 by the S-pick waveform check
(/wd1/mdenolle_data/tomo_v1/above_surface/check_spicks/scripts/c11_blasts.py, one query per eventtype with ObsPy
Client("USGS")); 3,733 events. Used by b4_catalog.py for the D-B071 event_flag (match within 3 s and 20 km). The local
data/magnitude_v4/anss_2010-16.csv holds only 556 explosions and 31 quarry blasts (its query did not include every
event type), so it is not used for this flag.
