# M0 coarse whole-margin inversion (margin_m0_v2)

dlnV = change from the start (D-B005 model averaged over 31 km boxes), interior nodes; % of nodes changed by more than 0.5 %; rms over nodes changed.

| tag | depth km | Vp nodes changed % | rms dlnVp changed % | Vs nodes changed % | rms dlnVs changed % |
|---|---|---|---|---|---|
| d1000_s20 | 7 | 44 | 8.78 | 38 | 4.98 |
| d1000_s20 | 15 | 45 | 6.37 | 42 | 4.00 |
| d1000_s20 | 25 | 42 | 5.76 | 39 | 3.85 |
| d1000_s20 | 40 | 36 | 5.36 | 30 | 3.68 |
| d200_s20 | 7 | 47 | 13.22 | 45 | 8.21 |
| d200_s20 | 15 | 47 | 9.00 | 46 | 6.32 |
| d200_s20 | 25 | 46 | 8.26 | 44 | 5.55 |
| d200_s20 | 40 | 41 | 8.60 | 37 | 5.92 |

## Runs

| tag | damp | rms_wt_final | cnd_first | cnd_last | aq_repeats | events | rms_dlnvp_pct | rms_dlnvs_pct | shift_h_med_km | shift_z_abs_med_km |
|---|---|---|---|---|---|---|---|---|---|---|
| d1000_s20 | 1000 | 0.223 | 39.3 | 14 | 7 | 6925 | 4.162 | 2.206 | 0.8973 | 0.7898 |
| d200_s20 | 200 | 0.201 | 338.3 | 115.6 | 11 | 6899 | 6.958 | 3.966 | 1.772 | 2.157 |
