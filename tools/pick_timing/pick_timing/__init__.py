"""pick_timing: find and correct timing errors in seismic phase picks.

Two failure modes are handled:

1. Day-merge drift: a picker that reads a whole day of MiniSEED and merges it into one trace times every
   sample by counting from the segment start at the nominal rate. Where the digitizer rate is not the
   nominal rate, the record headers are right and the count is wrong, so picks drift through the day.
   Fix: ``retime`` (each pick re-timed from its own record header). Find it with ``scan-headers``.
2. Station clock error: the record headers themselves are wrong (a free-running clock that drifts until
   a restart re-synchronises it, or a constant offset). Re-timing cannot fix it. Find it with
   ``detect-clock`` (network-relative residuals), confirm with a person (``report`` + ``ingest-review``),
   model it with ``model-clock`` and correct the picks with ``apply``.

Command line: ``python -m pick_timing --help``.
"""

__version__ = "0.1.0"

# sign convention used everywhere in this package:
#   c(t) = clock error = station header time - true time (s). Positive c: the station's picks are late and
#   its travel-time residuals are positive. The correction added to a pick time is -c(t).
SIGN_CONVENTION = ("c(t) = header time - true time (s); positive = picks late; "
                   "clock_correction_s = -c(t) is added to the pick time")
