"""Helpers to evaluate AxialPolCap on real P windows (see the Axial test notebook).

Windows come from axial_test_extract.py: raw counts, demeaned, 100 Hz, P at
sample `p_index` (300 = 3 s), 5 s long.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, resample_poly, sosfilt, sosfiltfilt

HALF = 32


def load_windows(path):
    z = np.load(path, allow_pickle=False)
    meta = pd.read_json(io.StringIO(str(z["meta"])), orient="split")
    return z["windows"].astype(np.float64), float(z["sr"]), int(z["p_index"]), meta


def bandpass(x, sr, lo=3.0, hi=20.0, order=4, zero_phase=True):
    """Demean, linear detrend, taper 5 %, Butterworth band-pass (FM_ML: 3-20 Hz).
    zero_phase=True mimics MATLAB filtfilt (assumed; UNKNOWN which FM_ML used)."""
    x = detrend(x - x.mean(axis=-1, keepdims=True), axis=-1, type="linear")
    n = x.shape[-1]
    k = int(0.05 * n)
    taper = np.ones(n)
    taper[:k] = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
    taper[-k:] = taper[:k][::-1]
    x = x * taper
    if lo is None and hi is None:
        return x
    if lo is None:
        sos = butter(order, hi, btype="lowpass", fs=sr, output="sos")
    elif hi is None:
        sos = butter(order, lo, btype="highpass", fs=sr, output="sos")
    else:
        sos = butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x, axis=-1) if zero_phase else sosfilt(sos, x, axis=-1)


def crop(x, p_index, shift=0, half=HALF):
    """64-sample crop with P at sample 32. shift>0 moves the window later
    (equivalent to a pick that is `shift` samples late)."""
    c = p_index + shift
    return x[..., c - half:c + half]


def as_200hz_reading(xf, p_index):
    """Interpretation B: model expects 200 Hz, 64 samples = +-0.16 s.
    Upsample the filtered 100 Hz trace by 2 and crop 64 samples around P."""
    up = resample_poly(xf, 2, 1, axis=-1)
    return crop(up, 2 * p_index)


def entropy(p_up):
    p = np.clip(np.stack([1 - p_up, p_up], -1), 1e-12, 1)
    return -(p * np.log(p)).sum(-1)


def snr_db(xf, p_index, sr, noise=(-1.0, -0.1), signal=(0.0, 0.5)):
    n0, n1 = (p_index + int(t * sr) for t in noise)
    s0, s1 = (p_index + int(t * sr) for t in signal)
    rn = np.sqrt(np.mean(xf[..., n0:n1] ** 2, -1))
    rs = np.sqrt(np.mean(xf[..., s0:s1] ** 2, -1))
    return 20 * np.log10(np.maximum(rs, 1e-20) / np.maximum(rn, 1e-20))


def first_motion_sign(xf, p_index, sr, k=4.0, search=(-0.05, 0.25), noise=(-1.0, -0.1)):
    """Model-free first-motion reference.
    On the filtered trace, find the first sample in `search` (s, relative to
    the pick) where |x| exceeds k * noise std, then take the sign of the first
    local extremum from there. Returns +1 (up), -1 (down), 0 (not found) and
    the onset offset in samples relative to the pick."""
    n0, n1 = (p_index + int(t * sr) for t in noise)
    s0, s1 = (p_index + int(t * sr) for t in search)
    sig = xf[..., n0:n1].std(-1)
    out = np.zeros(xf.shape[0], dtype=int)
    onset = np.full(xf.shape[0], np.nan)
    for i in range(xf.shape[0]):
        seg = xf[i, s0:s1]
        idx = np.flatnonzero(np.abs(seg) > k * sig[i])
        if not len(idx):
            continue
        j = idx[0]
        s = np.sign(seg[j])
        # walk to the first extremum of that lobe
        while j + 1 < len(seg) and np.sign(seg[j + 1]) == s and abs(seg[j + 1]) >= abs(seg[j]):
            j += 1
        out[i] = int(s)
        onset[i] = s0 + idx[0] - p_index
    return out, onset
