"""ELEP ensemble picker for one station-day.

Consolidates the three ``picking_utils*.py`` variants that produced the ver3 picks
(the originals are in ``legacy/``, and in git at tag ``pre-cleanup-2026-09``).
They differed only in how they chose channels; the ELEP core (5
EQTransformer models, semblance, stacking, 0.05 trigger) was identical. The
choices are now arguments, set per run in ``picking_config.csv``:

``channel_mode``
    ``'hh_bh_eh'``  HH if present, else BH, else EH           (picking_utils, _prio)
    ``'eh_only'``   EH only; a Z-only station gets Z copied to
                    all three inputs                           (picking_utils_prio_EH)
``vertical``
    regex for the vertical component: ``'Z'`` (picking_utils) or ``'[3Z]'`` (_prio*)
``skip_if_in``
    another output directory; a station-day already written there is skipped
    (picking_utils_prio skipped station-days of the 122-129 run).
"""
import gc
import logging
import os
import sys
import time

import numpy as np
import obspy
import pandas as pd
import seisbench.models as sbm
import torch
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import trigger_onset
from ELEP.elep.ensemble_coherence import ensemble_semblance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.data_client import get_waveforms  # noqa: E402

Logger = logging.getLogger(__name__)

device = torch.device("cpu")

# ELEP parameters (samples at 100 Hz)
TWIN = 6000                # model input length
STEP = 3000                # window step
L_BLND, R_BLND = 500, 500  # samples blinded on each side of a window
P_THRD, S_THRD = 0.05, 0.05
PRETRAIN_LIST = ['original', 'ethz', 'instance', 'scedc', 'stead']
PARAS_SEMBLANCE = {'semblance_order': 2, 'window_flag': True,
                   'semblance_win': 0.5, 'weight_flag': 'max'}

PICK_COLUMNS = ['network', 'station', 'location', 'band_inst', 'label', 'trace_starttime',
                'trigger_onset', 'pick_time', 'trigger_offset', 'max_prob', 'thresh_prob']


def pred_trigger_pick(pred, source_trace, label, thrd=0.1, **kwargs):
    """Single-threshold trigger on an ensemble prediction; one pick per trigger at its maximum."""
    if len(pred) < 300:
        raise ValueError('insufficient samples in pred')
    if not isinstance(source_trace, Trace):
        raise TypeError('source_trace must be type obspy.Trace')
    pred[:300] = 0.
    triggers = trigger_onset(pred, thrd, thrd, **kwargs)
    t0 = source_trace.stats.starttime
    sr = source_trace.stats.sampling_rate
    tid = source_trace.id
    picks = []
    for s0, s1 in triggers:
        to = t0 + s0 / sr
        tp = t0 + (s0 + np.argmax(pred[s0:s1 + 1])) / sr
        tf = t0 + s1 / sr
        pv = np.max(pred[s0:s1 + 1])
        picks.append(tid[:-1].split('.') + [label, t0, to, tp, tf, pv, thrd])
    return pd.DataFrame(data=picks, columns=PICK_COLUMNS)


def stacking(data, npts, l_blnd, r_blnd, nseg, twin=TWIN, step=STEP):
    """Stack overlapping window predictions by their maximum, ignoring blinded samples."""
    _data = data.copy()
    stack = np.full(npts, np.nan, dtype=np.float32)
    _data[:, :l_blnd] = np.nan
    _data[:, -r_blnd:] = np.nan
    stack[:twin] = _data[0, :]
    for iseg in range(nseg - 1):
        idx = step * (iseg + 1)
        stack[idx:idx + twin] = np.nanmax([stack[idx:idx + twin], _data[iseg + 1, :]], axis=0)
    return stack


def select_channels(st, channel_mode='hh_bh_eh', vertical='Z'):
    """Return the stream to pick on (possibly empty), following ``channel_mode``."""
    sdata = Stream()
    has_Z = bool(st.select(channel=f'??{vertical}'))
    if not has_Z:
        Logger.warning('No Vertical Component Data Present. Skipping')
        return sdata
    if channel_mode == 'hh_bh_eh':
        for band in ('HH', 'BH', 'EH'):
            if st.select(channel=f'{band}?'):
                sdata += st.select(channel=f'{band}?')
                break
    elif channel_mode == 'eh_only':
        sdata += st.select(channel='EH?')
    else:
        raise ValueError(f"unknown channel_mode {channel_mode!r}")
    return sdata


def order_components(sdata, raw, channel_mode='hh_bh_eh', vertical='Z'):
    """Order traces Z, E/1, N/2 (the first of each; extras appended).

    In ``eh_only`` mode a station whose raw stream ``raw`` lacks either
    horizontal (tested on any band, as picking_utils_prio_EH did) gets its Z
    trace copied twice instead.
    """
    if channel_mode == 'eh_only':
        has_N = bool(raw.select(channel='??[2N]'))
        has_E = bool(raw.select(channel='??[1E]'))
        if not (has_N and has_E):
            z_trace = sdata.select(channel='??Z')[0]
            sdata += z_trace.copy()
            sdata += z_trace.copy()
            return sdata
    _s2d, _s2x = Stream(), Stream()
    for _c in [vertical, '[1E]', '[2N]']:
        for _e, _tr in enumerate(sdata.select(channel=f'??{_c}')):
            if _e == 0:
                _s2d += _tr
            else:
                _s2x += _tr
    return _s2d + _s2x


def output_name(outdir, network, station, t1, t2):
    return os.path.join(outdir, f"{network}_{station}_{t1.strftime('%Y%m%d')}_{t2.strftime('%Y%m%d')}.csv")


def run_detection(network, station, t1, t2, outdir, channel_mode='hh_bh_eh', vertical='Z',
                  skip_if_in=None, source='pnwstore', models=None):
    """Pick one station-day and write ``outdir/NET_STA_YYYYMMDD_YYYYMMDD.csv``.

    Returns the file name written, or None when the station-day is skipped.
    ``models`` is an optional dict {name: EQTransformer} to avoid reloading
    the weights on every call.
    """
    save_file_name = output_name(outdir, network, station, t1, t2)
    # Safety catch against overwriting previous analyses
    if os.path.exists(save_file_name):
        Logger.info(f'File {save_file_name} already exists')
        return None
    if skip_if_in and os.path.exists(output_name(skip_if_in, network, station, t1, t2)):
        Logger.info(f'{network}.{station} {t1} already picked in {skip_if_in}')
        return None

    # NTS: This sampling scheme leans heavily on prior data QC going into PNW store
    try:
        _sdata = get_waveforms(network=network, station=station, channel='*',
                               starttime=UTCDateTime(t1), endtime=UTCDateTime(t2),
                               source=source)
    except obspy.clients.fdsn.header.FDSNNoDataException:
        Logger.warning(f"No data for {network}.{station} on {t1} - {t2}.")
        return None

    sdata = select_channels(_sdata, channel_mode, vertical)
    if len(sdata) == 0:
        Logger.warning("No stream returned. Skipping.")
        return None
    if np.abs(np.mean(sdata[0].data[1:] - sdata[0].data[0:-1])) <= 1e-8:
        Logger.warning("constant/no data in the stream. Skipping.")
        return None

    sdata.filter(type='bandpass', freqmin=4, freqmax=15)
    sdata.resample(100)
    # NTS: This may produce unintended swathes of filled gaps
    sdata.merge(fill_value='interpolate')

    dt = 1 / sdata[0].stats.sampling_rate

    # Make all the traces the same length
    max_starttime = max([tr.stats.starttime for tr in sdata])
    min_endtime = min([tr.stats.endtime for tr in sdata])
    for tr in sdata:
        tr.trim(starttime=max_starttime, endtime=min_endtime, nearest_sample=True)

    sdata = order_components(sdata, _sdata, channel_mode, vertical)

    # Reshape into overlapping windows
    arr_sdata = np.array(sdata)
    npts = arr_sdata.shape[1]
    nseg = int(np.floor((npts - TWIN) / STEP)) + 1
    tap = 0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, 6)))
    windows = np.zeros(shape=(nseg, 3, TWIN), dtype=np.float32)
    windows_std = np.zeros(shape=(nseg, 3, TWIN), dtype=np.float32)
    windows_max = np.zeros(shape=(nseg, 3, TWIN), dtype=np.float32)
    for iseg in range(nseg):
        idx = iseg * STEP
        windows[iseg, :] = arr_sdata[:, idx:idx + TWIN]
        windows[iseg, :] -= np.mean(windows[iseg, :], axis=-1, keepdims=True)
        # 'original' uses std normalization, the others max normalization
        windows_std[iseg, :] = windows[iseg, :] / np.std(windows[iseg, :]) + 1e-10
        windows_max[iseg, :] = windows[iseg, :] / (np.max(np.abs(windows[iseg, :]), axis=-1, keepdims=True))
    windows_std[:, :, :6] *= tap
    windows_std[:, :, -6:] *= tap[::-1]
    windows_max[:, :, :6] *= tap
    windows_max[:, :, -6:] *= tap[::-1]
    del windows

    # Predict on the base models. dim 0: 0 = P, 1 = S (EQTransformer output 0 is detection)
    batch_pred = np.zeros([2, len(PRETRAIN_LIST), nseg, TWIN], dtype=np.float32)
    for ipre, pretrain in enumerate(PRETRAIN_LIST):
        t0 = time.time()
        eqt = models[pretrain] if models else load_model(pretrain)
        x = torch.Tensor(windows_std if pretrain == 'original' else windows_max)
        with torch.no_grad():
            _torch_pred = eqt(x.to(device))
        batch_pred[0, ipre, :] = _torch_pred[1].detach().cpu().numpy()
        batch_pred[1, ipre, :] = _torch_pred[2].detach().cpu().numpy()
        Logger.debug(f"{pretrain}: {time.time() - t0:.1f} s")
    del _torch_pred, x, windows_std, windows_max
    gc.collect()

    # Ensemble semblance per window, then stack
    paras = dict(PARAS_SEMBLANCE, dt=dt)
    smb_pred = np.zeros([2, nseg, TWIN], dtype=np.float32)
    for iseg in range(nseg):
        smb_pred[0, iseg, :] = ensemble_semblance(batch_pred[0, :, iseg, :], paras)
        smb_pred[1, iseg, :] = ensemble_semblance(batch_pred[1, :, iseg, :], paras)
    smb_p = stacking(smb_pred[0, :], npts, L_BLND, R_BLND, nseg)
    smb_s = stacking(smb_pred[1, :], npts, L_BLND, R_BLND, nseg)
    del smb_pred, batch_pred

    idf_p = pred_trigger_pick(smb_p, sdata[0], 'P', thrd=P_THRD)
    idf_s = pred_trigger_pick(smb_s, sdata[0], 'S', thrd=S_THRD)
    df = pd.concat([idf_p, idf_s], axis=0, ignore_index=True)
    df.to_csv(save_file_name)
    return save_file_name


def load_model(pretrain):
    """EQTransformer with the ELEP overlap and blinding settings."""
    eqt = sbm.EQTransformer.from_pretrained(pretrain)
    eqt.to(device)
    eqt._annotate_args['overlap'] = ('Overlap between prediction windows in samples '
                                     '(only for window prediction models)', STEP)
    eqt._annotate_args['blinding'] = ('Number of prediction samples to discard on '
                                      'each side of each window prediction', (L_BLND, R_BLND))
    eqt.eval()
    return eqt


def load_models():
    return {p: load_model(p) for p in PRETRAIN_LIST}
