#!/usr/bin/env python3
"""AxialPolCap (FM_ML, Zhang et al., in prep. SRL) ported from Keras 3 to PyTorch.

Source model: FM_ML/06-models/PolarPicker_h5_20260308_154910.keras
(https://github.com/maochuan-zhang-UW/FM_ML). A .keras file is a zip with
config.json + model.weights.h5; we read the weights with h5py, so neither
TensorFlow nor Keras is needed at inference time.

Architecture (PolarCAP, Chakraborty et al. 2022, retrained on Axial OOI data):
  encoder : Conv1D(32,k32,relu) - Dropout(0.3) - BN - MaxPool(2)
            Conv1D(8,k16,relu)  - BN - MaxPool(2)          -> (16, 8) latent
  decoder : Conv1D(8,k16,tanh) - BN - Up(2) - Conv1D(32,k32,relu) - BN - Up(2)
            - Conv1D(1,k64,tanh)                            -> reconstruction
  head    : Flatten (channels_last order) - Dense(2, softmax)

Input contract (from FM_ML application scripts): vertical component, 100 Hz,
64 samples from P-32 to P+31 samples, each trace divided by max|x|.
Output: softmax probabilities [p_down, p_up]; class 0 = down (-1), 1 = up (+1).
FM_ML keeps a polarity when max(p) >= 0.80.

Usage:
  python axialpolcap_torch.py convert MODEL.keras OUT.pt
  python axialpolcap_torch.py selftest MODEL.keras   # random-input sanity checks
"""
from __future__ import annotations

import argparse
import io
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_SAMPLES = 64
SAMPLING_RATE = 100.0
P_INDEX = 32  # sample index of the P pick inside the 64-sample window
CLASS_TO_POLARITY = (-1, 1)
DEFAULT_THRESHOLD = 0.80


class SameConv1d(nn.Conv1d):
    """Conv1d with Keras/TF 'same' padding (extra sample on the right for even k)."""

    def forward(self, x):
        k = self.kernel_size[0]
        left = (k - 1) // 2
        return F.conv1d(F.pad(x, (left, k - 1 - left)), self.weight, self.bias)


def _bn(c):
    return nn.BatchNorm1d(c, eps=1e-3, momentum=0.01)


class AxialPolCap(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1 = SameConv1d(1, 32, 32)
        self.enc_drop = nn.Dropout(0.3)
        self.enc_bn1 = _bn(32)
        self.enc_conv2 = SameConv1d(32, 8, 16)
        self.enc_bn2 = _bn(8)
        self.dec_conv1 = SameConv1d(8, 8, 16)
        self.dec_bn1 = _bn(8)
        self.dec_conv2 = SameConv1d(8, 32, 32)
        self.dec_bn2 = _bn(32)
        self.dec_conv3 = SameConv1d(32, 1, 64)
        self.dense = nn.Linear(16 * 8, 2)

    def encode(self, x):  # x: (N, 1, 64)
        x = self.enc_bn1(self.enc_drop(F.relu(self.enc_conv1(x))))
        x = F.max_pool1d(x, 2)
        x = self.enc_bn2(F.relu(self.enc_conv2(x)))
        return F.max_pool1d(x, 2)  # (N, 8, 16)

    def decode(self, z):
        x = self.dec_bn1(torch.tanh(self.dec_conv1(z)))
        x = x.repeat_interleave(2, dim=-1)
        x = self.dec_bn2(F.relu(self.dec_conv2(x)))
        x = x.repeat_interleave(2, dim=-1)
        return torch.tanh(self.dec_conv3(x))

    def classify(self, z):
        # Keras Flatten on channels_last (16, 8): index = t * 8 + c
        return F.softmax(self.dense(z.transpose(1, 2).reshape(z.shape[0], -1)), dim=-1)

    def forward(self, x, with_reconstruction=False):
        """x: (N, 64) or (N, 1, 64), already max-normalized. Returns probs (N, 2)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.encode(x)
        p = self.classify(z)
        return (p, self.decode(z)) if with_reconstruction else p


# Keras weight paths -> (torch module name, kind)
_MAP = {
    "layers/sequential/layers/conv1d": ("enc_conv1", "conv"),
    "layers/sequential/layers/batch_normalization": ("enc_bn1", "bn"),
    "layers/sequential/layers/conv1d_1": ("enc_conv2", "conv"),
    "layers/sequential/layers/batch_normalization_1": ("enc_bn2", "bn"),
    "layers/sequential_1/layers/conv1d": ("dec_conv1", "conv"),
    "layers/sequential_1/layers/batch_normalization": ("dec_bn1", "bn"),
    "layers/sequential_1/layers/conv1d_1": ("dec_conv2", "conv"),
    "layers/sequential_1/layers/batch_normalization_1": ("dec_bn2", "bn"),
    "layers/sequential_1/layers/conv1d_2": ("dec_conv3", "conv"),
    "layers/sequential_2/layers/dense": ("dense", "dense"),
}


def load_keras(path) -> AxialPolCap:
    import h5py

    with zipfile.ZipFile(path) as zf:
        buf = io.BytesIO(zf.read("model.weights.h5"))
    model = AxialPolCap()
    sd = {}
    with h5py.File(buf, "r") as f:
        for kpath, (name, kind) in _MAP.items():
            v = [np.asarray(f[f"{kpath}/vars/{i}"]) for i in range(4 if kind == "bn" else 2)]
            if kind == "conv":  # Keras (k, in, out) -> torch (out, in, k)
                sd[f"{name}.weight"] = torch.from_numpy(v[0].transpose(2, 1, 0).copy())
                sd[f"{name}.bias"] = torch.from_numpy(v[1])
            elif kind == "dense":  # Keras (in, out) -> torch (out, in)
                sd[f"{name}.weight"] = torch.from_numpy(v[0].T.copy())
                sd[f"{name}.bias"] = torch.from_numpy(v[1])
            else:  # gamma, beta, moving_mean, moving_variance
                for key, arr in zip(("weight", "bias", "running_mean", "running_var"), v):
                    sd[f"{name}.{key}"] = torch.from_numpy(arr)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"weight mapping mismatch: missing={missing} unexpected={unexpected}")
    return model.eval()


def normalize(x: np.ndarray) -> np.ndarray:
    """Per-trace max-abs normalization, as in FM_ML (zero traces left as zero)."""
    m = np.max(np.abs(x), axis=-1, keepdims=True)
    m[m == 0] = 1
    return x / m


@torch.no_grad()
def predict(model: AxialPolCap, windows: np.ndarray, threshold=DEFAULT_THRESHOLD, batch=4096):
    """windows: (N, 64) raw Z windows (P at index 32). Returns dict of arrays."""
    x = torch.from_numpy(normalize(np.asarray(windows, dtype=np.float32)))
    probs = torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).numpy()
    conf = probs.max(axis=1)
    pol = np.where(probs[:, 1] > probs[:, 0], 1, -1)
    return {
        "p_up": probs[:, 1],
        "p_down": probs[:, 0],
        "confidence": conf,
        "polarity": np.where(conf >= threshold, pol, 0).astype(np.int8),
    }


def _selftest(path):
    m = load_keras(path)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, N_SAMPLES)).astype(np.float32)
    out = predict(m, x)
    # 1) flipping the sign of an impulsive arrival should flip the prediction
    t = np.arange(N_SAMPLES)
    pulse = np.where(t >= P_INDEX, np.sin(2 * np.pi * 8 * (t - P_INDEX) / SAMPLING_RATE)
                     * np.exp(-(t - P_INDEX) / 10), 0).astype(np.float32)
    pp = predict(m, np.stack([pulse, -pulse]))
    print("random noise: mean confidence %.3f, kept at 0.8: %d/256"
          % (out["confidence"].mean(), (out["polarity"] != 0).sum()))
    print("up pulse  p_up=%.3f  -> %+d" % (pp["p_up"][0], pp["polarity"][0]))
    print("down pulse p_up=%.3f -> %+d" % (pp["p_up"][1], pp["polarity"][1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert"); c.add_argument("keras"); c.add_argument("out")
    s = sub.add_parser("selftest"); s.add_argument("keras")
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        torch.save(load_keras(a.keras).state_dict(), a.out)
        print("wrote", a.out)
    else:
        _selftest(a.keras)


if __name__ == "__main__":
    main()
