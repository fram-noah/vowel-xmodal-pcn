"""Toy audio-visual "child-directed song" sequences with known vowel boundaries.

No licensed recordings are shipped with this project (see the repo README),
so this generator stands in for real audio-visual data while the
architecture in ``model.py`` is being built out. Each sequence is a series of
"vowel" segments (a fixed random prototype feature vector per vowel, in each
modality) with smooth formant/articulator-like transitions between them; a
"vowel boundary" is a frame where the underlying vowel identity changes.

The visual stream is generated from a *shifted* copy of the same label
sequence (articulators start moving a few frames before the sound arrives),
so the two modalities are correlated but not identical -- exactly the
cross-modal cue-integration setting ``model.py`` is built for.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class AVDataset:
    """Fixed per-vowel prototype features shared across all generated batches."""
    audio_proto: np.ndarray   # (n_vowels, dim_audio)
    visual_proto: np.ndarray  # (n_vowels, dim_visual)

    @property
    def n_vowels(self) -> int:
        return self.audio_proto.shape[0]

    @property
    def dim_audio(self) -> int:
        return self.audio_proto.shape[1]

    @property
    def dim_visual(self) -> int:
        return self.visual_proto.shape[1]


def make_av_dataset(n_vowels: int = 4, dim_audio: int = 8, dim_visual: int = 4,
                     seed: int = 0) -> AVDataset:
    """Draw fixed, unit-norm random prototype features for each of `n_vowels`."""
    rng = np.random.default_rng(seed)
    audio_proto = rng.normal(size=(n_vowels, dim_audio)).astype(np.float32)
    visual_proto = rng.normal(size=(n_vowels, dim_visual)).astype(np.float32)
    audio_proto /= np.linalg.norm(audio_proto, axis=1, keepdims=True)
    visual_proto /= np.linalg.norm(visual_proto, axis=1, keepdims=True)
    return AVDataset(audio_proto, visual_proto)


def _segment_labels(t_len: int, n_vowels: int, min_seg: int, max_seg: int,
                     rng: np.random.Generator) -> np.ndarray:
    """A (T,) array of vowel ids, piecewise-constant on random-length segments."""
    labels = np.empty(t_len, dtype=np.int64)
    t, prev = 0, -1
    while t < t_len:
        vowel = rng.integers(n_vowels - 1)
        vowel += vowel >= prev if prev >= 0 else 0   # avoid immediate repeats
        seg_len = int(rng.integers(min_seg, max_seg + 1))
        labels[t:t + seg_len] = vowel
        t += seg_len
        prev = vowel
    return labels[:t_len]


def _labels_to_features(labels: np.ndarray, proto: np.ndarray, ramp: int) -> np.ndarray:
    """Look up each frame's prototype, then smooth across time (formant transitions
    are gradual, not step functions). ``ramp`` is the smoothing window in frames."""
    feat = proto[labels]                       # (T, dim)
    if ramp > 1:
        kernel = np.ones(ramp, dtype=np.float32) / ramp
        pad_l = ramp // 2
        padded = np.pad(feat, ((pad_l, ramp - 1 - pad_l), (0, 0)), mode="edge")
        feat = np.stack(
            [np.convolve(padded[:, d], kernel, mode="valid") for d in range(feat.shape[1])],
            axis=1)
    return feat.astype(np.float32)


def generate_av_batch(dataset: AVDataset, batch_size: int, t_len: int,
                       rng: np.random.Generator, *, min_seg: int = 4, max_seg: int = 10,
                       ramp: int = 3, visual_lead: int = 2,
                       audio_noise: float = 0.1, visual_noise: float = 0.1) -> Dict[str, np.ndarray]:
    """One batch of synthetic audio-visual sequences with ground-truth boundaries.

    Returns a dict of ``(batch_size, t_len, dim)`` arrays keyed ``"audio"`` and
    ``"visual"`` (ready for ``Simulation``'s temporal clamping -- see the
    omni-pcn README's "Clamping options" -- Temporal clamp), plus
    ``(batch_size, t_len)`` arrays ``"boundary"`` (1 at each vowel onset) and
    ``"label"`` (the audio-aligned vowel id, for inspection/plotting).
    """
    audio = np.empty((batch_size, t_len, dataset.dim_audio), dtype=np.float32)
    visual = np.empty((batch_size, t_len, dataset.dim_visual), dtype=np.float32)
    boundary = np.zeros((batch_size, t_len), dtype=np.float32)
    label = np.empty((batch_size, t_len), dtype=np.int64)

    for b in range(batch_size):
        labels = _segment_labels(t_len, dataset.n_vowels, min_seg, max_seg, rng)
        # Visual articulation leads the acoustic signal by a few frames.
        vis_labels = np.concatenate([labels[visual_lead:], np.full(visual_lead, labels[-1])])

        audio[b] = _labels_to_features(labels, dataset.audio_proto, ramp)
        visual[b] = _labels_to_features(vis_labels, dataset.visual_proto, ramp)
        boundary[b, 1:] = (labels[1:] != labels[:-1]).astype(np.float32)
        label[b] = labels

    audio += rng.normal(scale=audio_noise, size=audio.shape).astype(np.float32)
    visual += rng.normal(scale=visual_noise, size=visual.shape).astype(np.float32)
    return {"audio": audio, "visual": visual, "boundary": boundary, "label": label}


def drop_modality_spans(x: np.ndarray, drop_prob: float, rng: np.random.Generator,
                         min_len: int = 3, max_len: int = 8) -> np.ndarray:
    """A ``(B, T)`` soft-clamp mask for ``x`` (shape ``(B, T, dim)``): 1 everywhere,
    except random ``[min_len, max_len]``-frame spans per sequence set to 0 with
    probability ``drop_prob`` -- standing in for a degraded/occluded modality
    (e.g. the camera is briefly blocked, or audio clips). Feed as the mask half
    of a masked/soft clamp (see the omni-pcn README's "Clamping options").
    """
    batch_size, t_len = x.shape[0], x.shape[1]
    mask = np.ones((batch_size, t_len), dtype=np.float32)
    for b in range(batch_size):
        if rng.random() < drop_prob:
            length = int(rng.integers(min_len, max_len + 1))
            start = int(rng.integers(0, max(1, t_len - length + 1)))
            mask[b, start:start + length] = 0.0
    return mask
