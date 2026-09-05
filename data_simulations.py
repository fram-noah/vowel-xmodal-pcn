import os
import pcn
os.environ["JAX_PLATFORMS"]= 'cpu'

import numpy as np
import jax.numpy as jnp
import optax
import pcn

## Exploration: Constructing a toy multimodal vowel/consonant classifier

SEED = 9
N_TRAIN, N_TEST = 800,200 # examples to make
BATCH = 40  # how many examples the network looks at in one go

SR = 16000 # sample rate in hertz
CLIP = 4096 # length of one sound clip in samples
N_MELS = 16 # how many frequency bands that the ear produces
N_VIDEO = 4 # number of mouth measurements per example, i.e. opening, width, rounding, jaw

front = pcn.PCNetwork(seed=SEED)  # network to simulate ear

with front:
    ear = pcn.AuditoryInput(n_samples=CLIP, sr=SR, n_mels=N_MELS, label='ear')

AUDIO_DIM = ear.dim  # numbers per clip after encoding

def vowel_wave(rng, t):
    # This is once voiced lip, with harmonics on a pitch, shaped by two peaks
    # rng is a numpy random generator, with t denoting time axis in seconds
    f0 = rng.uniform(180, 300)  # higher pitch for child-direct speech

    f1, f2 = rng.uniform(300, 800), rng.uniform(1000, 2500)  # Constructing formant centers
    wav = np.zeros_like(t)

    for k in range(1, 20):  # 19 harmonics of the pitch
        fk = k * f0
        gain = np.exp(-(((fk - f1) / 200) ** 2) + 0.6 * np.exp(-((fk - f2) / 300) ** 2))
        wav += gain * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))

    return wav

def consonant_wave(rng, t):
    # same inputs and ouput shape as vowel_wave

    noise = rng.standard_normal(len(t))
    noise = np.diff(noise, prepend=0)  # pushing energy t hihger bands

    env = np.exp(-t / rng.uniform(0.03, 0.1))  # shape onset with fast decay

    return noise * env

def make_clip(is_vowel, rng, t, audio_noise):  # one sound clip,normalized to the same loudness

    w = vowel_wave(rng, t) if is_vowel else consonant_wave(rng, t)

    w = w / (np.abs(w).max() + 1e-6)
    return w + rng.normal(0, audio_noise, len(t))


def make_mouth(is_vowel, rng, video_noise):
    # one set of mouth features: open for vowel, closed otherwise
    if is_vowel:
        mouth = np.array([1.0, 0.7, rng.uniform(0, 1), 0.8])

    else:
        mouth = np.array([0.1, 0.4, 0.2, 0.1])

    return mouth + rng.normal(0, video_noise, N_VIDEO)


def simulate(n, rng, audio_noise=0.05,
             video_noise=0.3):  # with n labelled examples as a dictionary of arrays, audio, video, label
    y = rng.integers(0, 2, size=n)
    # truth : 0 consonant, 1 vowel
    t = np.arange(CLIP) / SR
    waves = np.stack([make_clip(v, rng, t, audio_noise) for v in y]).astype(np.float32)
    video = np.stack([make_mouth(v, rng, video_noise) for v in y]).astype(np.float32)
    audio = np.asarray(ear.encode(jnp.asarray(waves)))  # putting all clips through the ear
    label = np.eye(2, dtype=np.float32)[
        y]  # one-hot encoding the truth about whether each example is a consonant or a vowel
    return {"audio": audio, "video": video, "label": label, "label_idx": y, "wave": waves}



rng = np.random.default_rng(SEED)
d = simulate(5, rng)
print({k: v.shape for k, v in d.items()})
print('labels:', d['label_idx'])