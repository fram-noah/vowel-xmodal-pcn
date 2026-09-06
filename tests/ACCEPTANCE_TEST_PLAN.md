# Acceptance Test Plan — `vowel-xmodal-pcn`

Scope: `tests/test_pcn.py`. This explains what each test checks, why that
check has value, and what still isn't covered. It should be updated whenever
tests are added, removed, or their target code changes.

## Why property-based tests, not just examples

An example-based test ("run it once with these numbers, check the output")
only proves the code works for the numbers you happened to pick. It's easy to
write a test that passes by coincidence — e.g. a batching bug that only shows
up at batch size 3, or a windowing off-by-one that only shows up when the
clip length isn't a clean multiple of the window size.

A property-based test (using [Hypothesis](https://hypothesis.readthedocs.io/))
states an invariant that should hold for *any* valid input, then generates
many varied inputs (Hypothesis defaults to 100 per test here) and checks the
invariant holds for all of them. This project's inputs — audio sample rates,
video framerates, clip lengths, batch sizes, layer dimensions — are exactly
the kind of parameter space where a single hardcoded example gives false
confidence.

## Test inventory

| Test | Property checked | Bug class it catches | Status |
|---|---|---|---|
| `test_windowmatch` | For any `(sample rate, framerate, duration)`, the number of audio windows equals `n_samples // samples_per_window`, and every window has the same length. | Wrong window-count arithmetic; a windowing scheme that silently drops or duplicates a video frame's worth of audio. | Real — tests `audio_window`, a fixed-up version of Noah's sketch. |
| `test_windowmatch_reconstructs_prefix` | Concatenating all windows back together reproduces the original signal's prefix exactly (byte-for-byte). | Off-by-one slicing, reordered/duplicated samples, or accidental mutation — bugs that `test_windowmatch` (which only counts and measures windows) would miss even though the count is right. | Real. |
| `test_audiospec` | `AuditoryInput.encode` returns a finite, correctly-shaped feature map for a batch of raw waveforms. | Crashes or NaN/Inf in the STFT → mel-power → compression pipeline. | Real — exercises `omni-pcn`'s `AuditoryInput`. |
| `test_audiospec_batch_independent` | Encoding a batch of *N* clips gives the same per-clip result as encoding each clip alone. | Batch-axis leakage from a wrong reshape/broadcast in the batched tensor pipeline — a bug class that shape/finiteness checks alone can't catch, since the *shape* comes out right even when values have leaked across the batch. | Real. |
| `test_greyscale` | `VisualInput(color="gray")` returns a finite, correctly-shaped feature map for a batch of greyscale frames. | Crashes or NaN/Inf in the retina→V1 feature pipeline for single-channel input. | Real. |
| `test_greyscale_batch_independent` | Same batch-independence property as above, for the video pathway. | Same as `test_audiospec_batch_independent`, video side. | Real. |
| `test_greyscale_rejects_color_input` | `VisualInput(color="gray")` raises `ValueError` on 3-channel input instead of mishandling it. | Silent misinterpretation of channel data (e.g. averaging RGB into a fake "gray" channel) instead of failing loudly. | Real. |
| `test_visualinput_rejects_invalid_channel_counts` | Generalizes the above: *any* channel count other than 1 or 3, at *any* spatial size, is rejected. | Same bug class, but the single hardcoded case (channels=3) could pass while other invalid counts (2, 4, 5, ...) slip through; this rules that out. | Real. |
| `test_connected` | A small multimodal network (audio + video → per-modality hidden layers → one *joint* prediction of a shared output layer) builds without a dangling layer or connection. | Structural wiring mistakes that `PCNetwork.build()` is supposed to catch. | **Stand-in** — exercises a placeholder architecture (`_build_toy_multimodal_net`), not the project's real model, which is still an open BrainHack 2026 goal. |
| `test_dimensionmatch` | For one fixed set of layer sizes, every layer's declared `dim` matches what was requested, and a forward pass through the network produces a correctly-shaped, finite prediction. | Shape mismatches between a layer's declared dimension and the data mapped to it. | **Stand-in**, same caveat as `test_connected`. |
| `test_dimensionmatch_property` | Generalizes `test_dimensionmatch` across many random layer-size combinations. | The single hardcoded case (8, 4, 16, 2) could pass by coincidence (e.g. a builder that silently always produces dim-8 layers); random sizes rule that out. | **Stand-in**, same caveat. |

## Known gaps (not yet covered)

- **The project's actual multimodal architecture.** `test_connected` /
  `test_dimensionmatch*` exercise a placeholder network mirroring a teammate's
  removed exploratory script (git history `940bd43`), not the real model.
  Swap in the real network builder once BrainHack goal 2 ("build an initial
  model architecture") is decided, and these tests should keep passing
  unchanged in spirit (still checking connectivity and dimension-match) even
  as the builder function changes.
- **The project's actual audio/video windowing scheme.** `audio_window` here
  is a fixed-up version of Noah's sketch, kept in the test file as a
  reference implementation to test against. Once BrainHack goal 3
  ("investigate windowing procedures") lands as real pipeline code, these
  tests should be pointed at that implementation instead.
- **`notebooks/01_exploration_nontemporal.ipynb`.** Not covered at all —
  pytest doesn't collect notebook cells, and the notebook itself doesn't
  finish building a network yet. If the data-simulation functions in it
  (`vowel_wave`, `consonant_wave`, `make_mouth`, `simulate`) get promoted into
  a real module, they should get their own tests (e.g. a property test that
  `simulate(n, rng)` always returns arrays of length `n` with `label_idx`
  consistent with `label`'s one-hot encoding).

## Running the tests

```
uv run pytest tests/ -v
```

Add `--hypothesis-show-statistics` to see how many generated examples each
property test ran. `test_dimensionmatch_property` is capped at 10 examples
(vs. the default 100) because JAX recompiles for each new layer-size
combination it sees; the other property tests run the full 100 (or 20 for the
batch-independence tests, which rebuild a sensory-input layer per example).
