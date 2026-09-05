# vowel-xmodal-pcn
Use multimodal predictive coding networks to detect vowel boundaries in child-directed song.

![Schematic of project structure](BrainHackImage.png)

## Summary
This project uses a multimodal predictive coding network, implemented in `omni-pcn`, to find vowel boundaries in
audiovisual recordings of child-directed song. It is intended to help human coders in situations where either the audio
or video signal is degraded or absent.

## Instructions

Clone the repository.
1) In your terminal or command prompt, navigate to the location you want the repository to live.
2) Run the command `git clone https://github.com/fram-noah/vowel-xmodal-pcn`

Download and install the `omni-pcn` package (see instructions below).

Start hacking!

## Dependencies
`omni-pcn`: https://github.com/nsdumont/omni-pcn

## The basic model (`av_pcn/`)

A first, deliberately minimal architecture built entirely from unmodified `omni-pcn`
building blocks (nothing in `omni-pcn` itself is changed):

```
audio(dim_audio) --Predict--+
                             +--> fusion(dim_hidden) --Predict(delay=1, timestep)--> fusion
visual(dim_visual) --Predict-+
fusion --(weak, learned precision)--> audio    (top-down reconstruction)
fusion --(weak, learned precision)--> visual   (top-down reconstruction)
```

- **Cross-modal cue integration**: separate `Predict` edges from `audio` and `visual`
  into a shared `fusion` layer (product-of-experts -- see the `omni-pcn` README's
  "Arbitrary graph structure" section). `fusion` settles at their precision-weighted
  combination, so the network keeps working when one modality is noisy, degraded, or
  absent -- the whole point of this project.
- **Time**: a single delayed self-`Predict` on `fusion` (`delay=1, delay_unit="timestep"`)
  is the entire "handle time-varying input" mechanism (see the `omni-pcn` README's
  "Delayed connections & temporal predictive coding" section). Sequences are fed with
  `omni-pcn`'s temporal clamping (a `(batch, T, dim)` array + `iterations_per_sample`
  a multiple of `T`), which is also the "windowing procedure" called for in the
  BrainHack goals below.
- **Generative top-down edges** let the network reconstruct a clamped-out/degraded
  modality from the other.

Build it with:

```python
from av_pcn import build_basic_av_pcn
net, layers, edges = build_basic_av_pcn(dim_audio=8, dim_visual=4, dim_hidden=16)
```

`examples/basic_av_demo.py` trains and evaluates this model end to end on synthetic
audio-visual sequences (`av_pcn/synthetic_data.py`), since no real dataset is
available (see "Data availability" below): cross-modal reconstruction of a missing
modality, and a first-pass vowel-boundary detector from the fusion layer's temporal
prediction error.

### Setup

`omni-pcn` is expected as a sibling checkout:

```
phoneme-boundaries/
├── .venv/            <- one shared environment for both projects
├── omni-pcn/
└── vowel-xmodal-pcn/
```

**One shared venv, at the `phoneme-boundaries/` level, not one per project.**
`vowel-xmodal-pcn` only exists to depend on `omni-pcn`, so two separate venvs
would just mean two ~1.3&nbsp;GB copies of `jax`/`torch` that can silently drift
out of sync -- and it's the reason "sometimes I open the parent folder,
sometimes the child folder" caused inconsistent behaviour: each folder's own
venv only gets auto-detected when *that exact folder* is the open workspace
root. A single venv one level up, referenced by an explicit interpreter path
(see below), works no matter which folder you open.

From `phoneme-boundaries/` (needs pip >= 25.1 for `omni-pcn`'s Apple-Silicon/
CUDA dependency groups; swap in `uv sync` per-project if you prefer `uv`):

```bash
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e omni-pcn             # add: pip install jax-mps==0.9.13  (Apple Silicon Metal)
                                               # or:  pip install -e omni-pcn --group cuda  (NVIDIA)
.venv/bin/pip install -e vowel-xmodal-pcn     # installs this repo's av_pcn package
.venv/bin/pip install matplotlib              # optional, for the demo's plot
```

```bash
.venv/bin/python vowel-xmodal-pcn/examples/basic_av_demo.py
```

**VS Code**: `.vscode/settings.json` in both this repo and the parent
`phoneme-boundaries/` folder pin `python.defaultInterpreterPath` to that
shared venv, so opening either as the workspace root resolves the same
interpreter automatically -- select it once (`Cmd+Shift+P` → *Python: Select
Interpreter*) and it should already be offered.

This is a starting point, not a finished model -- see the BrainHack 2026 goals below.

## BrainHack 2026 goals
In BrainHack 2026, we will aim to accomplish the following steps:
1) Familiarize participants with the `omni-pcn` package
2) Build an initial architecture capable of handling time-dependent visual and auditory inputs
3) Decide on a windowing procedure for handling time-varying audiovisual information

### Data availability
For BrainHack 2026, we will be working without a dataset. We have a collection of audiovisual recordings with human
vowel annotations, but since those recordings are not licensed for public, non-academic use, they will not be
included in this repository.

## Background readings
Although not required, these readings may give some useful information on the methods and theoretical background for
this project.

- van Zwol, B., Jefferson, R., van den Broek, E. L. (2026). Predictive coding networks and inference learning: Tutorial
and survey. *ACM Computing Surveys*, *58*(10), 1-47. doi: 10.1145/3797870. https://dl.acm.org/doi/full/10.1145/3797870
- Lovčević, I., Benders, T., Tsuji, S., & Fusaroli, R. (2025). Acoustic exaggeration of vowels in infant-directed
speech: A multimethod meta-analytic review. *Psychological Bulletin*, *151*(6), 669–695. doi: 10.1037/bul0000479.
https://psycnet.apa.org/record/2026-23414-001?doi=1
- Hilton, C. B., Moser, C. J., Bertolo, M., *et al*. (2022). Acoustic regularities in infant-directed speech and song
across cultures. *Nature Human Behavior*, *6*, 1545-1556. doi: 10.1038/s41562-022-01410-x.
https://www.nature.com/articles/s41562-022-01410-x
- Cox, C. Dideriksen, C. Keren-Portnoy, T., Roepstorff, A. Christiansen, M. H., & Fusaroli, R. (2023). Infant-directed
speech does not always involve exaggerated vowel distinctions: Evidence from Danish. *Child Development*, *94*(6),
1672-1696. doi: 10.1111/cdev.13950. https://academic.oup.com/chidev/article/94/6/1672/8255271