"""av_pcn: a minimal audio-visual predictive coding network, built on top of
the (unmodified) `omni-pcn` package.

This is the smallest cross-modal, time-aware model we could derive from
`omni-pcn`'s building blocks:

- Two sensory layers (``audio``, ``visual``) feed a shared ``fusion`` layer
  through *separate* ``Predict`` edges (cue integration / product-of-experts:
  each modality gets its own precision-weighted vote -- see the omni-pcn
  README's "Arbitrary graph structure" section). This is what lets the model
  keep working when one modality is degraded or missing, which is the whole
  point of this project.
- The fusion layer predicts its own next value at ``delay=1`` (a one-line
  temporal predictive-coding edge -- see the omni-pcn README's "Delayed
  connections & temporal predictive coding" section), giving the network a
  minimal notion of time/history without any custom recurrence code.
- Weak top-down edges from the fusion layer back to each sensory layer let the
  network reconstruct a clamped-out/degraded modality from the other.

Nothing in `omni-pcn` itself is modified; this package only imports and
composes it.
"""

from .model import build_basic_av_pcn
from .synthetic_data import generate_av_batch, make_av_dataset

__all__ = ["build_basic_av_pcn", "generate_av_batch", "make_av_dataset"]
