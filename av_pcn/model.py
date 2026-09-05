"""The basic audio-visual PCN architecture.

    audio(dim_audio) --Predict--+
                                 +--> fusion(dim_hidden) --Predict(delay=1,timestep)--> fusion
    visual(dim_visual) --Predict-+
    fusion --(weak, learnable-precision)--> audio   (top-down reconstruction)
    fusion --(weak, learnable-precision)--> visual   (top-down reconstruction)

Everything here is plain `pcn` (from `omni-pcn`) API calls -- no changes to
that package. See the module docstring in ``av_pcn/__init__.py`` for the
rationale behind each edge.
"""
from typing import Dict, Tuple

import pcn

GEN_INIT_PRECISION_EXP = -2  # top-down edges start weak (matches examples/multimodal_alphanum.py)


def build_basic_av_pcn(
    dim_audio: int,
    dim_visual: int,
    dim_hidden: int = 16,
    *,
    seed: int = 0,
    learn_precision_weights: bool = True,
    generative: bool = True,
) -> Tuple["pcn.PCNetwork", Dict[str, "pcn.Layer"], Dict[str, "pcn.Predict"]]:
    """Build the smallest audio-visual, time-aware PCN this repo needs.

    Args:
        dim_audio: Dimensionality of one audio frame's features (e.g. a small
            log-mel/formant-like vector -- see ``synthetic_data.py``).
        dim_visual: Dimensionality of one visual frame's features (e.g. a
            lip/mouth-shape proxy -- see ``synthetic_data.py``).
        dim_hidden: Size of the shared cross-modal ``fusion`` layer.
        seed: PCNetwork RNG seed.
        learn_precision_weights: Whether the two ``Predict`` edges into
            ``fusion`` learn input-dependent precision, i.e. the network
            learns *when* to trust audio vs. visual, instead of a fixed
            per-dimension weighting.
        generative: If True (default), add weak top-down edges from
            ``fusion`` back to ``audio``/``visual`` so a clamped-out or
            degraded modality can be reconstructed from the other (relax with
            one modality clamped, read the other's prediction).

    Returns:
        ``(net, layers, gen_edges)`` where ``net`` is the *built*
        ``pcn.PCNetwork``, ``layers`` maps ``{"audio", "visual", "fusion"}``
        to their ``pcn.Layer``, and ``gen_edges`` always has a ``"temporal"``
        key (the delayed fusion self-``Predict``) plus ``{"audio", "visual"}``
        top-down edges when ``generative=True``.
    """
    net = pcn.PCNetwork(seed=seed)
    # Freeze precision everywhere by default (init_precision=1.0); only the
    # edges below that explicitly opt back in learn it. Left on the omni-pcn
    # default (learned everywhere), the temporal self-edge can trivially
    # lower its own precision to "stop caring" about mispredictions instead
    # of learning to predict -- which quietly destroys it as a surprise/
    # boundary signal (its error is precision * (post - prediction)).
    net.config(learn_precision_weights=False, learn_precision_bias=False)
    with net:
        l_audio = pcn.Layer(dim=dim_audio, activation=pcn.Direct(), label="audio")
        l_visual = pcn.Layer(dim=dim_visual, activation=pcn.Direct(), label="visual")
        l_fusion = pcn.Layer(dim=dim_hidden, activation=pcn.Tanh(), label="fusion")

        # Cue integration: two independent predictions of `fusion`, one per
        # modality. `fusion` settles at their precision-weighted combination,
        # so if one modality is noisy/missing the other still drives it.
        pcn.Predict(l_audio, l_fusion, learn_precision_weights=learn_precision_weights)
        pcn.Predict(l_visual, l_fusion, learn_precision_weights=learn_precision_weights)

        # Temporal predictive coding: `fusion` predicts its own next timestep
        # from a latched (one input-timestep old) copy of itself. This is the
        # entire "handle time-varying input" mechanism -- one delayed
        # self-edge, no custom recurrence. Its error is a natural "surprise"
        # signal: it spikes whenever fusion changes faster than the learned
        # temporal prior expects -- e.g. at a vowel boundary.
        gen_edges = {"temporal": pcn.Predict(l_fusion, l_fusion, delay=1, delay_unit="timestep")}

        if generative:
            # Weak top-down reconstruction paths, so a degraded/absent
            # modality can be filled in from the other + the temporal prior.
            gen_edges["audio"] = pcn.Predict(
                l_fusion, l_audio, init_log_precision=GEN_INIT_PRECISION_EXP,
                learn_precision_bias=True)
            gen_edges["visual"] = pcn.Predict(
                l_fusion, l_visual, init_log_precision=GEN_INIT_PRECISION_EXP,
                learn_precision_bias=True)

    net.build()
    layers = {"audio": l_audio, "visual": l_visual, "fusion": l_fusion}
    return net, layers, gen_edges
