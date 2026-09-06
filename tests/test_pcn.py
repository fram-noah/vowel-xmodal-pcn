"""
Functions for testing scripts.
"""

import numpy
import jax.numpy as jnp
import pcn
import pytest
from pcn_model import *

# Things we need tests for [PLEASE ADD TO LIST!]:
#   1. Model structure is connected
#   2. Input dimensions match signal dimensions
#   3. Every frame has exactly one audio window matched to it
#   4. No computation errors in audio spectrograms
#   5. Video conversion to greyscale runs properly (or color handling is correct)

# `omni-pcn` already ships AuditoryInput/VisualInput sensory front-ends (see
# https://github.com/nsdumont/omni-pcn), so items 4 and 5 can be tested against
# those directly. Items 1-3 depend on this project's own windowing scheme and
# model architecture, which are still open BrainHack 2026 goals (see the issue),
# so they stay skipped until that code exists.

# def make_audiolayer():
#     net = pcn.PCNetwork(seed=0)
#     with net:
#         # Wraps the STFT -> mel-power -> compression pipeline described in
#         # audio.py. n_samples is the raw waveform length this layer expects;
#         # sr/n_fft/hop/n_mels are standard spectrogram parameters.
#         aud = pcn.AuditoryInput(
#             n_samples=4096, sr=16000, n_fft=512, hop=256, n_mels=32,
#             griffin_lim_iters=4, label="aud",
#         )
#
#     # Fake a batch of 3 raw audio clips (random noise -- we're only testing
#     # that the pipeline runs and produces sane numbers, not real audio content).
#     rng = numpy.random.default_rng(0)
#     wave = rng.standard_normal((3, 4096)).astype(numpy.float32)
#     # encode() runs the waveform through the spectrogram pipeline and
#     # flattens the result to (batch, feature_dim).
#     feats = aud.encode(jnp.asarray(wave))
#     return feats

def audio_window(audiosig, fs, framerate):
    samps_per_window = jnp.floor(fs / framerate)
    n_windows = audiosig.shape[0] / samps_per_window
    audio_windowed = []
    for iwin in range(n_windows):
        this_window = audiosig[samps_per_window * iwin,:]
        audio_windowed.append(this_window)
    return audio_windowed

class TestInputs:
    # def test_windowmatch(self):
    #     """Every video frame should have exactly one audio window matched to it."""
    #     audio_windowed = audio_window()
    #     assert len(audio_windowed) == n_frames
    #
    # def test_audiospec(self):
    #     """Audio spectrogram computation (omni-pcn's AuditoryInput) should run
    #     without errors and produce a finite, correctly-shaped feature map."""
    #     # `AuditoryInput` is a pcn.Layer, and layers can only be created while a
    #     # PCNetwork is "open" (inside the `with net:` block) -- that's how pcn
    #     # tracks which layers belong to which network.
    #     feats = make_audiolayer()
    #
    #     # Batch size preserved, and the flattened feature dim matches what
    #     # AuditoryInput reports as its output size.
    #     assert feats.shape == (3, aud.dim)
    #     assert feats.shape[1] == numpy.prod(aud.feature_shape)
    #     # No NaN/Inf sneaking out of the FFT/log/compression math.
    #     assert bool(numpy.all(numpy.isfinite(numpy.asarray(feats))))

    def test_greyscale(self):
        """Greyscale video frames (VisualInput with color='gray') should encode
        without errors and produce a finite, correctly-shaped feature map."""
        h, w = 28, 28
        net = pcn.PCNetwork(seed=0)
        with net:
            # in_shape=(1, h, w): 1 channel = greyscale. color="gray" tells
            # VisualInput to treat the single channel as luminance directly
            # (no RGB-to-opponent-color conversion).
            vis = pcn.VisualInput(in_shape=(1, h, w), color="gray", label="vis")

        # Fake a batch of 3 flattened greyscale frames (random pixels -- again,
        # only checking the pipeline runs correctly, not real video content).
        rng = numpy.random.default_rng(0)
        frames = rng.random((3, h * w)).astype(numpy.float32)
        feats = vis.encode(jnp.asarray(frames))

        assert feats.shape == (3, vis.dim)
        assert feats.shape[1] == numpy.prod(vis.feature_shape)
        assert bool(numpy.all(numpy.isfinite(numpy.asarray(feats))))

    def test_greyscale_rejects_color_input(self):
        """color='gray' should reject 3-channel (RGB) input rather than silently
        averaging or mishandling the channels."""
        # If someone accidentally passes a 3-channel (RGB) shape while asking
        # for color="gray", VisualInput should fail loudly (ValueError) instead
        # of silently doing something wrong with the extra channels.
        net = pcn.PCNetwork(seed=0)
        with net:
            with pytest.raises(ValueError):
                pcn.VisualInput(in_shape=(3, 28, 28), color="gray")


def _build_toy_multimodal_net(seed=0, n_mel=8, n_video=4, hidden=16, n_out=2):
    """A small stand-in multimodal network, not the project's real architecture
    (that's still an open BrainHack 2026 goal -- see the issue). It mirrors the
    fusion pattern from a teammate's exploratory notebook (commit 940bd43,
    later removed from main in d80ed4c): each modality (audio, video) rises
    through its own hidden layer, then BOTH hidden layers feed a single shared
    prediction of one output layer.

    This matters because of a gotcha documented in that same exploration: if
    you instead write two separate edges -- Predict(l_h_aud, l_label) and
    Predict(l_h_vid, l_label) -- pcn treats them as two independent error
    factors (product-of-experts), and the easier modality (video) can end up
    explaining the label on its own while the audio pathway never learns
    anything. Joining them as Predict([l_h_aud, l_h_vid], l_label) is what
    makes it a genuine joint fusion. This smoke test exists to catch a
    regression back to the broken (split) wiring once the real model lands.
    """
    net = pcn.PCNetwork(seed=seed)
    with net:
        l_aud = pcn.Layer(dim=n_mel, activation=pcn.Direct(), label="audio")
        l_vid = pcn.Layer(dim=n_video, activation=pcn.Direct(), label="video")
        l_h_aud = pcn.Layer(dim=hidden, activation=pcn.LeakyRelu(), label="hidden_aud")
        l_h_vid = pcn.Layer(dim=hidden, activation=pcn.LeakyRelu(), label="hidden_vid")
        l_label = pcn.Layer(dim=n_out, activation=pcn.Softmax(), label="label")
        pcn.Predict(l_aud, l_h_aud)
        pcn.Predict(l_vid, l_h_vid)
        pcn.Predict([l_h_aud, l_h_vid], l_label)  # joint fusion, NOT two edges
    net.build()
    return net, l_aud, l_vid, l_label


class TestModel:
    def test_connected(self):
        """Tests whether there are hanging or disconnected layers."""
        pcndef = PCNDef()
        net, handles = make_network_sequential(pcndef)
        layerlist = [layer.label for layer in net._layers]
        postlist = [conn.post.label for conn in net._predict_conns]
        prelist_tmp = [conn.pre for conn in net._predict_conns]
        prelist = []
        for ipre in prelist_tmp:
            if isinstance(ipre, list):
                prelist = prelist + [prelayer.label for prelayer in ipre]
            else:
                prelist = prelist + [ipre.label]
        connected_layers = list(set(postlist + prelist))
        assert all([layer in connected_layers for layer in layerlist])

    def test_dimensionmatch(self):
        """Input dimensions should match the signal dimensions the model expects.

        Same caveat as test_connected: exercises the stand-in architecture, to
        be replaced once the real multimodal model exists.
        """
        n_mel, n_video, n_out = 8, 4, 2
        net, l_aud, l_vid, l_label = _build_toy_multimodal_net(
            n_mel=n_mel, n_video=n_video, n_out=n_out)

        # Each layer's declared dim should match the signal it's meant to carry.
        assert l_aud.dim == n_mel
        assert l_vid.dim == n_video
        assert l_label.dim == n_out

        # Feed a batch where audio/video/label widths match those dims, and
        # confirm the network runs end-to-end without a shape mismatch and
        # produces a correctly-shaped, finite prediction.
        sim = pcn.Simulation(net)
        rng = numpy.random.default_rng(0)
        n = 6
        audio = rng.standard_normal((n, n_mel)).astype(numpy.float32)
        video = rng.standard_normal((n, n_video)).astype(numpy.float32)
        loader = [{"audio": audio, "video": video}]
        res = sim.test(loader, data_map={l_aud: "audio", l_vid: "video"},
                        iterations_per_sample=5,
                        record_map={"pred": (l_label.value, lambda v: v)})
        pred = numpy.asarray(res["pred"][0])

        assert pred.shape == (n, n_out)
        assert bool(numpy.all(numpy.isfinite(pred)))


# We can also add tests here that look at output parameters, but I've kept things
# to the model structure itself for now.
