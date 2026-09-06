"""
Functions for testing scripts.
"""

import numpy
import jax.numpy as jnp
import pcn
import pytest
from hypothesis import given, settings, strategies as st

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

def make_audiolayer():
    net = pcn.PCNetwork(seed=0)
    with net:
        # Wraps the STFT -> mel-power -> compression pipeline described in
        # audio.py. n_samples is the raw waveform length this layer expects;
        # sr/n_fft/hop/n_mels are standard spectrogram parameters.
        aud = pcn.AuditoryInput(
            n_samples=4096, sr=16000, n_fft=512, hop=256, n_mels=32,
            griffin_lim_iters=4, label="aud",
        )

    # Fake a batch of 3 raw audio clips (random noise -- we're only testing
    # that the pipeline runs and produces sane numbers, not real audio content).
    rng = numpy.random.default_rng(0)
    wave = rng.standard_normal((3, 4096)).astype(numpy.float32)
    # encode() runs the waveform through the spectrogram pipeline and
    # flattens the result to (batch, feature_dim).
    feats = aud.encode(jnp.asarray(wave))
    return feats, aud


def audio_window(audiosig, fs, framerate):
    """Split a raw audio signal into consecutive, non-overlapping windows, one
    per video frame -- `fs` is the audio sample rate, `framerate` the video fps.
    Trailing samples that don't fill a whole window are dropped.

    This is Noah's sketch (originally commented out, calling an undefined
    function with undefined args -- see commit 1c14d0f), fixed up: the
    original used float division for `n_windows` (can't `range()` a float) and
    indexed a single row (`audiosig[samps_per_window * iwin, :]`) instead of
    slicing a window of samples.
    """
    samples_per_window = int(fs // framerate)
    if samples_per_window <= 0:
        raise ValueError(f"framerate {framerate} too high for sample rate {fs}")
    n_windows = len(audiosig) // samples_per_window
    return [audiosig[i * samples_per_window:(i + 1) * samples_per_window]
            for i in range(n_windows)]


class TestInputs:
    # Property-based test (Noah wants PBT coverage here): instead of checking
    # one hand-picked (fs, framerate, duration), Hypothesis generates many
    # combinations and checks the invariant holds for all of them -- exactly
    # one audio window per "slot", every window the same length.
    @given(
        fs=st.integers(min_value=1000, max_value=48000),
        framerate=st.integers(min_value=1, max_value=60),
        duration=st.floats(min_value=0.05, max_value=3.0,
                            allow_nan=False, allow_infinity=False),
    )
    def test_windowmatch(self, fs, framerate, duration):
        """Every video frame should have exactly one audio window matched to it."""
        n_samples = int(fs * duration)
        audiosig = numpy.zeros(n_samples, dtype=numpy.float32)
        samples_per_window = fs // framerate

        windows = audio_window(audiosig, fs, framerate)

        expected_n_windows = n_samples // samples_per_window
        assert len(windows) == expected_n_windows
        assert all(len(w) == samples_per_window for w in windows)

    @given(
        seed=st.integers(min_value=0, max_value=2**31 - 1),
        fs=st.integers(min_value=1000, max_value=48000),
        framerate=st.integers(min_value=1, max_value=60),
        duration=st.floats(min_value=0.05, max_value=2.0,
                            allow_nan=False, allow_infinity=False),
    )
    def test_windowmatch_reconstructs_prefix(self, seed, fs, framerate, duration):
        """Concatenating the windows back together should exactly reproduce a
        prefix of the original signal (everything but the trailing remainder
        that doesn't fill a whole window). A stronger check than counting
        windows -- it also catches off-by-one slicing, dropped/duplicated/
        reordered samples, or accidental mutation of the signal."""
        n_samples = int(fs * duration)
        rng = numpy.random.default_rng(seed)
        audiosig = rng.standard_normal(n_samples).astype(numpy.float32)
        samples_per_window = fs // framerate

        windows = audio_window(audiosig, fs, framerate)

        kept = len(windows) * samples_per_window
        reconstructed = (numpy.concatenate(windows) if windows
                         else numpy.array([], dtype=numpy.float32))
        assert numpy.array_equal(reconstructed, audiosig[:kept])

    def test_audiospec(self):
        """Audio spectrogram computation (omni-pcn's AuditoryInput) should run
        without errors and produce a finite, correctly-shaped feature map."""
        # `AuditoryInput` is a pcn.Layer, and layers can only be created while a
        # PCNetwork is "open" (inside the `with net:` block) -- that's how pcn
        # tracks which layers belong to which network.
        feats, aud = make_audiolayer()

        # Batch size preserved, and the flattened feature dim matches what
        # AuditoryInput reports as its output size.
        assert feats.shape == (3, aud.dim)
        assert feats.shape[1] == numpy.prod(aud.feature_shape)
        # No NaN/Inf sneaking out of the FFT/log/compression math.
        assert bool(numpy.all(numpy.isfinite(numpy.asarray(feats))))

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1),
           n=st.integers(min_value=1, max_value=5))
    def test_audiospec_batch_independent(self, seed, n):
        """Encoding a batch of clips together should give the exact same
        per-clip features as encoding each clip alone. Batched tensor code
        (the STFT/mel-power pipeline here) can easily leak values across the
        batch axis through a wrong reshape/broadcast -- this property would
        catch that even though single-example shape/finite checks wouldn't.
        """
        _, aud = make_audiolayer()
        rng = numpy.random.default_rng(seed)
        waves = rng.standard_normal((n, 4096)).astype(numpy.float32)

        batched = numpy.asarray(aud.encode(jnp.asarray(waves)))
        singles = numpy.stack([
            numpy.asarray(aud.encode(jnp.asarray(waves[i:i + 1])))[0]
            for i in range(n)
        ])
        assert numpy.allclose(batched, singles, atol=1e-4)

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

    @settings(max_examples=20, deadline=None)
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1),
           n=st.integers(min_value=1, max_value=5))
    def test_greyscale_batch_independent(self, seed, n):
        """Same batch-independence property as test_audiospec_batch_independent,
        for the video pathway: a batch of frames encoded together should match
        encoding each frame alone."""
        h, w = 28, 28
        net = pcn.PCNetwork(seed=0)
        with net:
            vis = pcn.VisualInput(in_shape=(1, h, w), color="gray", label="vis")
        rng = numpy.random.default_rng(seed)
        frames = rng.random((n, h * w)).astype(numpy.float32)

        batched = numpy.asarray(vis.encode(jnp.asarray(frames)))
        singles = numpy.stack([
            numpy.asarray(vis.encode(jnp.asarray(frames[i:i + 1])))[0]
            for i in range(n)
        ])
        assert numpy.allclose(batched, singles, atol=1e-4)

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

    @given(channels=st.integers(min_value=2, max_value=8).filter(lambda c: c != 3),
           h=st.integers(min_value=4, max_value=32),
           w=st.integers(min_value=4, max_value=32))
    def test_visualinput_rejects_invalid_channel_counts(self, channels, h, w):
        """Generalizes test_greyscale_rejects_color_input: VisualInput only
        accepts 1 (gray) or 3 (RGB) input channels. Any other channel count
        should be rejected regardless of the spatial size or color setting."""
        net = pcn.PCNetwork(seed=0)
        with net:
            with pytest.raises(ValueError):
                pcn.VisualInput(in_shape=(channels, h, w))


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
        """Model structure should be fully connected (no dangling layers/inputs).

        This is a smoke test against a stand-in architecture (see
        `_build_toy_multimodal_net`), not the project's final model -- swap in
        the real network builder once that architecture is decided.
        """
        # net.build() raises if any layer/connection is left dangling, so
        # simply reaching this line without an exception is the assertion.
        _build_toy_multimodal_net()

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

    @settings(max_examples=10, deadline=None)
    @given(
        n_mel=st.integers(min_value=1, max_value=12),
        n_video=st.integers(min_value=1, max_value=12),
        hidden=st.integers(min_value=1, max_value=12),
        n_out=st.integers(min_value=2, max_value=6),
    )
    def test_dimensionmatch_property(self, n_mel, n_video, hidden, n_out):
        """Generalizes test_dimensionmatch: the single hardcoded (8, 4, 16, 2)
        case could pass by coincidence (e.g. a builder that silently always
        makes dim-8 layers). Random layer sizes rule that out -- the network
        should build and its dims should always match what was requested,
        whatever those sizes are. (max_examples kept low and deadline
        disabled: JAX recompiles for each new shape, so this is slower per
        example than the other property tests here.)
        """
        _, l_aud, l_vid, l_label = _build_toy_multimodal_net(
            n_mel=n_mel, n_video=n_video, hidden=hidden, n_out=n_out)
        assert l_aud.dim == n_mel
        assert l_vid.dim == n_video
        assert l_label.dim == n_out


# We can also add tests here that look at output parameters, but I've kept things
# to the model structure itself for now.