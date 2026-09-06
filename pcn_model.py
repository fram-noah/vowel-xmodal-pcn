import numpy as np
import jax.numpy as jnp
import pcn

# Hardcoded for now, on purpose -- we'll turn these into real Params once the
# skeleton itself is proven to work. Guessing dimensions is the #1 way to burn
# time on confusing errors, so these came from actually checking, not assuming:
#   - AUDIO_DIM=480 is what AuditoryInput(n_mels=32, n_samples=4096, ...)
#     actually reports for its output size (checked by running it directly).
#   - VIDEO_DIM=4 matches the toy mouth features in data_simulations.py.
AUDIO_DIM = 480
VIDEO_DIM = 4

# Picked arbitrarily for now -- we have no evidence yet that this is the
# "right" size. That's fine; you can't know the right hidden width before you
# have anything running to measure against. This is just enough to move
# forward and get real numbers to look at.
HIDDEN_DIM = 16

# Matches data_simulations.py's toy labels: vowel vs. consonant.
N_CLASSES = 2


def make_network(seed=0):
    """Step 4: add the output layer and the fusion connection.

    Fusion choice: a single joint edge, Predict([l_h_aud, l_h_vid], l_out),
    rather than two separate Predict edges. This is NOT a settled "correct"
    answer -- our own test file's comment says joint fusion avoids one
    modality dominating, but omni-pcn's own official multimodal_alphanum.py
    example ships with the split-edge version instead. We're picking joint
    to move forward with something concrete; treat this as a hypothesis to
    actually test later (e.g. by shuffling/zeroing one modality's input and
    checking whether it still affects the output), not a proven choice.
    """
    net = pcn.PCNetwork(seed=seed)
    with net:
        l_aud = pcn.AuditoryInput(
            n_samples=4096, sr=16000, n_fft=512, hop=256, n_mels=32,
            griffin_lim_iters=4, label="aud",
        )
        l_vid = pcn.Layer(dim=VIDEO_DIM, activation=pcn.Direct(), label="video")

        l_h_aud = pcn.Layer(dim=HIDDEN_DIM, activation=pcn.LeakyRelu(), label="hidden_aud")
        l_h_vid = pcn.Layer(dim=HIDDEN_DIM, activation=pcn.LeakyRelu(), label="hidden_vid")

        # "audio predicts its own hidden representation", same for video.
        # Explicit labels now (rather than leaving them default) because,
        # per omni-pcn's own examples, connection labels are how you later
        # target specific edges for different learning rates/optimizers.
        pcn.Predict(l_aud, l_h_aud, label="audio_bottom_up")
        pcn.Predict(l_vid, l_h_vid, label="video_bottom_up")

        # Softmax activation: output values must look like class
        # probabilities (all positive, summing to 1) -- matches classifying
        # into N_CLASSES categories, same as the toy stand-in network in
        # tests/test_pcn.py.
        l_out = pcn.Layer(dim=N_CLASSES, activation=pcn.Softmax(), label="output")
        pcn.Predict([l_h_aud, l_h_vid], l_out, label="joint_fusion")
    net.build()
    return net, l_aud, l_vid, l_h_aud, l_h_vid, l_out


if __name__ == "__main__":
    net, l_aud, l_vid, l_h_aud, l_h_vid, l_out = make_network()
    print("net.build() succeeded")
    print(f"  audio input dim  = {l_aud.dim}")
    print(f"  video input dim  = {l_vid.dim}")
    print(f"  audio hidden dim = {l_h_aud.dim}")
    print(f"  video hidden dim = {l_h_vid.dim}")
    print(f"  output dim       = {l_out.dim}")

    # --- Step 6: run one batch of fake (but correctly-shaped) data through it. ---
    # Nothing here checks whether the model is *good* -- only whether it can
    # settle on real data without falling over. l_aud is a sensory front-end,
    # so we clamp it with a *raw* waveform (matching n_samples=4096); l_vid is
    # a plain Direct layer, so we clamp it with feature-shaped data directly.
    print("\nrunning a tiny batch through the network...")
    rng = np.random.default_rng(0)
    batch_size = 4
    raw_audio = rng.standard_normal((batch_size, 4096)).astype(np.float32)
    raw_video = rng.standard_normal((batch_size, VIDEO_DIM)).astype(np.float32)

    sim = pcn.Simulation(net)
    result = sim.test(
        [{"audio": jnp.asarray(raw_audio), "video": jnp.asarray(raw_video)}],
        data_map={l_aud: "audio", l_vid: "video"},
        iterations_per_sample=5,
        record_map={"pred": (l_out.value, lambda v: v)},
    )
    pred = np.asarray(result["pred"][0])
    print(f"  output shape = {pred.shape} (expect ({batch_size}, {N_CLASSES}))")
    print(f"  any NaN/Inf? = {not bool(np.all(np.isfinite(pred)))}")
    print(f"  sample row 0 = {pred[0]}")
