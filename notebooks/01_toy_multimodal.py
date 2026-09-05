## Exploration 1: a toy multimodal vowel/consonant classifier in pcn
## Simulated data, no download. Mirrors examples/multimodal_alphanum.py at toy scale.
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax.numpy as jnp
import optax
import pcn

SEED = 0
N_TRAIN, N_TEST = 800, 200
BATCH = 40
N_MEL = 8          # fake mel bands
N_VIDEO = 4        # fake mouth features: opening, width, rounding, jaw
HIDDEN = 32
N_EPOCHS = 15
N_ITERS = 20       # inference+learning iterations per batch
N_EVAL_ITERS = 50

# ---------------------------------------------------------------- simulate data
def simulate(n, rng, audio_noise=0.3, video_noise=0.3):
    """Each frame is either a vowel (1) or a consonant (0).

    audio: vowels have energy concentrated in two 'formant' bands, consonants are
           flat / high-band. video: vowels have an open mouth, consonants closed.
    """
    y = rng.integers(0, 2, size=n)
    audio = np.zeros((n, N_MEL), np.float32)
    video = np.zeros((n, N_VIDEO), np.float32)
    for i, is_vowel in enumerate(y):
        if is_vowel:
            f1, f2 = rng.integers(0, 3), rng.integers(3, 6)   # two formant peaks
            audio[i, f1] += 1.0
            audio[i, f2] += 0.8
            video[i] = [1.0, 0.7, rng.uniform(0, 1), 0.8]      # open mouth
        else:
            audio[i, 5:] += 0.6                                # high-band noise burst
            video[i] = [0.1, 0.4, 0.2, 0.1]                    # mouth mostly closed
    audio += rng.normal(0, audio_noise, audio.shape).astype(np.float32)
    video += rng.normal(0, video_noise, video.shape).astype(np.float32)
    label = np.eye(2, dtype=np.float32)[y]                     # one-hot
    return {"audio": audio, "video": video, "label": label, "label_idx": y}

def batches(d, batch):
    n = len(d["label_idx"])
    return [{k: v[i:i + batch] for k, v in d.items()} for i in range(0, n, batch)]

rng = np.random.default_rng(SEED)
train = batches(simulate(N_TRAIN, rng), BATCH)
test = batches(simulate(N_TEST, rng), BATCH)

# ---------------------------------------------------------------- network
net = pcn.PCNetwork(seed=SEED)
net.config(use_bias=True, learn_precision_weights=False, learn_precision_bias=False)
with net:
    l_aud = pcn.Layer(dim=N_MEL, activation=pcn.Direct(), label="audio")
    l_vid = pcn.Layer(dim=N_VIDEO, activation=pcn.Direct(), label="video")
    l_h_aud = pcn.Layer(dim=HIDDEN, activation=pcn.LeakyRelu(), label="hidden_aud")
    l_h_vid = pcn.Layer(dim=HIDDEN, activation=pcn.LeakyRelu(), label="hidden_vid")
    l_label = pcn.Layer(dim=2, activation=pcn.Softmax(temperature=0.5), label="label")
    # bottom-up: each modality rises through its own hidden layer into ONE
    # summed label prediction (joint fusion). NOTE: writing this as two separate
    # edges, Predict(l_h_aud, l_label) + Predict(l_h_vid, l_label), gives a
    # product-of-experts with two error factors instead, and in this toy the
    # audio pathway then never learns (audio-only accuracy stays at chance).
    pcn.Predict(l_aud, l_h_aud)
    pcn.Predict(l_vid, l_h_vid)
    pcn.Predict([l_h_aud, l_h_vid], l_label)
net.build()

# ---------------------------------------------------------------- train
sim = pcn.Simulation(net)
values_opt = optax.sgd(0.1)
params_opt = optax.adam(1e-2)
full_map = {l_aud: "audio", l_vid: "video", l_label: "label"}

def batch_accuracy(pred, label):
    return float(np.mean(np.argmax(pred, -1) == np.argmax(label, -1)))

def classify(cond_map):
    record = {"acc": ((l_label.value, "label"), batch_accuracy)}
    r = sim.test(test, data_map=cond_map, record_map=record,
                 iterations_per_sample=N_EVAL_ITERS, verbose=False,
                 values_optimizer=values_opt)
    return 100 * np.mean(r["acc"])

# Occluded modality = clamped to zeros (the same thing training saw).
def zeroed(bs, key):
    return [{**b, key: np.zeros_like(b[key])} for b in bs]

conditions = {"audio+video": {l_aud: "audio", l_vid: "video"},
              "audio only": {l_aud: "audio", l_vid: "video_zero"},
              "video only": {l_aud: "audio_zero", l_vid: "video"}}

def add_zero_keys(bs):
    return [{**b, "audio_zero": np.zeros_like(b["audio"]),
             "video_zero": np.zeros_like(b["video"])} for b in bs]
test = add_zero_keys(test)

# Modality dropout: with prob DROP_P one modality is clamped to ZEROS for that
# batch (an 'occluded' sensor), so the label error has to be explained through
# the other modality. Leaving the layer *free* instead does not work: a free
# layer relaxes to whatever explains the clamped label, absorbing the error, so
# the other pathway's weights never need to change. Set DROP_P = 0.0 to watch
# the network ignore audio entirely (video is the easier modality).
DROP_P = 0.5
layers = {"audio": l_aud, "video": l_vid}

for epoch in range(N_EPOCHS):
    for b in train:
        dmap = full_map
        if rng.random() < DROP_P:
            key = "audio" if rng.random() < 0.5 else "video"
            b = {**b, key: np.zeros_like(b[key])}
        sim.train([b], data_map=dmap, epochs=1, iterations_per_sample=0,
                  learning_iterations_per_sample=N_ITERS, verbose=False,
                  params_optimizer=params_opt, values_optimizer=values_opt)
    if (epoch + 1) % 5 == 0:
        accs = {k: classify(m) for k, m in conditions.items()}
        print(f"epoch {epoch + 1:2d} | " + " | ".join(f"{k} {v:5.1f}%" for k, v in accs.items()))

# ---------------------------------------------------------------- degrade audio
print("\naudio degraded at test time (video intact):")
for noise in [0.3, 1.0, 2.0, 4.0]:
    noisy = add_zero_keys(batches(simulate(N_TEST, np.random.default_rng(1), audio_noise=noise), BATCH))
    test_backup, test = test, noisy
    print(f"  audio noise {noise:.1f}: both {classify(conditions['audio+video']):5.1f}%"
          f"  audio-only {classify(conditions['audio only']):5.1f}%")
    test = test_backup
