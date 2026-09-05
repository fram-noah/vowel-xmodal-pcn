import os; os.environ["JAX_PLATFORMS"]="cpu"
import numpy as np, jax.numpy as jnp, optax, pcn

SEED = 0
N_TRAIN, N_TEST = 160, 40    # utterances
BATCH = 20
T = 24                       # frames per utterance
SR, CLIP, N_MELS, N_VIDEO = 16000, 4096, 16, 4

front = pcn.PCNetwork(seed=SEED)
with front:
    ear = pcn.AuditoryInput(n_samples=CLIP, sr=SR, n_mels=N_MELS, label="ear")
AUDIO_DIM = ear.dim

def vowel_wave(rng, t, f0, f1, f2, phase):
    wav = np.zeros_like(t)
    for k in range(1, 20):
        fk = k * f0
        gain = np.exp(-((fk - f1) / 200) ** 2) + 0.6 * np.exp(-((fk - f2) / 300) ** 2)
        wav += gain * np.sin(2 * np.pi * fk * t + phase[k])
    return wav

def consonant_wave(rng, t, age):
    noise = np.diff(rng.standard_normal(len(t)), prepend=0)
    env = np.exp(-(t + age) / 0.06)
    return noise * env

def make_segments(rng, T):
    """Alternating consonant/vowel runs that fill T frames. Returns per-frame label."""
    y = np.zeros(T, int); i = 0; is_vowel = rng.random() < 0.5
    while i < T:
        run = rng.integers(3, 8) if is_vowel else rng.integers(1, 4)
        y[i:i+run] = int(is_vowel); i += run; is_vowel = not is_vowel
    return y

def simulate(n, rng, audio_noise=0.05, video_noise=0.15):
    t = np.arange(CLIP) / SR
    waves = np.zeros((n, T, CLIP), np.float32)
    video = np.zeros((n, T, N_VIDEO), np.float32)
    labels = np.zeros((n, T), int)
    for u in range(n):
        y = make_segments(rng, T); labels[u] = y
        mouth = np.array([0.1, 0.4, 0.2, 0.1])          # start closed
        params = None; age = 0.0
        for f in range(T):
            if f == 0 or y[f] != y[f-1]:                  # new segment: new sound
                params = (rng.uniform(180, 300), rng.uniform(300, 800), rng.uniform(1000, 2500),
                          rng.uniform(0, 2*np.pi, 20)); age = 0.0
            if y[f]:
                w = vowel_wave(rng, t, *params)
                target = np.array([1.0, 0.7, params[2] / 2500, 0.8])
            else:
                w = consonant_wave(rng, t, age)
                target = np.array([0.1, 0.4, 0.2, 0.1])
            age += CLIP / SR
            w = w / (np.abs(w).max() + 1e-6)
            waves[u, f] = w + rng.normal(0, audio_noise, CLIP)
            mouth = mouth + 0.6 * (target - mouth)        # mouth lags the sound
            video[u, f] = mouth + rng.normal(0, video_noise, N_VIDEO)
    audio = np.asarray(ear.encode(jnp.asarray(waves.reshape(-1, CLIP)))).reshape(n, T, AUDIO_DIM)
    label = np.eye(2, dtype=np.float32)[labels]
    return {"audio": audio, "video": video, "label": label, "label_idx": labels, "wave": waves}

def batches(d, batch):
    n = len(d["label_idx"])
    return [{k: v[i:i+batch] for k, v in d.items()} for i in range(0, n, batch)]

rng = np.random.default_rng(SEED)
tr_raw = simulate(N_TRAIN, rng); te_raw = simulate(N_TEST, rng)
mu, sd = tr_raw["audio"].mean((0,1)), tr_raw["audio"].std((0,1)) + 1e-6
for d in (tr_raw, te_raw): d["audio"] = ((d["audio"] - mu) / sd).astype(np.float32)
train, test = batches(tr_raw, BATCH), batches(te_raw, BATCH)
print({k: v.shape for k, v in train[0].items()})
print("labels of utterance 0:", "".join("V" if v else "c" for v in train[0]["label_idx"][0]))
print("mouth opening, utt 0:", np.round(train[0]["video"][0, :, 0], 1))

# ---- can the package train on (B, T, dim) with temporal clamp? ----
K = 8
HIDDEN = 32
net = pcn.PCNetwork(seed=SEED)
net.config(use_bias=True, learn_precision_weights=False, learn_precision_bias=False)
with net:
    l_aud = pcn.Layer(dim=AUDIO_DIM, activation=pcn.Direct(), label="audio")
    l_vid = pcn.Layer(dim=N_VIDEO, activation=pcn.Direct(), label="video")
    l_h = pcn.Layer(dim=HIDDEN, activation=pcn.Tanh(), label="hidden")
    l_label = pcn.Layer(dim=2, activation=pcn.Softmax(temperature=0.5), label="label")
    pcn.Predict([l_aud, l_vid], l_h)
    pcn.Predict(l_h, l_h, delay=1, delay_unit="timestep", label="transition")
    pcn.Predict(l_h, l_label)
net.build()
sim = pcn.Simulation(net)
vo, po = optax.sgd(0.1), optax.adam(1e-2)
full = {l_aud: "audio", l_vid: "video", l_label: "label"}
def per_frame_predictions(dmap):
    """Run the temporal clamp, log every iteration, keep the label value at the
    last relaxation step of each timestep. Returns (utterances, T, 2)."""
    out = []
    for b in test:
        sim.test([b], data_map=dmap, iterations_per_sample=K*T, log_every=1,
                 return_logs=True, verbose=False, values_optimizer=vo)
        v = np.asarray(sim.logs["values"][0][l_label._idx])   # (K*T, B, 2)
        out.append(v[K-1::K].transpose(1, 0, 2))              # (B, T, 2)
    return np.concatenate(out)
def acc(dmap):
    pred = per_frame_predictions(dmap).argmax(-1)
    truth = np.concatenate([b["label_idx"] for b in test])
    return 100 * np.mean(pred == truth)
import time; t0 = time.time()
for ep in range(10):
    for b in train:
        b = dict(b)
        if rng.random() < 0.5:
            key = "audio" if rng.random() < 0.5 else "video"
            b[key] = np.zeros_like(b[key])
        sim.train([b], data_map=full, epochs=1, iterations_per_sample=K*T, learning_iterations_per_sample=0,
                  verbose=False, params_optimizer=po, values_optimizer=vo)
    if (ep+1) % 5 == 0:
        z = lambda bs, key: [{**bb, key: np.zeros_like(bb[key])} for bb in bs]
        both = acc({l_aud: "audio", l_vid: "video"})
        test_b = test; test = z(test_b, "video"); a_only = acc({l_aud: "audio", l_vid: "video"}); test = test_b
        test = z(test_b, "audio"); v_only = acc({l_aud: "audio", l_vid: "video"}); test = test_b
        print(f"epoch {ep+1} | both {both:5.1f}% | audio only {a_only:5.1f}% | video only {v_only:5.1f}% | {time.time()-t0:.0f}s")
