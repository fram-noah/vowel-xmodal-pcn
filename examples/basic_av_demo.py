"""Smoke-test / demo for the basic audio-visual PCN (``av_pcn``).

Since no licensed audio-visual recordings ship with this repo (see the
top-level README), this script trains and evaluates the architecture on
synthetic audio-visual "song" sequences (``av_pcn.synthetic_data``) instead.
It demonstrates the three BrainHack 2026 goals end to end on a toy problem:

1. familiarize with `omni-pcn` (this whole script is just `pcn` API calls),
2. an architecture that takes time-dependent visual + auditory input
   (`av_pcn.model.build_basic_av_pcn`),
3. a windowing procedure for the time-varying input (temporal clamping: each
   input timestep gets a few relaxation iterations -- see
   ``omni-pcn``'s README, "Clamping options" -- Temporal clamp).

What it does:

1. Trains the network (unsupervised -- no vowel labels used) to predict each
   modality from the other and to predict its own next fused state.
2. Uses the *temporal self-prediction error* of the fusion layer -- how much
   the fused state deviates from what "nothing changed" would predict -- as a
   vowel-boundary detector, and scores it against the synthetic ground truth.
3. Clamps only the audio (then only the visual) stream and reads the
   network's reconstruction of the missing modality, to demonstrate the
   degraded/missing-modality use case described in the project README.

Run:  python examples/basic_av_demo.py   (needs `omni-pcn` installed, e.g.
`uv pip install -e ../omni-pcn` from this repo's root, or add it to this
repo's environment however you prefer -- see ../omni-pcn/README.md.)
"""
import sys
from pathlib import Path

import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from av_pcn import build_basic_av_pcn, generate_av_batch, make_av_dataset
from av_pcn.synthetic_data import drop_modality_spans

N_VOWELS = 3
DIM_AUDIO, DIM_VISUAL, DIM_HIDDEN = 6, 3, 8
T_LEN = 24                 # frames per sequence
BATCH_SIZE = 16
N_EPOCHS = 6
BATCHES_PER_EPOCH = 6
ITERS_PER_TIMESTEP = 3     # relaxation steps per input frame (the "window")
SOFTDROP_P = 0.3           # fraction of batches with one modality span dropped


def train_epoch(sim, layers, dataset, rng, values_opt, params_opt):
    full = {layers["audio"]: "audio", layers["visual"]: "visual"}
    for _ in range(BATCHES_PER_EPOCH):
        batch = generate_av_batch(dataset, BATCH_SIZE, T_LEN, rng)
        data_map = full
        if rng.random() < SOFTDROP_P:   # occasionally nudge one modality instead of clamping it
            key = "audio" if rng.random() < 0.5 else "visual"
            mask = drop_modality_spans(batch[key], drop_prob=1.0, rng=rng)
            mask = np.repeat(mask[..., None], batch[key].shape[-1], axis=-1)
            batch = {**batch, key + "_mask": mask}
            data_map = {**full, layers[key]: (key, key + "_mask")}
        sim.train([batch], data_map=data_map, epochs=1, iterations_per_sample=0,
                  learning_iterations_per_sample=ITERS_PER_TIMESTEP * T_LEN,
                  verbose=False, params_optimizer=params_opt, values_optimizer=values_opt)


def detect_boundaries(sim, layers, edges, dataset, rng, values_opt):
    """Both modalities clamped; use the fusion layer's temporal-prediction
    error trajectory (one value per input frame) as a boundary-detection
    signal, scored against the synthetic ground truth."""
    batch = generate_av_batch(dataset, BATCH_SIZE, T_LEN, rng)
    data_map = {layers["audio"]: "audio", layers["visual"]: "visual"}
    res = sim.test([batch], data_map=data_map, iterations_per_sample=ITERS_PER_TIMESTEP * T_LEN,
                   log_every=ITERS_PER_TIMESTEP, return_logs=True, verbose=False,
                   values_optimizer=values_opt)
    # `edges["temporal"]._idx` is this connection's slot in the per-connection
    # error trajectories; shape (batch, T_LEN, dim_hidden) after return_logs.
    temporal_error = np.asarray(res["errors"][edges["temporal"]._idx])
    surprise = np.linalg.norm(temporal_error, axis=-1)         # (batch, T_LEN)

    z = (surprise - surprise.mean(1, keepdims=True)) / (surprise.std(1, keepdims=True) + 1e-6)
    detected = z > 1.0
    truth = batch["boundary"] > 0.5
    hit = float((detected & truth).sum())
    precision = hit / max(1, detected.sum())
    recall = hit / max(1, truth.sum())
    return surprise, batch, precision, recall


def reconstruct_missing_modality(sim, layers, edges, dataset, rng, values_opt):
    """Clamp only audio (then only visual); read the top-down prediction of
    the other modality's *last frame* and correlate it with that frame's true
    (unclamped) value. ``record_map`` fires once per batch on the final
    relaxation state, i.e. after the last input timestep of the temporal
    clamp -- see the omni-pcn README's "Clamping options"."""
    batch = generate_av_batch(dataset, BATCH_SIZE, T_LEN, rng)
    scores = {}
    for src, tgt in (("audio", "visual"), ("visual", "audio")):
        res = sim.test([batch], data_map={layers[src]: src},
                       iterations_per_sample=ITERS_PER_TIMESTEP * T_LEN, verbose=False,
                       values_optimizer=values_opt,
                       record_map={"pred": ((layers[tgt].value, edges[tgt].error), np.subtract)})
        pred = np.concatenate(res["pred"])                # (batch, dim), last frame only
        true_last_frame = batch[tgt][:, -1, :]
        scores[tgt] = float(np.corrcoef(pred.ravel(), true_last_frame.ravel())[0, 1])
    return scores


def main():
    dataset = make_av_dataset(N_VOWELS, DIM_AUDIO, DIM_VISUAL, seed=0)
    net, layers, edges = build_basic_av_pcn(DIM_AUDIO, DIM_VISUAL, DIM_HIDDEN, seed=1)

    values_opt = optax.adam(0.1)
    params_opt = optax.adamw(1e-3, weight_decay=1e-3)
    import pcn
    sim = pcn.Simulation(net)

    train_rng = np.random.default_rng(42)
    for epoch in range(N_EPOCHS):
        train_epoch(sim, layers, dataset, train_rng, values_opt, params_opt)
        _, _, precision, recall = detect_boundaries(
            sim, layers, edges, dataset, np.random.default_rng(1000 + epoch), values_opt)
        print(f"epoch {epoch + 1}/{N_EPOCHS} | boundary precision {precision:.2f} "
              f"recall {recall:.2f}", flush=True)

    surprise, batch, precision, recall = detect_boundaries(
        sim, layers, edges, dataset, np.random.default_rng(999), values_opt)
    print(f"\nFinal boundary detection (temporal-surprise threshold): "
          f"precision {precision:.2f}, recall {recall:.2f}")

    recon = reconstruct_missing_modality(sim, layers, edges, dataset, np.random.default_rng(7), values_opt)
    print(f"Missing-modality reconstruction (corr with ground truth): "
          f"audio->visual {recon['visual']:.2f}, visual->audio {recon['audio']:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(surprise[0], label="fusion temporal-surprise")
        for t in np.flatnonzero(batch["boundary"][0] > 0.5):
            ax.axvline(t, color="red", alpha=0.3, linestyle="--")
        ax.set_xlabel("frame")
        ax.set_ylabel("|temporal prediction error|")
        ax.set_title("Vowel boundaries (red) vs. fusion-layer surprise")
        ax.legend()
        out = Path(__file__).with_name("basic_av_demo_boundaries.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
