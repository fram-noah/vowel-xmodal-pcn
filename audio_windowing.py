"""
Slice a raw audio signal into audio windows aligned to video frames.

Video is the source of truth for timing: however many samples the audio
track actually has, `audio_window` always returns exactly `n_frames`
(x `sub_windows_per_frame`) windows, aligned to each frame's own timestamp
(not a fixed integer hop -- real video framerates like 29.97/23.976 don't
divide fs evenly, and accumulating a rounded hop would drift over time).

By default there's one audio window per video frame. But audio carries
information at a much higher rate than video -- a phoneme transition
routinely happens within a single video frame's ~33ms span, or straddles the
boundary between two frames -- so compressing all of a frame's audio into one
window can throw away exactly the temporal detail this project cares about.
`sub_windows_per_frame` (k) lets a caller pull several audio sub-windows per
video frame instead of one, without changing anything else about how
windowing works: every sub-window still goes through the same strategy /
edge-handling logic, just spaced `hop/k` apart instead of `hop` apart.

Multiple windowing strategies are supported (see WINDOW_STRATEGIES) so callers
can choose the trade-off that fits their encoder/model, without changing the
calling convention. Pipeline, each stage independent of the ones before it:

    (frame index, sub-window index) -> timestamp -> audio sample position
        -> window bounds (strategy) -> actual array (edge handling)

Scaling note for later: this assumes every frame gets the *same* number of
sub-windows (k is one constant, not chosen per-frame), which is what makes a
plain 3-D (n_frames, k, window_samples) array work -- frame membership is
just an array axis, no bookkeeping needed. If a future design wants a
variable number of sub-windows per frame (e.g. more during rapid speech,
fewer during silence), that rectangular shape stops working; the fix at that
point is a flat (total_windows, window_samples) array plus a parallel
`frame_of[i]` index array recording which frame each row belongs to, rather
than trying to force a ragged case into this shape.
"""

import numpy as np

VALID_EDGE_MODES = ("pad", "error")


# ---------------------------------------------------------------------------
# Timing: turn a (frame, sub-window) pair into a single audio sample position.
# ---------------------------------------------------------------------------

def frame_sample(i, fs, framerate, k=1, j=0):
    """Audio sample position corresponding to the center of sub-window j (of
    k) within video frame i. With the defaults (k=1, j=0) this is just the
    center of frame i itself.

    Computed from each sub-window's own timestamp (i + (j+0.5)/k) / framerate,
    not by accumulating a fixed hop -- so non-integer framerates (29.97,
    23.976...) don't drift, they just round independently each time.
    """
    timestamp_seconds = (i + (j + 0.5) / k) / framerate
    return round(timestamp_seconds * fs)


# ---------------------------------------------------------------------------
# Strategies: given a center sample and a window width, decide the window's
# (start, end) sample range. Each one only needs a center + a spacing value
# ("hop") -- neither knows or cares whether that spacing is a whole video
# frame or a fraction of one, which is what lets sub_windows_per_frame work
# without changing any of these.
# ---------------------------------------------------------------------------

def _frame_start_bounds(center, window_samples, hop):
    """Window starts at the beginning of its unit (half a hop before its
    center) and covers exactly one hop by default. Only non-overlapping and
    gapless when window_samples == hop; wider or narrower values will
    overlap neighbors or leave gaps between them."""
    start = center - hop // 2
    end = start + window_samples
    return start, end


def _centered_bounds(center, window_samples, hop):
    """Center the window on its unit's timestamp. window_samples may exceed
    hop so neighboring units deliberately share context (useful for spectral
    analysis, which wants more than one hop's worth of signal)."""
    start = center - window_samples // 2
    end = start + window_samples
    return start, end


def _causal_bounds(center, window_samples, hop):
    """Window ends at its unit's timestamp, extending backward only. Never
    looks at future audio -- useful for streaming/online use, at the cost of
    a lag relative to the unit it's paired with."""
    end = center
    start = end - window_samples
    return start, end


WINDOW_STRATEGIES = {
    "frame_start": _frame_start_bounds,
    "centered": _centered_bounds,
    "causal": _causal_bounds,
}


# ---------------------------------------------------------------------------
# Edge handling: place one window's samples into its output row, clipping to
# whatever the real signal has and padding (or erroring) for the rest.
# ---------------------------------------------------------------------------

def _fill_one_window(audiosig, out_row, mask_row, start, end, edge, window_label):
    """Copy audiosig[start:end] into out_row (marking mask_row True for real
    samples), clipping to audiosig's bounds. out_row/mask_row are already
    zero-filled, so anything left uncopied is padding by construction.

    Raises ValueError instead of padding if edge == "error" and the window
    needs any padding to fit.
    """
    signal_length = len(audiosig)
    read_start, read_end = max(start, 0), min(end, signal_length)
    needs_padding = read_start > start or read_end < end or read_start >= read_end

    if needs_padding and edge == "error":
        raise ValueError(f"window {window_label} needs padding but edge='error'")
    if read_start >= read_end:
        return  # window falls entirely outside audiosig -- leave this row as all padding

    offset_into_row = read_start - start
    n_real_samples = read_end - read_start
    out_row[offset_into_row : offset_into_row + n_real_samples] = audiosig[read_start:read_end]
    mask_row[offset_into_row : offset_into_row + n_real_samples] = True


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def audio_window(audiosig, fs, framerate, n_frames,
                  strategy="frame_start", window_samples=None, edge="pad",
                  sub_windows_per_frame=1):
    """
    Parameters
    ----------
    audiosig : 1-D array of raw audio samples.
    fs : sample rate of `audiosig`, in Hz.
    framerate : video frame rate, in Hz. May be non-integer (29.97, ...).
    n_frames : number of video frames to produce windows for. Video is the
        source of truth here -- pass the frame count from the decoded video,
        not something derived from len(audiosig), since audio/video track
        lengths can differ by a little slop.
    strategy : one of WINDOW_STRATEGIES.keys(). Controls how each window's
        sample range is placed relative to its unit's timestamp.
    window_samples : samples per window. Defaults to the unit hop
        (fs / framerate / sub_windows_per_frame, rounded). Pass a larger
        value with strategy="centered" or "causal" to give each window more
        context than one unit's worth of audio.
    edge : "pad" (zero-pad windows that run past the start/end of
        `audiosig`) or "error" (raise if any window would need padding).
    sub_windows_per_frame : how many audio windows to produce per video
        frame (k). Default 1 matches one window per frame exactly as before.
        k > 1 divides each frame's time span into k equal sub-units and
        produces one window per sub-unit, for finer temporal resolution than
        the video frame rate provides -- useful since audio-relevant events
        (like phoneme transitions) can happen faster than one video frame.

    Returns
    -------
    windows : np.ndarray. Shape (n_frames, window_samples) when
        sub_windows_per_frame == 1 (unchanged from before); shape
        (n_frames, sub_windows_per_frame, window_samples) when > 1, so frame
        membership stays a plain array axis instead of needing separate
        bookkeeping (see the scaling note in the module docstring for what
        to do instead if k ever needs to vary per frame).
    mask : np.ndarray of bool, same shape as `windows` -- True where a sample
        is real audio, False where it's edge-padding. Padding is always 0 in
        `windows`, but 0 can also be a genuine sample value, so don't rely on
        `windows == 0` to tell them apart -- use `mask` (this matters for
        this project specifically: a PCN reasoning about degraded/missing
        input needs to tell "silence" apart from "no data here").
    """
    _validate_args(strategy, edge, sub_windows_per_frame)
    audiosig = np.asarray(audiosig)

    k = sub_windows_per_frame
    hop = fs / framerate                                     # samples per video frame
    unit_hop = hop / k                                        # samples per sub-window
    if window_samples is None:
        window_samples = round(unit_hop)
    bounds_fn = WINDOW_STRATEGIES[strategy]

    windows = np.zeros((n_frames, k, window_samples), dtype=audiosig.dtype)
    mask = np.zeros((n_frames, k, window_samples), dtype=bool)

    for i in range(n_frames):
        for j in range(k):
            center = frame_sample(i, fs, framerate, k, j)
            start, end = bounds_fn(center, window_samples, unit_hop)
            _fill_one_window(
                audiosig, windows[i, j], mask[i, j],
                start=int(start), end=int(end), edge=edge,
                window_label=f"(frame {i}, sub {j})",
            )

    if k == 1:
        # Backward-compatible 2-D shape when there's exactly one window per
        # frame -- squeeze away the now-trivial sub-window axis.
        return windows[:, 0, :], mask[:, 0, :]
    return windows, mask


def _validate_args(strategy, edge, sub_windows_per_frame):
    if strategy not in WINDOW_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {list(WINDOW_STRATEGIES)}")
    if edge not in VALID_EDGE_MODES:
        raise ValueError(f"edge must be one of {VALID_EDGE_MODES}, got {edge!r}")
    if sub_windows_per_frame < 1:
        raise ValueError(f"sub_windows_per_frame must be >= 1, got {sub_windows_per_frame}")
