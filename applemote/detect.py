"""Remote-button detection on the headset mic line.

Without Apple's identification exchange the remote still disturbs the mic line
on every volume press and release.  After the codec's AC coupling each of the
four cases (Vol+/Vol- x press/release) is a stable ~10 ms waveform, unlike
acoustic clicks from the mic inside the remote.  Detection:

  1. find fast edges on the raw signal,
  2. low-pass at 2 kHz (removes most acoustic/mechanical energy),
  3. normalized cross-correlation against per-class templates (+-1 ms lag),
  4. accept the best class if correlation and fitted amplitude are in range.

All signals here are mono float at 48 kHz in "unity-gain units": captured
samples divided by the codec's total capture gain.
"""
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

RATE = 48000
MS = RATE // 1000
PRE = 1 * MS                # template window starts 1 ms before the edge
LEN = 12 * MS               # ... and is 12 ms long
BASE = (20 * MS, 2 * MS)    # baseline: 20..2 ms before the edge
LAG = 1 * MS                # alignment search, +-1 ms
CLASSES = ("UP_PRESS", "UP_RELEASE", "DOWN_PRESS", "DOWN_RELEASE")
KEYS = {"UP_PRESS": "VOLUMEUP", "DOWN_PRESS": "VOLUMEDOWN"}

_B, _A = butter(2, 2000, fs=RATE)


def lowpass(x, zi=None):
    if zi is None:
        zi = lfilter_zi(_B, _A) * (x[0] if len(x) else 0.0)
    return lfilter(_B, _A, x, zi=zi)


def onsets(raw, edge_min=4e-4, edge_k=25.0, floor=None):
    """Indices where the sample-to-sample step is large in absolute terms and
    relative to the local noise floor."""
    d = np.abs(np.diff(raw))
    if floor is None:
        floor = np.median(d)
    return np.flatnonzero((d > edge_min) & (d > edge_k * (floor + 1e-12))) + 1


def window(lp, o):
    """Low-passed window around edge o (with +-LAG margin), baseline removed."""
    w = lp[o - PRE - LAG:o - PRE + LEN + LAG]
    return w - lp[o - BASE[0]:o - BASE[1]].mean()


def best_match(w, tpl):
    """Best normalized correlation of tpl within w over +-LAG -> (ncc, scale)."""
    segs = np.lib.stride_tricks.sliding_window_view(w[:LEN + 2 * LAG], LEN)
    tn = np.linalg.norm(tpl)
    dots = segs @ tpl
    ncc = dots / ((np.linalg.norm(segs, axis=1) + 1e-15) * tn)
    k = int(np.argmax(ncc))
    return float(ncc[k]), float(dots[k] / (tn * tn))


class Detector:
    """Streaming detector.  Feed blocks with process(); returns event dicts
    {"t", "cls", "ncc", "scale", "key"} where key is VOLUMEUP/VOLUMEDOWN for
    presses and None for releases."""

    def __init__(self, templates, ncc_min=0.85, scale=(0.35, 3.0), refractory_ms=20):
        self.tpl = {c: np.asarray(templates[c]) for c in CLASSES if c in templates}
        self.ncc_min, self.scale = ncc_min, scale
        self.refractory = refractory_ms * MS
        self.need_pre = BASE[0] + LAG
        self.need_post = LEN - PRE + LAG + 2 * MS
        self.raw = np.zeros(0)
        self.lp = np.zeros(0)
        self.zi = None
        self.base = 0           # absolute sample index of raw[0]
        self.scan_from = 0
        self.next_ok = 0
        self.near_misses = []   # rejected edges with ncc >= 0.6, for diagnostics

    def process(self, block):
        block = np.asarray(block, dtype=np.float64)
        if not len(block):
            return []
        if self.zi is None:
            self.zi = lfilter_zi(_B, _A) * block[0]
        lpb, self.zi = lfilter(_B, _A, block, zi=self.zi)
        self.raw = np.concatenate([self.raw, block])
        self.lp = np.concatenate([self.lp, lpb])
        events = []
        lo = max(self.scan_from, self.base + self.need_pre) - self.base
        hi = len(self.raw) - self.need_post
        if hi > lo:
            floor = np.median(np.abs(np.diff(self.raw[max(0, lo - 50 * MS):hi])))
            for o in onsets(self.raw[lo - 1:hi], floor=floor) + lo - 1:
                if o + self.base < self.next_ok:
                    continue
                # the +-LAG search already covers the next LAG samples of this edge
                self.next_ok = o + self.base + LAG
                ev = self.classify(window(self.lp, o))
                if ev is None:
                    continue
                ev["t"] = (o + self.base) / RATE
                if ev.pop("ok"):
                    events.append(ev)
                    self.next_ok = o + self.base + self.refractory
                else:
                    self.near_misses.append(ev)
            self.scan_from = hi + self.base
        keep = self.need_pre + self.need_post + 4 * MS
        if len(self.raw) > keep:
            drop = len(self.raw) - keep
            self.raw, self.lp = self.raw[drop:], self.lp[drop:]
            self.base += drop
        return events

    def classify(self, w):
        scores = {c: best_match(w, t) for c, t in self.tpl.items()}
        cls, (ncc, scale) = max(scores.items(), key=lambda kv: kv[1][0])
        ok = ncc >= self.ncc_min and self.scale[0] <= scale <= self.scale[1]
        if not ok and ncc < 0.6:
            return None
        return {"cls": cls, "ncc": ncc, "scale": scale, "key": KEYS.get(cls) if ok else None, "ok": ok}


def run(x, templates, block=480):
    """Run the detector over a whole signal; returns accepted events."""
    det = Detector(templates)
    events = []
    for s in range(0, len(x), block):
        events += det.process(x[s:s + block])
    return events


# --- calibration ----------------------------------------------------------

def candidate_windows(x, t0, t1, rate=RATE):
    """Edges in [t0, t1) of x -> (times, low-passed windows with +-LAG margin)."""
    lp, _ = lowpass(x)
    a, b = int(t0 * rate), int(t1 * rate)
    floor = np.median(np.abs(np.diff(x[a:b])))
    times, wins, last = [], [], -10 ** 9
    for o in onsets(x[a:b], floor=floor) + a:
        if o - last < 20 * MS or o - BASE[0] - LAG < 0 or o + LEN + LAG > len(x):
            continue
        last = o
        times.append(o / rate)
        wins.append(window(lp, o))
    return np.array(times), wins


def _largest_cluster(wins, idx, ncc_min):
    core = [w[LAG:LAG + LEN] for w in wins]
    best = []
    for i in idx:
        group = [j for j in idx if best_match(wins[j], core[i])[0] >= ncc_min]
        if len(group) > len(best):
            best = group
    return best


def _average(wins, group):
    seed = wins[group[0]][LAG:LAG + LEN]
    aligned = []
    for j in group:
        w = wins[j]
        k = max(range(2 * LAG + 1), key=lambda k: float(w[k:k + LEN] @ seed))
        aligned.append(w[k:k + LEN])
    return np.mean(aligned, axis=0)


def learn_templates(x, segments, ncc_min=0.85, rate=RATE):
    """Learn templates from a calibration recording.

    x: unity-gain mono signal.  segments: {"UP": (t0, t1), "DOWN": (t0, t1)},
    each holding presses of only that button.  In each segment the strongest
    edges form two clusters of similar shapes; the one that comes first in
    each press/release pair is the press.
    Returns ({class: template}, {button: report}).  Raises ValueError when a
    segment does not contain enough clean presses.
    """
    templates, report = {}, {}
    for button, (t0, t1) in segments.items():
        times, wins = candidate_windows(x, t0, t1, rate)
        if len(wins) < 6:
            raise ValueError(f"{button}: only {len(wins)} edges found; were the presses recorded?")
        energy = np.array([np.linalg.norm(w) for w in wins])
        strong = [i for i in range(len(wins)) if energy[i] > 0.3 * np.percentile(energy, 90)]
        a = _largest_cluster(wins, strong, ncc_min)
        b = _largest_cluster(wins, [i for i in strong if i not in a], ncc_min)
        if len(a) < 3 or len(b) < 3:
            raise ValueError(f"{button}: clusters of {len(a)} and {len(b)} similar edges; "
                             "need at least 3 clean presses and releases")
        ta, tb = times[a], times[b]
        # press -> release is the short gap, release -> next press the long one
        gap_ab = np.median([min([t for t in tb if t > s] or [np.inf]) - s for s in ta])
        gap_ba = np.median([min([t for t in ta if t > s] or [np.inf]) - s for s in tb])
        press, release = (a, b) if gap_ab < gap_ba else (b, a)
        templates[f"{button}_PRESS"] = _average(wins, press)
        templates[f"{button}_RELEASE"] = _average(wins, release)
        report[button] = {"edges": len(wins), "presses": len(press), "releases": len(release)}
    return templates, report


def validate(x, templates, segments, rate=RATE):
    """Keys produced in each segment -> {segment: {"VOLUMEUP": n, "VOLUMEDOWN": n}}."""
    events = run(x, templates)
    out = {}
    for name, (t0, t1) in segments.items():
        keys = [e["key"] for e in events if e["key"] and t0 <= e["t"] < t1]
        out[name] = {"VOLUMEUP": keys.count("VOLUMEUP"), "VOLUMEDOWN": keys.count("VOLUMEDOWN")}
    return out


def load_templates(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files if k in CLASSES}


def save_templates(path, templates, **meta):
    np.savez(path, **templates, **{f"meta_{k}": np.array(v) for k, v in meta.items()})
