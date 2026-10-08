"""applemote run: listen on the headset mic while it is free, emit volume keys.

States:
  unplugged          nothing captured, user gains untouched
  plugged, mic free  gains pinned to the detection operating point, listening
  plugged, mic busy  another app records from the headset mic: user gains are
                     given back and detection pauses (a clipped signal can
                     produce wrong-direction keys, which is worse than none)
"""
import os
import selectors
import signal
import subprocess
import sys
import time

import numpy as np

from . import detect, system
from .system import log


class Daemon:
    def __init__(self, cfg, verbose=False):
        self.cfg, self.verbose = cfg, verbose
        self.mixer = system.Mixer(cfg["card"])
        self.gains = system.Gains(self.mixer, cfg)
        self.keys = system.KeyOut(cfg["key_device"])
        self.templates = detect.load_templates(cfg["templates"])
        self.sel = selectors.DefaultSelector()
        self.cap = None
        self.detector = None
        self.scale = 1.0
        self.pending = b""
        self.recheck_at = 0.0
        self.watch(["stdbuf", "-oL", "amixer", "-c", cfg["card"], "events"], self.on_mixer_event)
        self.watch(["stdbuf", "-oL", "pactl", "subscribe"], self.on_pulse_event)

    def watch(self, cmd, handler):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        os.set_blocking(p.stdout.fileno(), False)
        self.sel.register(p.stdout, selectors.EVENT_READ, (handler, [b""]))

    # --- state ----------------------------------------------------------------
    def evaluate(self):
        plugged = self.mixer.jack(self.cfg["jack_control"])
        busy = system.other_consumers(self.cfg["source"]) if plugged else []
        if plugged and not busy:
            self.start()
        else:
            self.stop("headset mic in use by " + ", ".join(busy) if busy else
                      "headset unplugged")

    def start(self):
        if self.cap:
            return
        self.gains.borrow()
        self.scale = 10 ** (-self.gains.total_db() / 20)
        self.detector = detect.Detector(self.templates)
        self.pending = b""
        self.cap = system.capture(self.cfg["source"])
        os.set_blocking(self.cap.stdout.fileno(), False)
        self.sel.register(self.cap.stdout, selectors.EVENT_READ, (self.on_audio, None))
        log(f"listening (capture gain {self.gains.total_db():+.1f} dB)")

    def stop(self, why):
        if not self.cap:
            return
        self.sel.unregister(self.cap.stdout)
        self.cap.terminate()
        self.cap.wait()
        self.cap = None
        self.gains.give_back()
        log(f"paused: {why}")

    # --- inputs ---------------------------------------------------------------
    def on_audio(self, f, _):
        data = f.read()
        if not data:
            log("capture ended; restarting")
            self.stop("capture ended")
            self.recheck_at = time.monotonic() + 1
            return
        data = self.pending + data
        n = len(data) // 8 * 8
        self.pending = data[n:]
        x = np.frombuffer(data[:n], dtype="<f4")[0::2] * self.scale
        for ev in self.detector.process(x):
            if ev["key"]:
                log(f"KEY_{ev['key']} ({ev['cls']} ncc={ev['ncc']:.2f} scale={ev['scale']:.2f})")
                self.keys.send(ev["key"])
            elif self.verbose:
                log(f"{ev['cls']} ncc={ev['ncc']:.2f} scale={ev['scale']:.2f}")
        if self.verbose:
            for ev in self.detector.near_misses:
                log(f"near miss {ev['cls']} ncc={ev['ncc']:.2f} scale={ev['scale']:.2f}")
        self.detector.near_misses.clear()

    def lines(self, f, buf):
        *lines, buf[0] = (buf[0] + (f.read() or b"")).split(b"\n")
        return [ln.decode(errors="replace") for ln in lines]

    def on_mixer_event(self, f, buf):
        for line in self.lines(f, buf):
            if "event value" not in line:
                continue
            if f"'{self.cfg['jack_control']}'" in line:
                self.recheck_at = time.monotonic() + 0.1
            elif self.cap and any(f"'{c}" in line for c, _ in self.gains.pin):
                self.gains.user_changed()

    def on_pulse_event(self, f, buf):
        if any("source-output" in line for line in self.lines(f, buf)):
            self.recheck_at = time.monotonic() + 0.1

    def run(self):
        self.evaluate()
        if not self.cap:
            log("waiting for a free headset mic")
        while True:
            timeout = max(0.0, self.recheck_at - time.monotonic()) if self.recheck_at else None
            for key, _ in self.sel.select(timeout):
                handler, buf = key.data
                handler(key.fileobj, buf)
            if self.recheck_at and time.monotonic() >= self.recheck_at:
                self.recheck_at = 0.0
                self.evaluate()

    def shutdown(self, *_):
        self.stop("exiting")
        self.keys.close()
        sys.exit(0)


def main(args):
    cfg = system.load_config()
    if not cfg["source"] or not cfg["templates"]:
        log(f"not configured: run 'applemote setup' (config: {system.CONFIG_FILE})")
        return 1
    d = Daemon(cfg, args.verbose)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, d.shutdown)
    try:
        d.run()
    finally:
        d.stop("exiting")
