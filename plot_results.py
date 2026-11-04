"""Draws the benchmark chart and a demo image of the detector at two input sizes."""
import json
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import ASSETS

b = json.load(open("results/benchmark.json"))
rows = b["rows"]
sizes = sorted({r["size"] for r in rows})
fig, a = plt.subplots(figsize=(7.5, 4.3), constrained_layout=True)
for (m, th), c in zip([("yolo11n", 1), ("yolo11n", 2), ("yolov8n", 1), ("yolov8n", 2)], ["#12b5cb", "#1a73e8", "#f4b400", "#e8710a"]):
    y = [next(r["fps"] for r in rows if r["model"] == m and r["size"] == s and r["threads"] == th) for s in sizes]
    a.plot(sizes, y, "o-", c=c, lw=2, label=f"{m}, {th} thread{'s' if th > 1 else ''}")
a.set(xlabel="Input size (pixels)", ylabel="Frames per second, model only", title="Smaller input buys speed (x86 cloud CPU, not a Raspberry Pi)")
a.set_xticks(sizes)
a.grid(alpha=.25)
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False)
fig.savefig("results/speed.png", dpi=170)
plt.close(fig)

det_model = YOLO("yolo11n.pt")
fig, ax = plt.subplots(1, 3, figsize=(12, 5.4), constrained_layout=True)
det = {}
for a, s in zip(ax, (160, 320, 640)):
    r = det_model.predict(str(ASSETS / "bus.jpg"), imgsz=s, conf=.4, verbose=False)[0]
    det[s] = len(r.boxes)
    a.imshow(cv2.cvtColor(r.plot(line_width=3), cv2.COLOR_BGR2RGB))
    a.set_title(f"Input {s} px: {len(r.boxes)} objects")
    a.axis("off")
fig.savefig("results/detections.png", dpi=150)
json.dump(det, open("results/detections.json", "w"))
print(det)

import numpy as np

COL = {"yolo11n": "#1a73e8", "yolov8n": "#e8710a"}
get = lambda m, s, t: next(r for r in rows if r["model"] == m and r["size"] == s and r["threads"] == t)

# Time per frame
fig, a = plt.subplots(figsize=(7.5, 4.1), constrained_layout=True)
x = np.arange(len(sizes))
for i, m in enumerate(COL):
    v = [get(m, s, 2)["ms"] for s in sizes]
    bars = a.bar(x + (i - .5) * .38, v, .38, color=COL[m], label=m)
    a.bar_label(bars, fmt="%.1f", fontsize=8)
a.set_xticks(x, sizes)
a.set(xlabel="Input size (pixels)", ylabel="Milliseconds per frame", title="Time per frame, 2 threads (x86 cloud CPU)")
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False)
fig.savefig("results/latency.png", dpi=170)
plt.close(fig)

# Gain from a second thread
fig, a = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
for m in COL:
    a.plot(sizes, [get(m, s, 1)["ms"] / get(m, s, 2)["ms"] for s in sizes], "o-", c=COL[m], lw=2, label=m)
a.axhline(2, c="k", ls="--", label="Perfect doubling")
a.set(xlabel="Input size (pixels)", ylabel="Speed-up with 2 threads", title="How much a second CPU thread helps", ylim=(1, 2.1))
a.set_xticks(sizes)
a.grid(alpha=.25)
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False)
fig.savefig("results/thread_speedup.png", dpi=170)
plt.close(fig)

# Confidence threshold
confs = [.1, .2, .3, .4, .5, .6, .7, .8]
fig, a = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
sweep = {}
for s, c in zip((160, 320, 640), ("#f4b400", "#12b5cb", "#d01884")):
    sweep[s] = [len(det_model.predict(str(ASSETS / "bus.jpg"), imgsz=s, conf=k, verbose=False)[0].boxes) for k in confs]
    a.plot(confs, sweep[s], "o-", c=c, lw=2, label=f"Input {s} px")
a.set(xlabel="Confidence threshold", ylabel="Objects kept", title="Stricter thresholds keep fewer objects (one sample image)")
a.grid(alpha=.25)
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False)
fig.savefig("results/confidence_sweep.png", dpi=170)
plt.close(fig)

# Confidence of each object by size
fig, a = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
best = {}
for i, s in enumerate((160, 320, 640)):
    r = det_model.predict(str(ASSETS / "bus.jpg"), imgsz=s, conf=.25, verbose=False)[0]
    v = sorted((float(x) for x in r.boxes.conf), reverse=True)
    best[s] = v
    a.scatter([i] * len(v), v, s=70, c="#1a73e8", alpha=.75)
a.set_xticks(range(3), ["160 px", "320 px", "640 px"])
a.set(ylabel="Confidence", ylim=(0, 1), title="Larger input raises confidence on the same image")
a.grid(alpha=.25)
a.spines[["top", "right"]].set_visible(False)
fig.savefig("results/confidence_by_size.png", dpi=170)
plt.close(fig)
json.dump(dict(objects_at_conf_04=det, conf_sweep=sweep, confidence_by_size=best), open("results/detections.json", "w"), indent=2)
print("sweep", sweep, "\nbest", {k: [round(x, 2) for x in v] for k, v in best.items()})
print("speedup", {m_: [round(get(m_, s, 1)["ms"] / get(m_, s, 2)["ms"], 2) for s in sizes] for m_ in COL})
