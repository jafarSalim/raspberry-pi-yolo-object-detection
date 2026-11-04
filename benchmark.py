"""Measures detection speed by model, input size and CPU threads.

Run it on your own Raspberry Pi to get real numbers for that board.
"""
import argparse, json, pathlib, platform, statistics, time
import cv2, numpy as np, onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils import ASSETS

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["yolo11n", "yolov8n"])
ap.add_argument("--sizes", nargs="+", type=int, default=[160, 256, 320, 416, 640])
ap.add_argument("--threads", nargs="+", type=int, default=[1, 2])
ap.add_argument("--repeats", type=int, default=30)
a = ap.parse_args()

OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)
img = cv2.imread(str(ASSETS / "bus.jpg"))


def tensor(size):
    x = cv2.resize(img, (size, size))[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return x


rows = []
for name in a.models:
    model = YOLO(name + ".pt")
    for size in a.sizes:
        path = model.export(format="onnx", imgsz=size, simplify=False, verbose=False, opset=17)
        target = pathlib.Path(f"{name}_{size}.onnx")
        pathlib.Path(path).replace(target)
        for th in a.threads:
            o = ort.SessionOptions()
            o.intra_op_num_threads = th
            o.inter_op_num_threads = 1
            s = ort.InferenceSession(str(target), o, providers=["CPUExecutionProvider"])
            x = tensor(size)
            inp = s.get_inputs()[0].name
            for _ in range(3):
                s.run(None, {inp: x})
            t = []
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                s.run(None, {inp: x})
                t.append(1000 * (time.perf_counter() - t0))
            ms = statistics.median(t)
            rows.append(dict(model=name, size=size, threads=th, ms=ms, fps=1000 / ms, mb=target.stat().st_size / 1e6))
            print(rows[-1], flush=True)
        target.unlink()
json.dump(dict(machine=platform.processor() or platform.machine(), system=platform.platform(), rows=rows),
          open(OUT / "benchmark.json", "w"), indent=2)
