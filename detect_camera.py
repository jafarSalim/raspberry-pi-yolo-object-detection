"""Live object detection for a Raspberry Pi camera or any USB webcam.

Counts objects by class, and stays small enough to run with no display.
Not yet tested on a Raspberry Pi. See the README.
"""
import argparse, collections, time
import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--size", type=int, default=256, help="Smaller is faster, larger finds small objects")
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--source", default="0", help="Camera index, video file or RTSP address")
ap.add_argument("--show", action="store_true", help="Open a preview window")
ap.add_argument("--picamera", action="store_true", help="Use the Raspberry Pi camera via picamera2")
a = ap.parse_args()

model = YOLO(a.model)


def frames():
    if a.picamera:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
        cam.start()
        while True:
            yield cam.capture_array()
    else:
        cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f


last = time.time()
for n, frame in enumerate(frames(), 1):
    r = model.predict(frame, imgsz=a.size, conf=a.conf, verbose=False)[0]
    count = collections.Counter(model.names[int(c)] for c in r.boxes.cls)
    now = time.time()
    print(f"{1 / (now - last):5.1f} FPS  {dict(count)}", flush=True)
    last = now
    if a.show:
        cv2.imshow("detections", r.plot())
        if cv2.waitKey(1) == 27:
            break
