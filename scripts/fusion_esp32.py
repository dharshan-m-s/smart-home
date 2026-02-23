import cv2
import time
import os
import torch
import random
from ultralytics import YOLO
from threading import Thread, Lock

# =========================
# CONFIG
# =========================
ESP32_STREAM_URL = "http://192.168.4.1:81/stream"

YOLO_SIZE = 320
DETECT_EVERY_N_FRAMES = 3
COCO_CONF = 0.35

BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
TEXT_THICKNESS = 2

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;60000"

# =========================
# CAMERA THREAD (LATEST FRAME ONLY)
# =========================
class Camera:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = Lock()
        self.running = True
        Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()

# =========================
# MAIN
# =========================
def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    DEVICE = "cuda"
    print(f"🔥 Using device: {DEVICE}")

    # =========================
    # LOAD MODELS
    # =========================
    print("⏳ Loading YOLOv8 COCO model...")

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "yolov8n.pt"
    )

    coco_model = YOLO(model_path)
    coco_model.to(DEVICE)

    CLASS_NAMES = coco_model.names

    # =========================
    # GENERATE COLORS (ONE PER CLASS)
    # =========================
    random.seed(42)
    CLASS_COLORS = {
        cls_id: (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
        for cls_id in CLASS_NAMES
    }

    print("✅ Model loaded")

    # =========================
    # CAMERA
    # =========================
    cam = Camera(ESP32_STREAM_URL)
    time.sleep(2)

    # =========================
    # GPU WARM-UP
    # =========================
    dummy = torch.zeros((1, 3, YOLO_SIZE, YOLO_SIZE), device="cuda")
    coco_model.predict(dummy, verbose=False)

    frame_count = 0
    last_boxes = None

    # =========================
    # LOOP
    # =========================
    while True:
        start = time.time()

        frame = cam.read()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        frame_count += 1

        # =========================
        # YOLO INFERENCE
        # =========================
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            yolo_frame = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
            results = coco_model.predict(
                yolo_frame,
                imgsz=YOLO_SIZE,
                conf=COCO_CONF,
                device=DEVICE,
                verbose=False
            )[0]
            last_boxes = results.boxes

        # =========================
        # DRAW RESULTS
        # =========================
        output = frame.copy()
        sx = w / YOLO_SIZE
        sy = h / YOLO_SIZE

        if last_boxes is not None and len(last_boxes) > 0:
            for box in last_boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = CLASS_NAMES[cls_id]
                color = CLASS_COLORS[cls_id]

                x1 = int(x1 * sx)
                x2 = int(x2 * sx)
                y1 = int(y1 * sy)
                y2 = int(y2 * sy)

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    color,
                    BOX_THICKNESS
                )

                cv2.putText(
                    output,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 6)),
                    FONT,
                    FONT_SCALE,
                    color,
                    TEXT_THICKNESS,
                    cv2.LINE_AA
                )

        # =========================
        # FPS
        # =========================
        fps = 1.0 / (time.time() - start + 1e-6)
        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (20, 40),
            FONT,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("ESP32 – YOLOv8 COCO (ALL CLASSES)", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
