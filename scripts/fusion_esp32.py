import cv2
import time
import os
import torch
from ultralytics import YOLO
from threading import Thread, Lock

# =========================
# CONFIG
# =========================
ESP32_STREAM_URL = "http://192.168.4.1:81/stream"
YOLO_SIZE = 320
DETECT_EVERY_N_FRAMES = 3
PERSON_CONF = 0.35

# YOLO DEFAULT STYLE (DO NOT TOUCH)
BOX_COLOR = (255, 0, 0)  # Blue (BGR)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
TEXT_THICKNESS = 2

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;60000"

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

    # =====================================
    # CHECK GPU
    # =====================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Using device: {DEVICE}")

    print("⏳ Loading models...")

    coco_model = YOLO("yolov8n.pt")
    fire_model = YOLO(str(fire_model_path))

    # ---- MOVE TO GPU ----
    coco_model.to(DEVICE)
    fire_model.to(DEVICE)

    print("✅ Models loaded on", DEVICE)

    import os
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8n.pt")
    model = YOLO(model_path).to("cuda")

    cam = Camera(ESP32_STREAM_URL)
    time.sleep(2)

    # GPU warm-up (IMPORTANT)
    dummy = torch.zeros((1, 3, YOLO_SIZE, YOLO_SIZE), device="cuda")
    model.predict(dummy, verbose=False)

    frame_count = 0
    last_boxes = None

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
            results = model.predict(
                yolo_frame,
                imgsz=YOLO_SIZE,
                conf=PERSON_CONF,
                classes=[0],
                device="cuda",
                verbose=False
            )[0]
            last_boxes = results.boxes

        # =========================
        # DRAW (NATURAL LOOK)
        # =========================
        output = frame.copy()

        # Resize for YOLO
        yolo_frame = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))

        # =========================
        # GPU INFERENCE
        # =========================
        r_coco = coco_model.predict(
            yolo_frame,
            conf=COCO_CONF,
            verbose=False,
            device=DEVICE
        )[0]

        r_fire = fire_model.predict(
            yolo_frame,
            conf=FIRE_CONF,
            verbose=False,
            device=DEVICE
        )[0]

        now = time.time()

        if r_coco.boxes is not None and len(r_coco.boxes) > 0:
            last_coco_seen = now

        if r_fire.boxes is not None and len(r_fire.boxes) > 0:
            last_fire_seen = now

        coco_present = (now - last_coco_seen) < PERSIST_TIME
        fire_present = (now - last_fire_seen) < PERSIST_TIME

        sx = w / YOLO_SIZE
        sy = h / YOLO_SIZE

        if last_boxes is not None and len(last_boxes) > 0:
            for box in last_boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                x1 = int(x1 * sx)
                x2 = int(x2 * sx)
                y1 = int(y1 * sy)
                y2 = int(y2 * sy)

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    BOX_COLOR,
                    BOX_THICKNESS
                )

                cv2.putText(
                    output,
                    f"person {conf:.2f}",
                    (x1, y1 - 6),
                    FONT,
                    FONT_SCALE,
                    BOX_COLOR,
                    TEXT_THICKNESS,
                    cv2.LINE_AA
                )

        # =========================
        # FPS (REAL)
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

        cv2.imshow("ESP32 – Fire + COCO Detection [GPU]", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
