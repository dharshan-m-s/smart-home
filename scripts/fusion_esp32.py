import cv2
import time
import numpy as np
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

# DISPLAY SCALE (THIS CONTROLS VISUAL SIZE)
DISPLAY_SCALE = 2.0   # 1.0 = original, 2.0 = big & clear

# YOLO STYLE (SCALED LATER)
BASE_BOX_THICKNESS = 2
BASE_FONT_SCALE = 0.6
BASE_TEXT_THICKNESS = 2
BOX_COLOR = (255, 0, 0)  # YOLO blue

FONT = cv2.FONT_HERSHEY_SIMPLEX

# =========================
# CAMERA THREAD
# =========================
class LatestFrame:
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
# REAL CLARITY ENHANCEMENT
# =========================
def enhance_clarity(img):
    # Contrast (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Unsharp mask
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = cv2.addWeighted(img, 1.35, blur, -0.35, 0)

    return img

# =========================
# DRAW BIG, CLEAR LABEL
# =========================
def draw_label(img, text, x, y, font_scale, thickness):
    (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - 10), (x + tw + 8, y), BOX_COLOR, -1)
    cv2.putText(
        img,
        text,
        (x + 4, y - 4),
        FONT,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

# =========================
# MAIN
# =========================
def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt").to("cuda")

    cam = LatestFrame(ESP32_STREAM_URL)
    time.sleep(2)

    # GPU warm-up
    dummy = np.zeros((YOLO_SIZE, YOLO_SIZE, 3), dtype=np.uint8)
    model.predict(dummy, device="cuda", verbose=False)

    frame_id = 0
    last_boxes = []

    while True:
        start = time.time()
        frame = cam.read()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        frame_id += 1

        # =========================
        # YOLO INFERENCE
        # =========================
        if frame_id % DETECT_EVERY_N_FRAMES == 0:
            yolo_frame = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
            results = model.predict(
                yolo_frame,
                device="cuda",
                conf=PERSON_CONF,
                classes=[0],
                imgsz=YOLO_SIZE,
                verbose=False
            )[0]
            last_boxes = results.boxes

        # =========================
        # PROCESS FRAME
        # =========================
        output = enhance_clarity(frame.copy())

        # Scale factors
        sx = w / YOLO_SIZE
        sy = h / YOLO_SIZE

        # Scaled drawing params
        box_thickness = int(BASE_BOX_THICKNESS * DISPLAY_SCALE)
        font_scale = BASE_FONT_SCALE * DISPLAY_SCALE
        text_thickness = int(BASE_TEXT_THICKNESS * DISPLAY_SCALE)

        if last_boxes is not None:
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
                    box_thickness
                )

                draw_label(
                    output,
                    f"person {conf:.2f}",
                    x1,
                    y1,
                    font_scale,
                    text_thickness
                )

        fps = 1.0 / (time.time() - start + 1e-6)
        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (20, 40),
            FONT,
            font_scale,
            (0, 255, 0),
            text_thickness,
            cv2.LINE_AA
        )

        # =========================
        # DISPLAY (UPSCALE PROPERLY)
        # =========================
        display = cv2.resize(
            output,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_CUBIC
        )

        cv2.imshow("ESP32 – BIG BOX, CLEAR TEXT", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()