import cv2
import time
import os
import torch
import random
import serial
from ultralytics import YOLO
from threading import Thread, Lock

# =========================
# CONFIG
# =========================
ESP32_STREAM_URL = "http://192.168.4.1:81/stream"
SERIAL_PORT = "COM4"      # CHANGE
SERIAL_BAUD = 9600

YOLO_SIZE = 320
DETECT_EVERY_N_FRAMES = 3
COCO_CONF = 0.35

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;60000"

# =========================
# GLOBAL SENSOR STATE
# =========================
pir_state = 0
sensor_lock = Lock()

# =========================
# SERIAL THREAD
# =========================
def serial_reader():
    global pir_state

    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        print("✅ Arduino connected")
    except Exception as e:
        print("❌ Serial error:", e)
        return

    while True:
        try:
            line = ser.readline().decode().strip()
            if not line:
                continue

            # Example: PIR=1,GAS=420,...
            for item in line.split(","):
                if item.startswith("PIR="):
                    with sensor_lock:
                        pir_state = int(item.split("=")[1])

        except Exception:
            pass


# =========================
# CAMERA THREAD
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

    # ---- Start serial thread ----
    Thread(target=serial_reader, daemon=True).start()

    # ---- Load model ----
    model = YOLO("yolov8n.pt")
    model.to(DEVICE)
    CLASS_NAMES = model.names

    cam = Camera(ESP32_STREAM_URL)
    time.sleep(2)

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
        # YOLO
        # =========================
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            yolo_frame = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
            results = model.predict(
                yolo_frame,
                imgsz=YOLO_SIZE,
                conf=COCO_CONF,
                device=DEVICE,
                classes=[0],      # PERSON ONLY
                verbose=False
            )[0]
            last_boxes = results.boxes

        output = frame.copy()
        sx = w / YOLO_SIZE
        sy = h / YOLO_SIZE

        person_detected = False

        if last_boxes is not None and len(last_boxes) > 0:
            person_detected = True
            for box in last_boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                x1, x2 = int(x1*sx), int(x2*sx)
                y1, y2 = int(y1*sy), int(y2*sy)

                cv2.rectangle(output, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(output, f"Person {conf:.2f}",
                            (x1,y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(255,0,0),2)

        # =========================
        # FUSION
        # =========================
        with sensor_lock:
            pir = pir_state

        if person_detected and pir == 1:
            cv2.putText(output, "🚨 INTRUSION DETECTED",
                        (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,(0,0,255),3)

        # =========================
        # FPS
        # =========================
        fps = 1.0 / (time.time() - start + 1e-6)
        cv2.putText(output, f"FPS: {fps:.1f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.imshow("Fusion: Camera + PIR", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()