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
SERIAL_PORT = "COM3"
SERIAL_BAUD = 9600

MODEL_PATH = "yolov8m.pt"   # Change to yolov8n.pt if FPS is low
CONF_THRES = 0.4

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

            for item in line.split(","):
                if item.startswith("PIR="):
                    with sensor_lock:
                        pir_state = int(item.split("=")[1])
        except:
            pass


# =========================
# MAIN
# =========================
def main():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    Thread(target=serial_reader, daemon=True).start()

    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    if not cap.isOpened():
        print("❌ Failed to open camera stream")
        return

    person_detected = False

    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        # 🔥 TRACK instead of predict
        results = model.track(
            frame,
            conf=CONF_THRES,
            device=DEVICE,
            persist=True,
            verbose=False
        )[0]

        output = frame.copy()
        person_detected = False

        if results.boxes is not None:
            for box in results.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                label = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "person":
                    person_detected = True
                    color = (0, 0, 255)
                else:
                    random.seed(track_id)
                    color = (
                        random.randint(0,255),
                        random.randint(0,255),
                        random.randint(0,255)
                    )

                cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)
                cv2.putText(output,
                            f"{label} ID:{track_id} {conf:.2f}",
                            (x1,y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color, 2)

        # =========================
        # SENSOR FUSION
        # =========================
        with sensor_lock:
            pir = pir_state

        if person_detected and pir == 1:
            cv2.putText(output,
                        "🚨 INTRUSION DETECTED",
                        (40,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0,0,255),3)

        # FPS
        fps = 1.0 / (time.time() - start + 1e-6)
        cv2.putText(output,
                    f"FPS: {fps:.1f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),2)

        cv2.imshow("Fusion System", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()