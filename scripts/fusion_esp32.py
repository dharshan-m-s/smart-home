import cv2
from ultralytics import YOLO
from pathlib import Path
import time

ESP32_URL = "http://192.168.1.10:81/stream"  # CHANGE THIS

def main():
    root = Path(__file__).resolve().parents[1]

    fire_model_path = root / "models" / "fire_smoke_best.pt"
    if not fire_model_path.exists():
        print("❌ Fire model not found:", fire_model_path)
        return

    print("⏳ Loading models...")
    person_model = YOLO("yolov8n.pt")       # COCO → person
    fire_model = YOLO(str(fire_model_path)) # Fire + Smoke
    print("✅ Models loaded")

    cap = cv2.VideoCapture(ESP32_URL)
    if not cap.isOpened():
        print("❌ Cannot open ESP32 stream:", ESP32_URL)
        return

    print("✅ ESP32 fusion running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        start = time.time()

        # PERSON only
        r_person = person_model.predict(
            frame,
            conf=0.35,
            classes=[0],
            verbose=False
        )[0]

        out = r_person.plot()

        # FIRE + SMOKE
        r_fire = fire_model.predict(
            frame,
            conf=0.45,
            verbose=False
        )[0]

        out = r_fire.plot(img=out)

        fps = 1.0 / (time.time() - start + 1e-6)
        cv2.putText(
            out, f"FPS: {fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow("ESP32 – Person + Fire/Smoke", out)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
