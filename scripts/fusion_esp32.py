import cv2
from ultralytics import YOLO
from pathlib import Path
import time

ESP32_URL = "http://192.168.1.10:81/stream"  # CHANGE THIS

def main():
    root = Path(__file__).resolve().parents[1]

    fire_path = root / "models" / "fire_smoke_best.pt"
    obj_path  = root / "models" / "home_objects_v1.pt"

    if not fire_path.exists():
        print("❌ Missing fire model:", fire_path)
        return
    if not obj_path.exists():
        print("❌ Missing object model:", obj_path)
        return

    print("⏳ Loading models...")
    coco_person_model = YOLO("yolov8n.pt")      # COCO for PERSON only
    object_model = YOLO(str(obj_path))          # indoor objects
    fire_model = YOLO(str(fire_path))           # fire + smoke
    print("✅ Models loaded")

    cap = cv2.VideoCapture(ESP32_URL)
    if not cap.isOpened():
        print("❌ Cannot open ESP32 stream:", ESP32_URL)
        return

    print("✅ Running ESP32 fusion. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        start = time.time()

        # 1) Person only from COCO
        r_person = coco_person_model.predict(frame, conf=0.35, classes=[0], verbose=False)[0]
        out = r_person.plot()

        # 2) Objects from your indoor model
        r_obj = object_model.predict(frame, conf=0.35, verbose=False)[0]
        out = r_obj.plot(img=out)

        # 3) Fire/Smoke from your custom model
        r_fire = fire_model.predict(frame, conf=0.45, verbose=False)[0]
        out = r_fire.plot(img=out)

        fps = 1.0 / (time.time() - start + 1e-6)
        cv2.putText(out, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ESP32 Fusion: Person + Objects + Fire/Smoke", out)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
