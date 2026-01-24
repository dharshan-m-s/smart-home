import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    model_path = Path(__file__).resolve().parents[1] / "models" / "fire_smoke_best.pt"
    if not model_path.exists():
        print("❌ fire_smoke_best.pt not found in ai/models/")
        return

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not found")
        return

    print("✅ Fire+Smoke model running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.predict(frame, conf=0.4, verbose=False)[0]
        out = res.plot()

        cv2.imshow("Fire+Smoke Detection", out)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
