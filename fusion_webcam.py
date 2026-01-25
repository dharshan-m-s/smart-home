import cv2
from ultralytics import YOLO
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    fire_model_path = project_root / "models" / "fire_smoke_best.pt"

    if not fire_model_path.exists():
        print("❌ Fire model not found:", fire_model_path)
        return

    # Model 1: COCO pretrained (person/pets/objects)
    coco_model = YOLO("yolov8n.pt")

    # Model 2: Custom fire+smoke model
    fire_model = YOLO(str(fire_model_path))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not found")
        return

    print("✅ Fusion running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            r1 = coco_model.predict(frame, conf=0.4, verbose=False)[0]
            r2 = fire_model.predict(frame, conf=0.4, verbose=False)[0]

            out = r1.plot()
            out = r2.plot(img=out)

            cv2.imshow("Smart Security AI (Fusion)", out)
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
