from ultralytics import YOLO
from pathlib import Path
import os

def main():
    ai_root = Path(__file__).resolve().parents[1]  # smart-home/ai
    data_yaml = ai_root / "dataset_fire_smoke" / "data.yaml"
    runs_dir = ai_root / "runs"
    models_dir = ai_root / "models"

    runs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # If you are on CPU locally, set:
    # device = "cpu"
    # batch = 8
    # imgsz = 512
    #
    # If you are on Colab GPU, keep:
    device = 0
    batch = 16
    imgsz = 640

    model = YOLO("yolov8n.pt")  # pretrained base

    model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(runs_dir),
        name="fire_smoke_train",
        pretrained=True,
        patience=10
    )

    # Save best.pt to ai/models/
    best_path = runs_dir / "fire_smoke_train" / "weights" / "best.pt"
    if best_path.exists():
        out_path = models_dir / "fire_smoke_best.pt"
        out_path.write_bytes(best_path.read_bytes())
        print(f"✅ Saved model: {out_path}")
    else:
        print("❌ best.pt not found. Check training output folder.")

if __name__ == "__main__":
    main()
