import cv2
import requests
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
from datetime import datetime

def main():
    # ESP32 camera stream URL (default ESP32-CAM IP)
    ESP32_IP = "192.168.1.100"  # Change to your ESP32 IP
    ESP32_PORT = 81
    STREAM_URL = f"http://{ESP32_IP}:{ESP32_PORT}/stream"
    
    project_root = Path(__file__).resolve().parent
    fire_model_path = project_root / "models" / "fire_smoke_v2.pt"

    if not fire_model_path.exists():
        print("❌ Fire model not found:", fire_model_path)
        return

    # Model 1: COCO pretrained (person/pets/objects)
    coco_model = YOLO("yolov8n.pt")

    # Model 2: Custom fire+smoke model
    fire_model = YOLO(str(fire_model_path))

    print(f"🔌 Connecting to ESP32 at {STREAM_URL}...")
    
    try:
        response = requests.get(STREAM_URL, stream=True, timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to connect to ESP32. Status: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ ESP32 connection error: {e}")
        print(f"   Make sure ESP32 is running at {ESP32_IP}")
        return

    print("✅ Fusion ESP32 running. Press Q to quit.")
    
    bytes_data = b""
    frame_count = 0
    fire_alerts = []

    for chunk in response.iter_content(chunk_size=1024):
        bytes_data += chunk
        
        # Extract JPEG frame from MJPEG stream
        a = bytes_data.find(b'\xff\xd8')  # JPEG start marker
        b = bytes_data.find(b'\xff\xd9')  # JPEG end marker
        
        if a != -1 and b != -1:
            jpg_data = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            
            # Decode frame
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            frame_count += 1
            
            try:
                # Run detections
                r1 = coco_model.predict(frame, conf=0.4, verbose=False)[0]
                r2 = fire_model.predict(frame, conf=0.4, verbose=False)[0]
                
                # Check for fire/smoke detections
                if len(r2.boxes) > 0:
                    alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    fire_alerts.append(alert_time)
                    print(f"🔥 FIRE/SMOKE DETECTED at {alert_time}")
                
                # Overlay both detections
                out = r1.plot()
                out = r2.plot(img=out)
                
                # Add frame info
                cv2.putText(out, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(out, f"Alerts: {len(fire_alerts)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if len(fire_alerts) > 0 else (0, 255, 0), 2)
                
                cv2.imshow("Smart Security AI (ESP32 Fusion)", out)
                
            except Exception as e:
                print(f"⚠️  Prediction error: {e}")
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cv2.destroyAllWindows()
    
    print(f"\n📊 Session Summary:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Fire/Smoke alerts: {len(fire_alerts)}")
    if fire_alerts:
        print(f"   Alert times: {', '.join(fire_alerts)}")

if __name__ == "__main__":
    main()