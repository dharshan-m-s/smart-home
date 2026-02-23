import serial
import csv
import time
from datetime import datetime
import os

# ==========================
# CONFIG
# ==========================
PORT = "COM4"      # CHANGE
BAUD = 9600
SAVE_INTERVAL = 1   # seconds (1 or 2)
CSV_FILE = "sensor_dataset.csv"
# ==========================


# ---- Open serial ----
print("Connecting to Arduino...")
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
print("✅ Connected")

# ---- CSV header ----
header = ["timestamp", "PIR", "GAS", "WATER", "TEMP", "HUM", "ALERT"]

# Create file with header if not exists
file_exists = os.path.exists(CSV_FILE)

csv_file = open(CSV_FILE, "a", newline="")
writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow(header)
    print("📄 Created new dataset file")

last_save = 0

while True:
    try:
        line = ser.readline().decode().strip()
        if not line:
            continue

        # Example:
        # PIR=1,GAS=420,WATER=120,TEMP=28.3,HUM=62.1,ALERT=1

        data = {}
        for item in line.split(","):
            if "=" in item:
                k, v = item.split("=")
                data[k] = v

        now = time.time()

        # Save only every interval
        if now - last_save >= SAVE_INTERVAL:
            timestamp = datetime.now().isoformat()

            row = [
                timestamp,
                data.get("PIR", ""),
                data.get("GAS", ""),
                data.get("WATER", ""),
                data.get("TEMP", ""),
                data.get("HUM", ""),
                data.get("ALERT", "")
            ]

            writer.writerow(row)
            csv_file.flush()   # ensure immediate write

            print("Saved:", row)

            last_save = now

    except KeyboardInterrupt:
        print("\nStopping logger")
        break

    except Exception as e:
        print("Error:", e)

csv_file.close()