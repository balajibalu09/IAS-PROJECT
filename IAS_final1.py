import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time
from openpyxl import Workbook
from datetime import datetime

# ==============================
# CONFIG
# ==============================
FOCAL_LENGTH = 800

DANGER_DISTANCE = 50
WARNING_DISTANCE = 100
CONF_THRESHOLD = 0.3
ALERT_INTERVAL = 2

KNOWN_WIDTHS = {
    "person": 50,
    "bottle": 7,
    "cup": 8,
    "chair": 40,
    "laptop": 35,
    "car": 180,
    "dog": 60,
    "cell phone": 7,
    "book": 20
}

# ==============================
# INIT
# ==============================
model = YOLO("yolov8n.pt")

engine = pyttsx3.init()
engine.setProperty('rate', 170)

wb = Workbook()
ws = wb.active
ws.title = "Distance Data"

# ✅ Updated Excel header
ws.append(["Time", "Object", "Distance (cm)", "Status", "Direction"])

graph_data = []
last_voice_time = 0

# ==============================
# FUNCTIONS
# ==============================
def calculate_distance(W, f, p):
    return (W * f) / p if p != 0 else 0

def get_direction(x, width):
    if x < width/3:
        return "LEFT"
    elif x < 2*width/3:
        return "CENTER"
    else:
        return "RIGHT"

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera error")
    exit()

print("🚀 SYSTEM RUNNING")

# ==============================
# LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.convertScaleAbs(frame, alpha=0.9, beta=-20)

    h, w, _ = frame.shape
    dashboard = np.zeros((h, 300, 3), dtype=np.uint8)
    radar = np.zeros((h, 300, 3), dtype=np.uint8)

    results = model(frame)

    danger_objects = []
    object_count = 0
    closest_distance = float('inf')

    cx_radar, cy_radar = 150, h-50

    # Radar circles
    for r in range(50, 250, 50):
        cv2.circle(radar, (cx_radar, cy_radar), r, (50,50,50), 1)

    # =========================
    # DETECTION
    # =========================
    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):

            if confs[i] < CONF_THRESHOLD:
                continue

            x1,y1,x2,y2 = map(int, box)
            label = model.names[int(classes[i])]

            x_center = (x1+x2)//2
            direction = get_direction(x_center, w)

            # Distance
            if label in KNOWN_WIDTHS:
                pixel_w = x2 - x1
                distance = calculate_distance(KNOWN_WIDTHS[label], FOCAL_LENGTH, pixel_w)
            else:
                distance = 150

            distance = round(distance,1)
            object_count += 1

            # Status
            if distance < DANGER_DISTANCE:
                color = (0,0,255)
                status = "DANGER"
                danger_objects.append(f"{label} on {direction}")
            elif distance < WARNING_DISTANCE:
                color = (0,255,255)
                status = "WARNING"
            else:
                color = (0,255,0)
                status = "SAFE"

            # Draw
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,
                        f"{label} {distance}cm [{direction}]",
                        (x1,max(y1-10,20)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            # Radar
            if direction == "LEFT":
                rx = cx_radar - 80
            elif direction == "RIGHT":
                rx = cx_radar + 80
            else:
                rx = cx_radar

            ry = int(cy_radar - min(distance,200)*2)
            cv2.circle(radar,(rx,ry),8,color,-1)

            # ✅ Excel logging with status + direction
            ws.append([
                datetime.now().strftime("%H:%M:%S"),
                label,
                distance,
                status,
                direction
            ])

            # Track closest
            if distance < closest_distance:
                closest_distance = distance

    # =========================
    # VOICE ALERT
    # =========================
    if len(danger_objects) > 0:
        if time.time() - last_voice_time > ALERT_INTERVAL:
            alert_text = "Warning " + ", ".join(danger_objects)
            engine.stop()
            engine.say(alert_text)
            engine.runAndWait()
            last_voice_time = time.time()

    # =========================
    # GRAPH
    # =========================
    if closest_distance != float('inf'):
        graph_data.append(closest_distance)
        graph_data = graph_data[-50:]

    sx, sy = 20, 250
    gw, gh = 250, 100

    cv2.rectangle(dashboard,(sx,sy),(sx+gw,sy+gh),(255,255,255),1)

    if len(graph_data) > 1:
        max_val = max(graph_data)
        for i in range(1,len(graph_data)):
            x1 = sx + int((i-1)*gw/50)
            y1 = sy + gh - int(graph_data[i-1]/max_val*gh)

            x2 = sx + int(i*gw/50)
            y2 = sy + gh - int(graph_data[i]/max_val*gh)

            cv2.line(dashboard,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.putText(dashboard,"Distance Graph",(sx,sy-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    # =========================
    # DASHBOARD TEXT
    # =========================
    cv2.putText(dashboard,
                "Monocular Vision-Based Distance Inference",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    cv2.putText(dashboard,f"Objects: {object_count}",(30,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    status_main = "DANGER" if len(danger_objects)>0 else "SAFE"
    color_main = (0,0,255) if status_main=="DANGER" else (0,255,0)

    cv2.putText(dashboard,f"STATUS: {status_main}",(30,150),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,color_main,2)

    # =========================
    # DISPLAY
    # =========================
    combined = np.hstack((frame, dashboard, radar))
    cv2.imshow("FINAL SYSTEM", combined)

    if cv2.waitKey(1) == 27:
        break

# ==============================
# SAVE EXCEL
# ==============================
cap.release()
cv2.destroyAllWindows()

wb.save("distance_data.xlsx")
print("✅ Excel saved")
