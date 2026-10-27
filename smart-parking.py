# SMART PARKING SYSTEM - SINGLE FLOOR VIDEO
# YOLO + OpenCV
# Input: 103979.mp4

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# 1. LOAD YOLO MODEL
# --------------------------------------------------
model = YOLO("yolo11n.pt")       # You can use your trained model here

VIDEO_PATH = "103979.mp4"
OUTPUT_PATH = "smart_parking_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# --------------------------------------------------
# 2. PARKING SPACE COORDINATES
# --------------------------------------------------
# Coordinates are for this 1280x720 video.
# Format:
# (x1, y1, x2, y2)

parking_spaces = {
    "B-P01": (20, 305, 345, 510),
    "B-P02": (310, 350, 665, 610),
    "B-P03": (620, 350, 915, 615),
    "B-P04": (885, 370, 1200, 610),
}

# --------------------------------------------------
# 3. SETTINGS
# --------------------------------------------------

CAR_CLASS = 2       # COCO class ID for car
CONFIDENCE = 0.35

# Percentage of parking-space area that must
# overlap with a detected car
OCCUPANCY_THRESHOLD = 0.15


# --------------------------------------------------
# 4. CHECK PARKING OCCUPANCY
# --------------------------------------------------

def calculate_overlap(car_box, parking_box):

    cx1, cy1, cx2, cy2 = car_box
    px1, py1, px2, py2 = parking_box

    ix1 = max(cx1, px1)
    iy1 = max(cy1, py1)
    ix2 = min(cx2, px2)
    iy2 = min(cy2, py2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0

    intersection = (ix2 - ix1) * (iy2 - iy1)

    parking_area = (px2 - px1) * (py2 - py1)

    return intersection / parking_area


# --------------------------------------------------
# 5. PROCESS VIDEO
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    car_boxes = []

    # --------------------------------------------------
    # 6. GET ONLY CAR DETECTIONS
    # --------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # IMPORTANT:
            # Only process CAR.
            # Arrows, signs, markings etc. are ignored.
            if cls != CAR_CLASS:
                continue

            if conf < CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            car_boxes.append(
                (x1, y1, x2, y2)
            )

            # Draw car detection
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"CAR {conf:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    # --------------------------------------------------
    # 7. CHECK EACH PARKING SPACE
    # --------------------------------------------------

    occupied_count = 0

    for space_id, parking_box in parking_spaces.items():

        occupied = False

        for car_box in car_boxes:

            overlap = calculate_overlap(
                car_box,
                parking_box
            )

            if overlap >= OCCUPANCY_THRESHOLD:

                occupied = True
                break

        x1, y1, x2, y2 = parking_box

        if occupied:

            occupied_count += 1

            # Occupied parking space
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            status = "OCCUPIED"

        else:

            # Available parking space
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            status = "AVAILABLE"

        # Parking ID
        cv2.putText(
            frame,
            space_id,
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        # Status
        cv2.putText(
            frame,
            status,
            (x1 + 10, y1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # --------------------------------------------------
    # 8. PARKING STATISTICS
    # --------------------------------------------------

    total_spaces = len(parking_spaces)
    available_spaces = total_spaces - occupied_count

    occupancy_percentage = (
        occupied_count / total_spaces * 100
        if total_spaces > 0
        else 0
    )

    # --------------------------------------------------
    # 9. INFORMATION PANEL
    # --------------------------------------------------

    cv2.rectangle(
        frame,
        (20, 20),
        (390, 145),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "SMART PARKING - FLOOR B",
        (35, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Spaces : {total_spaces}",
        (35, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Occupied     : {occupied_count}",
        (35, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Available    : {available_spaces}",
        (35, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------
    # 10. SHOW OUTPUT
    # --------------------------------------------------

    cv2.imshow(
        "Smart Mall Parking System",
        frame
    )

    out.write(frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 11. RELEASE
# --------------------------------------------------

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed.")
print("Output saved as:", OUTPUT_PATH)