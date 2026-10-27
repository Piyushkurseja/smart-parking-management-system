import cv2
import json
import os

from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "smartparking.mp4"

MODEL_PATH = "yolo11n.pt"

PARKING_FILE = "config/parking_spaces.json"

OUTPUT_VIDEO = "outputs/smart_parking_result.mp4"


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)


# ==========================================
# LOAD PARKING SPACES
# ==========================================

if not os.path.exists(PARKING_FILE):

    print("ERROR: Parking spaces not found.")

    print(
        "Run setup_spaces.py first."
    )

    exit()


with open(PARKING_FILE, "r") as file:

    parking_spaces = json.load(file)


print(
    f"Loaded {len(parking_spaces)} parking spaces."
)


# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Cannot open video.")

    exit()


# Video properties

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ==========================================
# VIDEO WRITER
# ==========================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ==========================================
# VEHICLE CLASSES
# ==========================================

VEHICLE_CLASSES = [

    "car",
    "motorcycle",
    "bus",
    "truck"

]


# ==========================================
# PROCESS VIDEO
# ==========================================

print("\nProcessing video...")

frame_count = 0


while True:

    ret, frame = cap.read()

    if not ret:

        break


    frame_count += 1


    # ======================================
    # YOLO DETECTION
    # ======================================

    results = model.predict(

        frame,

        conf=0.35,

        verbose=False

    )


    detected_vehicles = []


    # ======================================
    # EXTRACT VEHICLE BOXES
    # ======================================

    for result in results:

        if result.boxes is None:

            continue


        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = model.names[
                class_id
            ]


            if class_name not in VEHICLE_CLASSES:

                continue


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            center_x = (
                x1 + x2
            ) // 2


            center_y = (
                y1 + y2
            ) // 2


            detected_vehicles.append({

                "x1": x1,
                "y1": y1,

                "x2": x2,
                "y2": y2,

                "center_x": center_x,
                "center_y": center_y,

                "class": class_name,

                "confidence": confidence

            })


            # Draw vehicle box

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (255, 255, 0),

                2

            )


            cv2.putText(

                frame,

                f"{class_name} {confidence:.2f}",

                (x1, max(y1 - 5, 20)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 255, 0),

                2

            )


    # ======================================
    # CHECK EACH PARKING SPACE
    # ======================================

    occupied_count = 0

    empty_count = 0


    for space in parking_spaces:


        slot_id = space["id"]

        x = space["x"]

        y = space["y"]

        w = space["w"]

        h = space["h"]


        occupied = False


        # Check whether vehicle center
        # is inside parking space

        for vehicle in detected_vehicles:


            cx = vehicle["center_x"]

            cy = vehicle["center_y"]


            if (

                x <= cx <= x + w

                and

                y <= cy <= y + h

            ):

                occupied = True

                break


        # ==================================
        # OCCUPIED
        # ==================================

        if occupied:

            occupied_count += 1

            color = (
                0,
                0,
                255
            )

            status = "OCCUPIED"


        # ==================================
        # EMPTY
        # ==================================

        else:

            empty_count += 1

            color = (
                0,
                255,
                0
            )

            status = "EMPTY"


        # Draw parking space

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            color,

            2

        )


        # Parking number

        cv2.putText(

            frame,

            f"{slot_id}",

            (x + 5, y + 25),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2

        )


        # Status

        cv2.putText(

            frame,

            status,

            (x + 5, y + h - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            color,

            2

        )


    # ======================================
    # DISPLAY STATISTICS
    # ======================================

    total_spaces = len(
        parking_spaces
    )


    cv2.rectangle(

        frame,

        (10, 10),

        (420, 130),

        (30, 30, 30),

        -1

    )


    cv2.putText(

        frame,

        "SMART PARKING SYSTEM",

        (25, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"TOTAL SPACES: {total_spaces}",

        (25, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2

    )


    cv2.putText(

        frame,

        f"OCCUPIED: {occupied_count}",

        (25, 100),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 0, 255),

        2

    )


    cv2.putText(

        frame,

        f"EMPTY: {empty_count}",

        (220, 100),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2

    )


    # ======================================
    # SHOW VIDEO
    # ======================================

    cv2.imshow(

        "Smart Parking Detection",

        frame

    )


    # Save frame

    out.write(
        frame
    )


    # Press Q to stop

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# RELEASE
# ==========================================

cap.release()

out.release()

cv2.destroyAllWindows()


print("\nProcessing completed.")

print(
    f"Output saved to: {OUTPUT_VIDEO}"
)