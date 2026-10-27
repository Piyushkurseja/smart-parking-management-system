import cv2
import json
import os
import time
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_FILES = {
    1: os.path.join(BASE_DIR, "1.mp4"),
    2: os.path.join(BASE_DIR, "2.mp4"),
    3: os.path.join(BASE_DIR, "3.mp4"),
}

PARKING_FILES = {
    1: os.path.join(BASE_DIR, "parking_spaces_1.json"),
    2: os.path.join(BASE_DIR, "parking_spaces_2.json"),
    3: os.path.join(BASE_DIR, "parking_spaces_3.json"),
}

SECTION_NAMES = {
    1: "A",
    2: "B",
    3: "C",
}

# Change this if your model is different
MODEL_PATH = "yolov8n.pt"

# COCO class 2 = car
CAR_CLASS_ID = 2

# Minimum overlap required between car and parking space
OCCUPANCY_THRESHOLD = 0.25

# How many frames a car must remain detected
# before changing a space to occupied.
STABLE_FRAMES = 3


# ============================================================
# COLORS
# ============================================================

WHITE = (245, 245, 245)
BLACK = (0, 0, 0)

GREEN = (0, 220, 60)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)

BLUE = (255, 150, 0)
GREY = (150, 160, 170)

DARK = (8, 15, 22)
CARD = (15, 25, 35)


# ============================================================
# PARKING SPACE LOADER
# ============================================================

def load_parking_spaces(filename):

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"\nParking file not found:\n{filename}\n\n"
            f"Create the file using setup_spaces.py."
        )

    with open(filename, "r") as f:
        data = json.load(f)

    # --------------------------------------------------------
    # Support different JSON structures
    # --------------------------------------------------------

    if isinstance(data, dict):

        if "parking_spaces" in data:
            data = data["parking_spaces"]

        elif "spaces" in data:
            data = data["spaces"]

        elif "slots" in data:
            data = data["slots"]

        else:

            # Sometimes JSON is:
            # {"1A-101": {...}, "1A-102": {...}}

            converted = []

            for key, value in data.items():

                if isinstance(value, dict):

                    item = value.copy()

                    if "id" not in item:
                        item["id"] = key

                    converted.append(item)

            data = converted

    if not isinstance(data, list):

        raise RuntimeError(
            f"\nInvalid parking JSON:\n{filename}\n"
        )

    spaces = []

    for index, item in enumerate(data):

        if not isinstance(item, dict):
            continue

        slot_id = str(
            item.get(
                "id",
                item.get(
                    "slot_id",
                    item.get(
                        "name",
                        f"P{index + 1:03d}"
                    )
                )
            )
        )

        points = None

        # ----------------------------------------------------
        # polygon / points / coordinates
        # ----------------------------------------------------

        if "points" in item:
            points = item["points"]

        elif "polygon" in item:
            points = item["polygon"]

        elif "coordinates" in item:
            points = item["coordinates"]

        if points is not None:

            try:

                pts = []

                for p in points:

                    if isinstance(p, dict):
                        x = int(p["x"])
                        y = int(p["y"])

                    else:
                        x = int(p[0])
                        y = int(p[1])

                    pts.append([x, y])

                if len(pts) >= 3:

                    pts = np.array(
                        pts,
                        dtype=np.int32
                    )

                    x, y, w, h = cv2.boundingRect(pts)

                    spaces.append({
                        "id": slot_id,
                        "polygon": pts,
                        "bbox": (x, y, w, h),
                        "center": (
                            x + w // 2,
                            y + h // 2
                        )
                    })

                    continue

            except Exception:
                pass

        # ----------------------------------------------------
        # x/y/width/height
        # ----------------------------------------------------

        try:

            if all(
                key in item
                for key in ["x", "y", "width", "height"]
            ):

                x = int(item["x"])
                y = int(item["y"])
                w = int(item["width"])
                h = int(item["height"])

                pts = np.array([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ], dtype=np.int32)

                spaces.append({
                    "id": slot_id,
                    "polygon": pts,
                    "bbox": (x, y, w, h),
                    "center": (
                        x + w // 2,
                        y + h // 2
                    )
                })

                continue

        except Exception:
            pass

        # ----------------------------------------------------
        # x1/y1/x2/y2
        # ----------------------------------------------------

        try:

            if all(
                key in item
                for key in ["x1", "y1", "x2", "y2"]
            ):

                x1 = int(item["x1"])
                y1 = int(item["y1"])
                x2 = int(item["x2"])
                y2 = int(item["y2"])

                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)

                pts = np.array([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ], dtype=np.int32)

                spaces.append({
                    "id": slot_id,
                    "polygon": pts,
                    "bbox": (x, y, w, h),
                    "center": (
                        x + w // 2,
                        y + h // 2
                    )
                })

        except Exception:
            pass

    if len(spaces) == 0:

        raise RuntimeError(
            f"\nNo valid parking spaces found in:\n"
            f"{filename}\n\n"
            f"Expected each space to contain:\n"
            f"points / polygon / coordinates\n"
            f"or x / y / width / height\n"
            f"or x1 / y1 / x2 / y2"
        )

    print(
        f"Loaded {len(spaces)} parking spaces from "
        f"{os.path.basename(filename)}"
    )

    return spaces


# ============================================================
# IOU FUNCTION
# ============================================================

def calculate_iou(box1, box2):

    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2

    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)

    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)

    if inter_x2 <= inter_x1:
        return 0.0

    if inter_y2 <= inter_y1:
        return 0.0

    intersection = (
        (inter_x2 - inter_x1) *
        (inter_y2 - inter_y1)
    )

    area1 = (
        (x2 - x1) *
        (y2 - y1)
    )

    area2 = (
        (a2 - a1) *
        (b2 - b1)
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# CHECK WHETHER CAR OCCUPIES SPACE
# ============================================================

def car_inside_space(car_box, space):

    sx, sy, sw, sh = space["bbox"]

    space_box = (
        sx,
        sy,
        sx + sw,
        sy + sh
    )

    iou = calculate_iou(
        car_box,
        space_box
    )

    if iou >= OCCUPANCY_THRESHOLD:
        return True

    # --------------------------------------------------------
    # Center point test
    # --------------------------------------------------------

    x1, y1, x2, y2 = car_box

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    polygon = space["polygon"]

    inside = cv2.pointPolygonTest(
        polygon,
        (cx, cy),
        False
    )

    if inside >= 0:
        return True

    return False


# ============================================================
# DETECT CARS
# ============================================================

def detect_cars(model, frame):

    results = model.predict(
        frame,
        classes=[CAR_CLASS_ID],
        conf=0.35,
        iou=0.45,
        verbose=False
    )

    cars = []

    if len(results) == 0:
        return cars

    result = results[0]

    if result.boxes is None:
        return cars

    for box in result.boxes:

        cls = int(
            box.cls[0].item()
        )

        if cls != CAR_CLASS_ID:
            continue

        confidence = float(
            box.conf[0].item()
        )

        if confidence < 0.35:
            continue

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = map(
            int,
            xyxy
        )

        cars.append(
            (x1, y1, x2, y2, confidence)
        )

    return cars


# ============================================================
# UPDATE PARKING STATUS
# ============================================================

def update_spaces(
        spaces,
        cars,
        stability
):

    occupied_ids = set()

    for space in spaces:

        is_detected = False

        for car in cars:

            car_box = car[:4]

            if car_inside_space(
                car_box,
                space
            ):

                is_detected = True
                break

        slot_id = space["id"]

        if is_detected:

            stability[slot_id] = min(
                stability.get(slot_id, 0) + 1,
                STABLE_FRAMES
            )

        else:

            stability[slot_id] = max(
                stability.get(slot_id, 0) - 1,
                0
            )

        # ----------------------------------------------------
        # Only declare occupied after stable detection
        # ----------------------------------------------------

        space["occupied"] = (
            stability[slot_id]
            >= STABLE_FRAMES
        )

        if space["occupied"]:
            occupied_ids.add(slot_id)

    return occupied_ids


# ============================================================
# DRAW OCCUPIED POINTER
# ============================================================

def draw_occupied_pointer(
        frame,
        space
):

    if not space.get("occupied", False):
        return

    polygon = space["polygon"]

    # Find center of parking space
    cx, cy = space["center"]

    # --------------------------------------------------------
    # Find label position ABOVE the car
    # --------------------------------------------------------

    min_y = int(
        np.min(polygon[:, 1])
    )

    label_w = 150
    label_h = 42

    tx = max(
        5,
        cx - label_w // 2
    )

    ty = max(
        5,
        min_y - 60
    )

    # --------------------------------------------------------
    # Pointer line
    # --------------------------------------------------------

    cv2.line(
        frame,
        (
            tx + label_w // 2,
            ty + label_h
        ),
        (cx, cy),
        RED,
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Pointer dot
    # --------------------------------------------------------

    cv2.circle(
        frame,
        (cx, cy),
        5,
        RED,
        -1
    )

    # --------------------------------------------------------
    # Occupied label
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (tx, ty),
        (tx + label_w, ty + label_h),
        (10, 10, 10),
        -1
    )

    cv2.rectangle(
        frame,
        (tx, ty),
        (tx + label_w, ty + label_h),
        RED,
        2
    )

    cv2.putText(
        frame,
        space["id"],
        (tx + 8, ty + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "OCCUPIED",
        (tx + 8, ty + 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        RED,
        2,
        cv2.LINE_AA
    )


# ============================================================
# DRAW CURRENT FLOOR HEADER
# ============================================================

def draw_header(
        frame,
        floor,
        empty,
        occupied,
        total
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (390, 110),
        (5, 10, 15),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.88,
        frame,
        0.12,
        0
    )

    cv2.putText(
        frame,
        f"FLOOR {floor} | SECTION {SECTION_NAMES[floor]}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"AVAILABLE : {empty}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        GREEN,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"OCCUPIED : {occupied}",
        (200, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        RED,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"TOTAL : {total}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        GREY,
        1,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# DASHBOARD
# ============================================================

def draw_dashboard(
        frame,
        floor_data,
        current_floor,
        recommended
):

    H, W = frame.shape[:2]

    dashboard_width = 380

    # --------------------------------------------------------
    # Dashboard starts at right side
    # --------------------------------------------------------

    x0 = W - dashboard_width

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x0, 0),
        (W, H),
        DARK,
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.95,
        frame,
        0.05,
        0
    )

    def txt(
            value,
            x,
            y,
            size=0.55,
            color=WHITE,
            thickness=1
    ):

        cv2.putText(
            frame,
            str(value),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            thickness,
            cv2.LINE_AA
        )

    # ========================================================
    # TITLE
    # ========================================================

    txt(
        "SMART MALL",
        x0 + 25,
        45,
        0.78,
        WHITE,
        2
    )

    txt(
        "PARKING SYSTEM",
        x0 + 25,
        75,
        0.58,
        (0, 200, 255),
        2
    )

    cv2.line(
        frame,
        (x0 + 20, 95),
        (W - 20, 95),
        (50, 60, 70),
        1
    )

    # ========================================================
    # CURRENT FLOOR
    # ========================================================

    txt(
        f"CURRENT FLOOR : {current_floor}",
        x0 + 25,
        130,
        0.65,
        WHITE,
        2
    )

    txt(
        f"SECTION {SECTION_NAMES[current_floor]}",
        x0 + 25,
        155,
        0.50,
        GREY,
        1
    )

    # ========================================================
    # FLOOR CARDS
    # ========================================================

    y = 185

    for floor in [1, 2, 3]:

        data = floor_data[floor]

        total = data["total"]
        occupied = data["occupied"]
        empty = data["empty"]

        active = floor == current_floor

        if active:
            card_color = (25, 45, 60)
        else:
            card_color = (15, 25, 35)

        cv2.rectangle(
            frame,
            (x0 + 15, y),
            (W - 15, y + 120),
            card_color,
            -1
        )

        # Floor number
        txt(
            f"FLOOR {floor}",
            x0 + 30,
            y + 30,
            0.60,
            WHITE,
            2
        )

        txt(
            f"SECTION {SECTION_NAMES[floor]}",
            x0 + 30,
            y + 53,
            0.42,
            GREY,
            1
        )

        # Available
        txt(
            "AVAILABLE",
            x0 + 160,
            y + 28,
            0.35,
            GREY,
            1
        )

        txt(
            empty,
            x0 + 165,
            y + 58,
            0.85,
            GREEN,
            2
        )

        # Occupied
        txt(
            "OCCUPIED",
            x0 + 250,
            y + 28,
            0.35,
            GREY,
            1
        )

        txt(
            occupied,
            x0 + 260,
            y + 58,
            0.85,
            RED,
            2
        )

        txt(
            f"TOTAL SPACES : {total}",
            x0 + 30,
            y + 92,
            0.42,
            WHITE,
            1
        )

        y += 135

    # ========================================================
    # AVAILABLE PARKING
    # ========================================================

    cv2.line(
        frame,
        (x0 + 20, y - 5),
        (W - 20, y - 5),
        (50, 60, 70),
        1
    )

    txt(
        "AVAILABLE PARKING",
        x0 + 25,
        y + 35,
        0.62,
        WHITE,
        2
    )

    current_available = floor_data[
        current_floor
    ]["available_ids"]

    # --------------------------------------------------------
    # Show available IDs
    # --------------------------------------------------------

    if len(current_available) == 0:

        txt(
            "NO PARKING AVAILABLE",
            x0 + 25,
            y + 70,
            0.48,
            RED,
            2
        )

    else:

        display_ids = current_available[:10]

        line_y = y + 70

        for i, slot_id in enumerate(display_ids):

            col = i % 2
            row = i // 2

            px = x0 + 25 + col * 170
            py = line_y + row * 38

            cv2.rectangle(
                frame,
                (px, py - 23),
                (px + 150, py + 8),
                (10, 45, 25),
                -1
            )

            cv2.rectangle(
                frame,
                (px, py - 23),
                (px + 150, py + 8),
                GREEN,
                1
            )

            txt(
                slot_id,
                px + 10,
                py,
                0.48,
                GREEN,
                2
            )

    # ========================================================
    # RECOMMENDED PARKING
    # ========================================================

    rec_y = H - 175

    cv2.rectangle(
        frame,
        (x0 + 15, rec_y),
        (W - 15, H - 15),
        (12, 50, 22),
        -1
    )

    cv2.rectangle(
        frame,
        (x0 + 15, rec_y),
        (W - 15, H - 15),
        GREEN,
        1
    )

    txt(
        "RECOMMENDED PARKING",
        x0 + 30,
        rec_y + 30,
        0.55,
        WHITE,
        2
    )

    if recommended is not None:

        rf, rs = recommended

        txt(
            "GO TO",
            x0 + 30,
            rec_y + 60,
            0.40,
            GREY,
            1
        )

        txt(
            f"FLOOR {rf}",
            x0 + 30,
            rec_y + 92,
            0.70,
            GREEN,
            2
        )

        txt(
            f"SECTION {SECTION_NAMES[rf]}",
            x0 + 30,
            rec_y + 120,
            0.48,
            WHITE,
            1
        )

        txt(
            rs,
            x0 + 30,
            rec_y + 153,
            0.70,
            GREEN,
            3
        )

    else:

        txt(
            "PARKING FULL",
            x0 + 30,
            rec_y + 90,
            0.60,
            RED,
            2
        )

    return frame


# ============================================================
# FIND BEST AVAILABLE PARKING
# ============================================================

def find_recommendation(
        floor_data,
        current_floor
):

    # First try current floor
    current_available = floor_data[
        current_floor
    ]["available_ids"]

    if current_available:

        return (
            current_floor,
            current_available[0]
        )

    # Otherwise search other floors
    for floor in [1, 2, 3]:

        if floor == current_floor:
            continue

        available = floor_data[
            floor
        ]["available_ids"]

        if available:

            return (
                floor,
                available[0]
            )

    return None


# ============================================================
# CREATE FLOOR DATA
# ============================================================

def create_floor_data(
        spaces_by_floor,
        occupied_by_floor
):

    result = {}

    for floor in [1, 2, 3]:

        spaces = spaces_by_floor[floor]

        occupied_ids = occupied_by_floor[floor]

        all_ids = [
            s["id"]
            for s in spaces
        ]

        available_ids = [
            sid
            for sid in all_ids
            if sid not in occupied_ids
        ]

        result[floor] = {

            "total": len(all_ids),

            "occupied": len(
                occupied_ids
            ),

            "empty": len(
                available_ids
            ),

            "available_ids":
                available_ids
        }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SMART MALL PARKING SYSTEM")
    print("=" * 60)

    print()
    print("Detection Mode : ONLY CAR")
    print("YOLO Class     : 2")
    print()

    # ========================================================
    # LOAD YOLO
    # ========================================================

    print("Loading YOLO model...")

    model = YOLO(
        os.path.join(
            BASE_DIR,
            MODEL_PATH
        )
    )

    print("YOLO model loaded successfully.")

    # ========================================================
    # LOAD PARKING SPACES
    # ========================================================

    spaces_by_floor = {}

    for floor in [1, 2, 3]:

        spaces_by_floor[floor] = \
            load_parking_spaces(
                PARKING_FILES[floor]
            )

    # ========================================================
    # OPEN VIDEOS
    # ========================================================

    captures = {}

    for floor in [1, 2, 3]:

        video = VIDEO_FILES[floor]

        if not os.path.exists(video):

            raise FileNotFoundError(
                f"\nVideo not found:\n{video}"
            )

        cap = cv2.VideoCapture(video)

        if not cap.isOpened():

            raise RuntimeError(
                f"Unable to open:\n{video}"
            )

        captures[floor] = cap

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        print(
            f"Floor {floor}: "
            f"{width}x{height} "
            f"| FPS {fps:.1f}"
        )

    print()
    print("=" * 60)
    print("CONTROLS")
    print("=" * 60)
    print("1 = Floor 1")
    print("2 = Floor 2")
    print("3 = Floor 3")
    print("Q = Exit")
    print("=" * 60)

    # ========================================================
    # STABILITY MEMORY
    # ========================================================

    stability = {
        1: {},
        2: {},
        3: {}
    }

    # ========================================================
    # CURRENT FLOOR
    # ========================================================

    current_floor = 1

    cv2.namedWindow(
        "SMART MALL PARKING",
        cv2.WINDOW_NORMAL
    )

    # Fullscreen
    cv2.setWindowProperty(
        "SMART MALL PARKING",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        cap = captures[
            current_floor
        ]

        ret, frame = cap.read()

        # ----------------------------------------------------
        # Restart video when finished
        # ----------------------------------------------------

        if not ret:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            ret, frame = cap.read()

            if not ret:
                continue

        # ====================================================
        # DETECT CARS
        # ====================================================

        cars = detect_cars(
            model,
            frame
        )

        # ====================================================
        # UPDATE CURRENT FLOOR
        # ====================================================

        occupied_ids = update_spaces(
            spaces_by_floor[current_floor],
            cars,
            stability[current_floor]
        )

        # ====================================================
        # DRAW ONLY OCCUPIED POINTERS
        # ====================================================

        for space in spaces_by_floor[
            current_floor
        ]:

            if space.get(
                "occupied",
                False
            ):

                draw_occupied_pointer(
                    frame,
                    space
                )

        # ====================================================
        # CALCULATE FLOOR DATA
        # ====================================================

        occupied_by_floor = {}

        for floor in [1, 2, 3]:

            occupied_by_floor[floor] = set()

            for space in spaces_by_floor[floor]:

                if space.get(
                    "occupied",
                    False
                ):

                    occupied_by_floor[
                        floor
                    ].add(
                        space["id"]
                    )

        floor_data = create_floor_data(
            spaces_by_floor,
            occupied_by_floor
        )

        # ====================================================
        # RECOMMENDATION
        # ====================================================

        recommended = find_recommendation(
            floor_data,
            current_floor
        )

        # ====================================================
        # CURRENT FLOOR HEADER
        # ====================================================

        data = floor_data[
            current_floor
        ]

        frame = draw_header(
            frame,
            current_floor,
            data["empty"],
            data["occupied"],
            data["total"]
        )

        # ====================================================
        # DASHBOARD
        # ====================================================

        frame = draw_dashboard(
            frame,
            floor_data,
            current_floor,
            recommended
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "SMART MALL PARKING",
            frame
        )

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        elif key == ord("1"):

            current_floor = 1

            print(
                "Switched to Floor 1"
            )

        elif key == ord("2"):

            current_floor = 2

            print(
                "Switched to Floor 2"
            )

        elif key == ord("3"):

            current_floor = 3

            print(
                "Switched to Floor 3"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    for cap in captures.values():

        cap.release()

    cv2.destroyAllWindows()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()