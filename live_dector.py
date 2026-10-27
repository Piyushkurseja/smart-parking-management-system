import cv2
import json
import os
import numpy as np
from ultralytics import YOLO




# ============================================================
# SMART MALL PARKING SYSTEM
#
# VIDEO SCREEN  <---->  DASHBOARD
#
# V = Video
# D = Dashboard
# 1 = Floor 1
# 2 = Floor 2
# 3 = Floor 3
# Q = Exit
# ============================================================


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)

VIDEO_FILES = {
    1: os.path.join(BASE_DIR, "1.mp4"),
    2: os.path.join(BASE_DIR, "2.mp4"),
    3: os.path.join(BASE_DIR, "3.mp4")
}

PARKING_FILES = {
    1: os.path.join(
        BASE_DIR,
        "parking_spaces_1.json"
    ),
    2: os.path.join(
        BASE_DIR,
        "parking_spaces_2.json"
    ),
    3: os.path.join(
        BASE_DIR,
        "parking_spaces_3.json"
    )
}


# ============================================================
# YOLO
# ============================================================

CAR_CLASS = 2

CONFIDENCE = 0.35

STABLE_FRAMES = 3


# ============================================================
# COLORS
# ============================================================

WHITE = (245, 245, 245)

GREY = (155, 165, 175)

GREEN = (0, 220, 70)

RED = (0, 0, 255)

CYAN = (255, 220, 0)

BLUE = (255, 160, 0)

DARK = (7, 13, 20)

CARD = (18, 29, 40)


# ============================================================
# FLOOR SECTIONS
# ============================================================

SECTIONS = {
    1: "A",
    2: "B",
    3: "C"
}


# ============================================================
# LOAD PARKING SPACES
# ============================================================

def load_parking_spaces(filename):

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"\nParking file not found:\n"
            f"{filename}\n\n"
            f"Expected files:\n"
            f"parking_spaces_1.json\n"
            f"parking_spaces_2.json\n"
            f"parking_spaces_3.json"
        )

    with open(
        filename,
        "r"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # Support multiple JSON structures
    # --------------------------------------------------------

    if isinstance(data, dict):

        if "parking_spaces" in data:

            data = data["parking_spaces"]

        elif "spaces" in data:

            data = data["spaces"]

        elif "slots" in data:

            data = data["slots"]

        else:

            converted = []

            for key, value in data.items():

                if isinstance(value, dict):

                    obj = value.copy()

                    if "id" not in obj:

                        obj["id"] = key

                    converted.append(obj)

            data = converted

    if not isinstance(data, list):

        raise RuntimeError(
            f"Invalid parking JSON:\n{filename}"
        )

    spaces = []

    for index, item in enumerate(data):

        if not isinstance(item, dict):
            continue

        # ====================================================
        # SLOT ID
        # ====================================================

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

        polygon = None

        # ====================================================
        # POINTS / POLYGON / COORDINATES
        # ====================================================

        raw_points = None

        if "points" in item:

            raw_points = item["points"]

        elif "polygon" in item:

            raw_points = item["polygon"]

        elif "coordinates" in item:

            raw_points = item["coordinates"]

        if raw_points is not None:

            try:

                points = []

                for p in raw_points:

                    if isinstance(p, dict):

                        px = int(p["x"])
                        py = int(p["y"])

                    else:

                        px = int(p[0])
                        py = int(p[1])

                    points.append(
                        [px, py]
                    )

                if len(points) >= 3:

                    polygon = np.array(
                        points,
                        dtype=np.int32
                    )

            except Exception:

                polygon = None

        # ====================================================
        # X Y WIDTH HEIGHT
        # ====================================================

        if polygon is None:

            try:

                if all(
                    k in item
                    for k in [
                        "x",
                        "y",
                        "width",
                        "height"
                    ]
                ):

                    x = int(item["x"])
                    y = int(item["y"])

                    w = int(item["width"])
                    h = int(item["height"])

                    polygon = np.array(
                        [
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ],
                        dtype=np.int32
                    )

            except Exception:

                polygon = None

        # ====================================================
        # X1 Y1 X2 Y2
        # ====================================================

        if polygon is None:

            try:

                if all(
                    k in item
                    for k in [
                        "x1",
                        "y1",
                        "x2",
                        "y2"
                    ]
                ):

                    x1 = int(item["x1"])
                    y1 = int(item["y1"])

                    x2 = int(item["x2"])
                    y2 = int(item["y2"])

                    polygon = np.array(
                        [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]
                        ],
                        dtype=np.int32
                    )

            except Exception:

                polygon = None

        if polygon is None:
            continue

        x, y, w, h = cv2.boundingRect(
            polygon
        )

        cx = x + w // 2
        cy = y + h // 2

        spaces.append(
            {
                "id": slot_id,
                "polygon": polygon,
                "bbox": (
                    x,
                    y,
                    w,
                    h
                ),
                "center": (
                    cx,
                    cy
                ),
                "occupied": False
            }
        )

    if not spaces:

        raise RuntimeError(
            f"\nNo valid parking spaces found in:\n"
            f"{filename}"
        )

    return spaces


# ============================================================
# IOU
# ============================================================

def calculate_iou(
        box1,
        box2
):

    x1, y1, x2, y2 = box1

    a1, b1, a2, b2 = box2

    ix1 = max(
        x1,
        a1
    )

    iy1 = max(
        y1,
        b1
    )

    ix2 = min(
        x2,
        a2
    )

    iy2 = min(
        y2,
        b2
    )

    if ix2 <= ix1 or iy2 <= iy1:

        return 0.0

    intersection = (
        (ix2 - ix1) *
        (iy2 - iy1)
    )

    area1 = (
        (x2 - x1) *
        (y2 - y1)
    )

    area2 = (
        (a2 - a1) *
        (b2 - b1)
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:

        return 0.0

    return intersection / union


# ============================================================
# CAR INSIDE PARKING SPACE
# ============================================================

def car_inside_space(
        car_box,
        space
):

    x, y, w, h = space["bbox"]

    space_box = (
        x,
        y,
        x + w,
        y + h
    )

    # --------------------------------------------------------
    # IOU
    # --------------------------------------------------------

    if calculate_iou(
        car_box,
        space_box
    ) >= 0.25:

        return True

    # --------------------------------------------------------
    # CAR CENTER
    # --------------------------------------------------------

    x1, y1, x2, y2 = car_box

    cx = int(
        (x1 + x2) / 2
    )

    cy = int(
        (y1 + y2) / 2
    )

    result = cv2.pointPolygonTest(
        space["polygon"],
        (
            cx,
            cy
        ),
        False
    )

    return result >= 0


# ============================================================
# DETECT ONLY CARS
# ============================================================

def detect_cars(
        model,
        frame
):

    results = model.predict(
        frame,
        classes=[CAR_CLASS],
        conf=CONFIDENCE,
        iou=0.45,
        verbose=False
    )

    cars = []

    if not results:

        return cars

    result = results[0]

    if result.boxes is None:

        return cars

    for box in result.boxes:

        cls = int(
            box.cls[0].item()
        )

        # ONLY CAR
        if cls != CAR_CLASS:

            continue

        confidence = float(
            box.conf[0].item()
        )

        coords = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        x1, y1, x2, y2 = map(
            int,
            coords
        )

        cars.append(
            {
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "confidence":
                    confidence
            }
        )

    return cars


# ============================================================
# UPDATE PARKING STATUS
# ============================================================

def update_occupancy(
        spaces,
        cars,
        memory
):

    for space in spaces:

        slot_id = space["id"]

        car_found = False

        for car in cars:

            if car_inside_space(
                car["box"],
                space
            ):

                car_found = True

                break

        # ----------------------------------------------------
        # Stability
        # ----------------------------------------------------

        if car_found:

            memory[slot_id] = min(
                memory.get(
                    slot_id,
                    0
                ) + 1,
                STABLE_FRAMES
            )

        else:

            memory[slot_id] = max(
                memory.get(
                    slot_id,
                    0
                ) - 1,
                0
            )

        space["occupied"] = (
            memory[slot_id]
            >= STABLE_FRAMES
        )


# ============================================================
# DRAW CAR DETECTIONS
# ============================================================

def draw_cars(
        frame,
        cars
):

    for car in cars:

        x1, y1, x2, y2 = car["box"]

        cv2.rectangle(
            frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            BLUE,
            2
        )

        label = (
            f"CAR "
            f"{car['confidence']:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    20,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            WHITE,
            1,
            cv2.LINE_AA
        )


# ============================================================
# DRAW OCCUPIED PARKING POINTER
# ============================================================

def draw_occupied_pointer(
        frame,
        space
):

    if not space["occupied"]:

        return

    polygon = space["polygon"]

    cx, cy = space["center"]

    min_y = int(
        np.min(
            polygon[:, 1]
        )
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label_width = 155
    label_height = 48

    tx = max(
        5,
        cx -
        label_width // 2
    )

    ty = max(
        5,
        min_y - 65
    )

    # --------------------------------------------------------
    # Pointer
    # --------------------------------------------------------

    cv2.line(
        frame,
        (
            tx +
            label_width // 2,
            ty +
            label_height
        ),
        (
            cx,
            cy
        ),
        RED,
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (
            cx,
            cy
        ),
        6,
        RED,
        -1
    )

    # --------------------------------------------------------
    # ID BOX
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (
            tx,
            ty
        ),
        (
            tx +
            label_width,
            ty +
            label_height
        ),
        (
            8,
            8,
            8
        ),
        -1
    )

    cv2.rectangle(
        frame,
        (
            tx,
            ty
        ),
        (
            tx +
            label_width,
            ty +
            label_height
        ),
        RED,
        2
    )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    cv2.putText(
        frame,
        space["id"],
        (
            tx + 8,
            ty + 21
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # OCCUPIED
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "OCCUPIED",
        (
            tx + 8,
            ty + 41
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        RED,
        2,
        cv2.LINE_AA
    )


# ============================================================
# FLOOR STATISTICS
# ============================================================

def get_floor_statistics(
        spaces
):

    total = len(spaces)

    occupied = sum(
        1
        for s in spaces
        if s["occupied"]
    )

    available = (
        total -
        occupied
    )

    available_ids = [
        s["id"]
        for s in spaces
        if not s["occupied"]
    ]

    occupied_ids = [
        s["id"]
        for s in spaces
        if s["occupied"]
    ]

    return {
        "total": total,
        "occupied": occupied,
        "available": available,
        "available_ids":
            available_ids,
        "occupied_ids":
            occupied_ids
    }


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(all_floor_data, preferred_floor=None):
    """Choose a confirmed available slot across all floors.

    Prefer the selected floor when possible; otherwise search every floor.
    The floor cards still show availability for all three floors.
    """
    order = [1, 2, 3]
    if preferred_floor in (1, 2, 3):
        order = [preferred_floor] + [f for f in (1, 2, 3) if f != preferred_floor]

    for floor in order:
        data = all_floor_data.get(floor, {})
        available_ids = data.get("available_ids", [])
        if available_ids:
            return floor, available_ids[0]
    return None


# ============================================================
# DASHBOARD
# ============================================================

def create_dashboard(
        all_floor_data,
        selected_floor
):

    WIDTH = 1400
    HEIGHT = 850

    img = np.zeros(
        (
            HEIGHT,
            WIDTH,
            3
        ),
        dtype=np.uint8
    )

    img[:] = DARK

    # ========================================================
    # HEADER
    # ========================================================

    cv2.putText(
        img,
        "SMART MALL PARKING",
        (
            55,
            65
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        WHITE,
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "LIVE PARKING MANAGEMENT DASHBOARD",
        (
            58,
            98
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        CYAN,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        f"CURRENT FLOOR : {selected_floor}",
        (
            1060,
            65
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.line(
        img,
        (
            45,
            125
        ),
        (
            1355,
            125
        ),
        (
            55,
            70,
            85
        ),
        1
    )

    # ========================================================
    # FLOOR CARDS
    # ========================================================

    card_width = 405
    card_height = 300

    positions = {
        1: (
            45,
            160
        ),
        2: (
            497,
            160
        ),
        3: (
            949,
            160
        )
    }

    for floor in [
        1,
        2,
        3
    ]:

        x, y = positions[floor]

        data = all_floor_data[floor]

        selected = (
            floor ==
            selected_floor
        )

        # ----------------------------------------------------
        # Card
        # ----------------------------------------------------

        if selected:

            fill = (
                28,
                48,
                62
            )

            border = CYAN

        else:

            fill = CARD

            border = (
                55,
                70,
                85
            )

        cv2.rectangle(
            img,
            (
                x,
                y
            ),
            (
                x +
                card_width,
                y +
                card_height
            ),
            fill,
            -1
        )

        cv2.rectangle(
            img,
            (
                x,
                y
            ),
            (
                x +
                card_width,
                y +
                card_height
            ),
            border,
            2
        )

        # ----------------------------------------------------
        # Floor
        # ----------------------------------------------------

        cv2.putText(
            img,
            f"FLOOR {floor}",
            (
                x + 25,
                y + 45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            f"SECTION {SECTIONS[floor]}",
            (
                x + 25,
                y + 72
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            GREY,
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Available
        # ----------------------------------------------------

        cv2.putText(
            img,
            "AVAILABLE",
            (
                x + 25,
                y + 115
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            GREY,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            str(data["available"]),
            (
                x + 25,
                y + 155
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            GREEN,
            3,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Occupied
        # ----------------------------------------------------

        cv2.putText(
            img,
            "OCCUPIED",
            (
                x + 160,
                y + 115
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            GREY,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            str(data["occupied"]),
            (
                x + 160,
                y + 155
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            RED,
            3,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        cv2.putText(
            img,
            f"TOTAL SPACES : {data['total']}",
            (
                x + 25,
                y + 195
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            WHITE,
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Available parking
        # ----------------------------------------------------

        cv2.putText(
            img,
            "AVAILABLE PARKING",
            (
                x + 25,
                y + 230
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            GREY,
            1,
            cv2.LINE_AA
        )

        available_ids = data[
            "available_ids"
        ]

        if available_ids:

            # Show up to 2 IDs
            display_ids = available_ids[:2]

            text_value = "   ".join(
                display_ids
            )

            cv2.putText(
                img,
                text_value,
                (
                    x + 25,
                    y + 267
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                GREEN,
                2,
                cv2.LINE_AA
            )

        else:

            cv2.putText(
                img,
                "FULL",
                (
                    x + 25,
                    y + 267
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                RED,
                2,
                cv2.LINE_AA
            )

    # ========================================================
    # RECOMMENDED PARKING
    # ========================================================

    rec_x = 45
    rec_y = 500

    rec_w = 1310
    rec_h = 205

    cv2.rectangle(
        img,
        (
            rec_x,
            rec_y
        ),
        (
            rec_x + rec_w,
            rec_y + rec_h
        ),
        (
            15,
            38,
            25
        ),
        -1
    )

    cv2.rectangle(
        img,
        (
            rec_x,
            rec_y
        ),
        (
            rec_x + rec_w,
            rec_y + rec_h
        ),
        GREEN,
        2
    )

    cv2.putText(
        img,
        "RECOMMENDED PARKING",
        (
            rec_x + 30,
            rec_y + 48
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        WHITE,
        2,
        cv2.LINE_AA
    )

    recommendation = get_recommendation(
        all_floor_data, selected_floor
    )

    if recommendation:

        floor, slot = recommendation

        cv2.putText(
            img,
            "GO TO",
            (
                rec_x + 30,
                rec_y + 95
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            GREY,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            f"FLOOR {floor}",
            (
                rec_x + 30,
                rec_y + 135
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            GREEN,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            f"SECTION {SECTIONS[floor]}",
            (
                rec_x + 30,
                rec_y + 170
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            slot,
            (
                rec_x + 400,
                rec_y + 145
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            GREEN,
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            "AVAILABLE",
            (
                rec_x + 405,
                rec_y + 180
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            GREEN,
            2,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            img,
            "NO PARKING SPACE AVAILABLE",
            (
                rec_x + 30,
                rec_y + 125
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            RED,
            2,
            cv2.LINE_AA
        )

    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        img,
        "1  FLOOR 1",
        (
            50,
            765
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "2  FLOOR 2",
        (
            230,
            765
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "3  FLOOR 3",
        (
            410,
            765
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "V = VIDEO",
        (
            650,
            765
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        CYAN,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "Q = EXIT",
        (
            850,
            765
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        GREY,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "LIVE",
        (
            1240,
            765
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        GREEN,
        2,
        cv2.LINE_AA
    )

    return img


# ============================================================
# VIDEO SCREEN
# ============================================================

def run_video_screen(
        model,
        floor,
        spaces,
        cap,
        memory
):

    window_name = (
        f"PARKING VIDEO - FLOOR {floor}"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    while True:

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

                break

        # ====================================================
        # ONLY CAR DETECTION
        # ====================================================

        cars = detect_cars(
            model,
            frame
        )

        # ====================================================
        # PARKING STATUS
        # ====================================================

        update_occupancy(
            spaces,
            cars,
            memory
        )

        # ====================================================
        # DRAW CAR BOXES
        # ====================================================

        draw_cars(
            frame,
            cars
        )

        # ====================================================
        # DRAW OCCUPIED POINTERS
        # ====================================================

        for space in spaces:

            draw_occupied_pointer(
                frame,
                space
            )

        # ====================================================
        # STATISTICS
        # ====================================================

        total = len(spaces)

        occupied = sum(
            1
            for s in spaces
            if s["occupied"]
        )

        available = (
            total -
            occupied
        )

        # ====================================================
        # TOP INFORMATION
        # ====================================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (
                15,
                15
            ),
            (
                500,
                115
            ),
            (
                5,
                10,
                15
            ),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.86,
            frame,
            0.14,
            0
        )

        cv2.putText(
            frame,
            f"FLOOR {floor} | "
            f"SECTION {SECTIONS[floor]}",
            (
                30,
                48
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"AVAILABLE : {available}",
            (
                30,
                82
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            GREEN,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"OCCUPIED : {occupied}",
            (
                220,
                82
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            RED,
            2,
            cv2.LINE_AA
        )

        # ====================================================
        # CONTROLS
        # ====================================================

        cv2.putText(
            frame,
            "D = DASHBOARD   "
            "1/2/3 = FLOOR   "
            "Q = EXIT",
            (
                30,
                frame.shape[0] - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            WHITE,
            1,
            cv2.LINE_AA
        )

        # ====================================================
        # FULL VIDEO
        # ====================================================

        cv2.imshow(
            window_name,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if key == ord("q"):

            cv2.destroyWindow(
                window_name
            )

            return "exit"

        # ----------------------------------------------------
        # Dashboard
        # ----------------------------------------------------

        if key == ord("d"):

            cv2.destroyWindow(
                window_name
            )

            return "dashboard"

        # ----------------------------------------------------
        # Change floor directly
        # ----------------------------------------------------

        if key == ord("1"):

            cv2.destroyWindow(
                window_name
            )

            return 1

        if key == ord("2"):

            cv2.destroyWindow(
                window_name
            )

            return 2

        if key == ord("3"):

            cv2.destroyWindow(
                window_name
            )

            return 3

    cv2.destroyWindow(
        window_name
    )

    return "dashboard"


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("SMART MALL PARKING SYSTEM - LIVE MULTI-FLOOR UPDATE")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}\n"
            "Place yolov8n.pt in the same folder as this script."
        )

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded. Loading parking spaces and videos...")

    spaces_by_floor = {}
    captures = {}
    memories = {1: {}, 2: {}, 3: {}}
    latest_frames = {1: None, 2: None, 3: None}
    floor_has_frame = {1: False, 2: False, 3: False}

    try:
        for floor in (1, 2, 3):
            spaces_by_floor[floor] = load_parking_spaces(PARKING_FILES[floor])

            video_path = VIDEO_FILES[floor]
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video for Floor {floor} not found: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video for Floor {floor}: {video_path}")
            captures[floor] = cap

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Floor {floor}: {len(spaces_by_floor[floor])} spaces | {width}x{height} | {fps:.1f} FPS")

        selected_floor = 1
        current_mode = "dashboard"  # Start with dashboard so all floors are visible.
        video_window = "SMART MALL PARKING - VIDEO"
        dashboard_window = "SMART MALL PARKING - DASHBOARD"

        cv2.namedWindow(dashboard_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(dashboard_window, 1400, 850)

        print("\nControls: 1/2/3 = select floor | V = video | D = dashboard | Q/ESC = exit")
        print("All three video feeds are processed continuously in both display modes.\n")

        while True:
            # IMPORTANT FIX:
            # Read and update every floor on every loop, not only the selected floor.
            # This keeps the recommendation and all floor cards refreshed together.
            for floor in (1, 2, 3):
                cap = captures[floor]
                ret, frame = cap.read()

                if not ret:
                    # Loop the recorded clip, resetting its stability memory to avoid
                    # carrying occupancy evidence across the end/start boundary.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    memories[floor].clear()
                    for space in spaces_by_floor[floor]:
                        space["occupied"] = False
                    ret, frame = cap.read()

                if not ret or frame is None:
                    floor_has_frame[floor] = False
                    continue

                floor_has_frame[floor] = True
                processed_frame = frame.copy()
                cars = detect_cars(model, processed_frame)
                update_occupancy(spaces_by_floor[floor], cars, memories[floor])
                draw_cars(processed_frame, cars)

                for space in spaces_by_floor[floor]:
                    draw_occupied_pointer(processed_frame, space)

                total = len(spaces_by_floor[floor])
                occupied = sum(1 for sp in spaces_by_floor[floor] if sp["occupied"])
                available = total - occupied

                overlay = processed_frame.copy()
                cv2.rectangle(overlay, (15, 15), (520, 125), (5, 10, 15), -1)
                processed_frame = cv2.addWeighted(overlay, 0.84, processed_frame, 0.16, 0)
                cv2.putText(processed_frame, f"FLOOR {floor} | SECTION {SECTIONS[floor]}",
                            (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.70, WHITE, 2, cv2.LINE_AA)
                cv2.putText(processed_frame, f"AVAILABLE: {available}",
                            (30, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
                cv2.putText(processed_frame, f"OCCUPIED: {occupied}",
                            (250, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
                cv2.putText(processed_frame, "D=Dashboard  V=Video  1/2/3=Floor  Q=Exit",
                            (25, processed_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)
                latest_frames[floor] = processed_frame

            # Compute fresh stats for all floors after all feeds have been updated.
            all_floor_data = {
                floor: get_floor_statistics(spaces_by_floor[floor])
                for floor in (1, 2, 3)
            }

            if current_mode == "dashboard":
                dashboard_img = create_dashboard(all_floor_data, selected_floor)
                cv2.imshow(dashboard_window, dashboard_img)
            else:
                if latest_frames[selected_floor] is not None:
                    cv2.imshow(video_window, latest_frames[selected_floor])

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("1"):
                selected_floor = 1
            elif key == ord("2"):
                selected_floor = 2
            elif key == ord("3"):
                selected_floor = 3
            elif key in (ord("d"), ord("D")):
                current_mode = "dashboard"
                try:
                    cv2.destroyWindow(video_window)
                except cv2.error:
                    pass
            elif key in (ord("v"), ord("V")):
                current_mode = "video"
                try:
                    cv2.destroyWindow(dashboard_window)
                except cv2.error:
                    pass
                cv2.namedWindow(video_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(video_window, 1280, 720)

    finally:
        for cap in captures.values():
            cap.release()
        cv2.destroyAllWindows()

    print("\nSMART MALL PARKING SYSTEM CLOSED.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()