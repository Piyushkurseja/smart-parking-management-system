import cv2
import json
import os
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# ============================================================
# SMART MALL PARKING SYSTEM - 3 PARKING ZONES / FLOORS
#
# Keys:
#   1 = Zone/Floor 1
#   2 = Zone/Floor 2
#   3 = Zone/Floor 3
#   Q / ESC = Exit
#
# IMPORTANT:
# - No fixed number of parking spaces.
# - Spaces are read automatically from each JSON file.
# - EMPTY spaces have NO label on the camera.
# - MOVING vehicles have NO label on the camera.
# - Only confirmed OCCUPIED spaces show:
#       P001 OCCUPIED
# - Dashboard shows counts and recommended parking.
# ============================================================

MODEL_PATH = "yolov8n.pt"

ZONES = {
    1: {
        "video": "1.mp4",
        "json": "parking_spaces_1.json",
        "floor": 1,
        "section": "A"
    },
    2: {
        "video": "2.mp4",
        "json": "parking_spaces_2.json",
        "floor": 2,
        "section": "B"
    },
    3: {
        "video": "3.mp4",
        "json": "parking_spaces_3.json",
        "floor": 3,
        "section": "C"
    }
}

CONFIDENCE = 0.25
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

PARK_CONFIRM_SECONDS = 2.5
EMPTY_CONFIRM_SECONDS = 2.0
MIN_OVERLAP = 0.30
MAX_MOVEMENT = 8

WINDOW_NAME = "SMART MALL PARKING SYSTEM"

# ============================================================
# ZONE STATE
# ============================================================

zone_data = {}
vehicle_history = defaultdict(lambda: defaultdict(list))


# ============================================================
# LOAD PARKING SPACES
# ============================================================

def load_parking_spaces(filename, frame_width, frame_height):

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"\nParking file not found:\n{os.path.abspath(filename)}\n"
        )

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "spaces" in data:
            data = data["spaces"]
        elif "parking_spaces" in data:
            data = data["parking_spaces"]
        else:
            data = [data]

    if not isinstance(data, list):
        data = []

    spaces = []

    for index, item in enumerate(data, start=1):

        if not isinstance(item, dict):
            continue

        sid = str(
            item.get("id")
            or item.get("name")
            or item.get("slot_id")
            or f"P{index:03d}"
        )

        points = None

        # ----------------------------------------------------
        # points / polygon / coordinates
        # ----------------------------------------------------

        raw_points = (
            item.get("points")
            or item.get("polygon")
            or item.get("coordinates")
        )

        if isinstance(raw_points, list):

            temp = []

            for p in raw_points:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        px = max(
                            0,
                            min(int(float(p[0])), frame_width - 1)
                        )
                        py = max(
                            0,
                            min(int(float(p[1])), frame_height - 1)
                        )
                        temp.append([px, py])
                    except:
                        pass

            if len(temp) >= 3:
                points = temp

        # ----------------------------------------------------
        # x / y / width / height
        # ----------------------------------------------------

        if points is None and all(
            k in item for k in ["x", "y", "width", "height"]
        ):
            try:
                x = int(float(item["x"]))
                y = int(float(item["y"]))
                w = int(float(item["width"]))
                h = int(float(item["height"]))

                if w > 0 and h > 0:

                    x1 = max(0, min(x, frame_width - 1))
                    y1 = max(0, min(y, frame_height - 1))
                    x2 = max(0, min(x + w, frame_width - 1))
                    y2 = max(0, min(y + h, frame_height - 1))

                    if x2 > x1 and y2 > y1:
                        points = [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]
                        ]
            except:
                pass

        # ----------------------------------------------------
        # x1 / y1 / x2 / y2
        # ----------------------------------------------------

        if points is None and all(
            k in item for k in ["x1", "y1", "x2", "y2"]
        ):
            try:
                x1 = max(
                    0,
                    min(int(float(item["x1"])), frame_width - 1)
                )
                y1 = max(
                    0,
                    min(int(float(item["y1"])), frame_height - 1)
                )
                x2 = max(
                    0,
                    min(int(float(item["x2"])), frame_width - 1)
                )
                y2 = max(
                    0,
                    min(int(float(item["y2"])), frame_height - 1)
                )

                if x2 > x1 and y2 > y1:
                    points = [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ]
            except:
                pass

        if points is not None:
            spaces.append({
                "id": sid,
                "points": points
            })

    # Remove duplicate IDs
    used = set()

    for i, space in enumerate(spaces, start=1):
        sid = space["id"]

        if sid in used:
            sid = f"P{i:03d}"

        space["id"] = sid
        used.add(sid)

    if not spaces:
        raise RuntimeError(
            f"\nNo valid parking spaces found in:\n"
            f"{os.path.abspath(filename)}\n\n"
            "Run setup_spaces.py for this zone first."
        )

    print(f"Loaded {len(spaces)} parking spaces from {filename}")

    return spaces


# ============================================================
# GEOMETRY
# ============================================================

def polygon_center(points):
    arr = np.array(points, dtype=np.int32)
    return (
        int(arr[:, 0].mean()),
        int(arr[:, 1].mean())
    )


def overlap_ratio(box, polygon):

    x1, y1, x2, y2 = map(int, box)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    mask = np.zeros(
        (y2 - y1, x2 - x1),
        dtype=np.uint8
    )

    shifted = []

    for px, py in polygon:
        shifted.append([
            int(px - x1),
            int(py - y1)
        ])

    cv2.fillPoly(
        mask,
        [np.array(shifted, dtype=np.int32)],
        255
    )

    intersection = cv2.countNonZero(mask)
    vehicle_area = max(1, (x2 - x1) * (y2 - y1))

    return intersection / vehicle_area


# ============================================================
# MOVEMENT TRACKING
# ============================================================

def get_movement(zone, track_id, cx, cy):

    history = vehicle_history[zone][track_id]

    history.append((cx, cy))

    if len(history) > 8:
        history.pop(0)

    if len(history) < 3:
        return 0

    x0, y0 = history[0]

    return float(
        ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5
    )


# ============================================================
# MATCH VEHICLE TO PARKING SPACE
# ============================================================

def find_vehicle(space, detections):

    best = None
    best_ratio = 0

    for detection in detections:

        ratio = overlap_ratio(
            detection["box"],
            space["points"]
        )

        if ratio >= MIN_OVERLAP and ratio > best_ratio:
            best = detection
            best_ratio = ratio

    return best


# ============================================================
# UPDATE OCCUPANCY
# ============================================================

def update_parking(zone, spaces, detections):

    now = time.time()

    states = zone_data[zone]["states"]

    for space in spaces:

        sid = space["id"]
        state = states[sid]

        vehicle = find_vehicle(space, detections)

        # ----------------------------------------------------
        # No vehicle
        # ----------------------------------------------------

        if vehicle is None:

            if state["status"] == "OCCUPIED":

                if state["empty_since"] is None:
                    state["empty_since"] = now

                elif (
                    now - state["empty_since"]
                    >= EMPTY_CONFIRM_SECONDS
                ):
                    state["status"] = "EMPTY"
                    state["vehicle_id"] = None
                    state["inside_since"] = None
                    state["empty_since"] = None

            else:

                state["status"] = "EMPTY"
                state["vehicle_id"] = None
                state["inside_since"] = None
                state["empty_since"] = None

            continue

        # ----------------------------------------------------
        # Vehicle found
        # ----------------------------------------------------

        track_id = vehicle["id"]
        movement = vehicle["movement"]

        state["empty_since"] = None

        # ----------------------------------------------------
        # Moving vehicle
        #
        # IMPORTANT:
        # Moving vehicles do NOT get OCCUPIED.
        # ----------------------------------------------------

        if movement > MAX_MOVEMENT:

            state["status"] = "MOVING"
            state["vehicle_id"] = track_id
            state["inside_since"] = None

            continue

        # ----------------------------------------------------
        # Stationary vehicle
        # ----------------------------------------------------

        if state["vehicle_id"] != track_id:

            state["vehicle_id"] = track_id
            state["inside_since"] = now
            state["status"] = "MOVING"

        else:

            if state["inside_since"] is None:
                state["inside_since"] = now

            parked_time = now - state["inside_since"]

            if parked_time >= PARK_CONFIRM_SECONDS:
                state["status"] = "OCCUPIED"
            else:
                state["status"] = "MOVING"


# ============================================================
# OCCUPIED POINTER ONLY
# ============================================================

def draw_pointer(frame, space, status):

    # EMPTY = nothing
    # MOVING = nothing
    # ONLY OCCUPIED is shown.

    if status != "OCCUPIED":
        return

    points = space["points"]
    sid = space["id"]

    cx, cy = polygon_center(points)

    top_point = min(
        points,
        key=lambda p: p[1]
    )

    tx, ty = top_point

    label = f"{sid}  OCCUPIED"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 2

    (tw, th), _ = cv2.getTextSize(
        label,
        font,
        scale,
        thickness
    )

    label_x = cx
    label_y = max(30, ty - 18)

    label_x = max(
        tw // 2 + 5,
        min(
            label_x,
            frame.shape[1] - tw // 2 - 5
        )
    )

    bx1 = label_x - tw // 2 - 7
    by1 = label_y - th - 7
    bx2 = label_x + tw // 2 + 7
    by2 = label_y + 6

    # Small label background
    cv2.rectangle(
        frame,
        (bx1, by1),
        (bx2, by2),
        (10, 10, 10),
        -1
    )

    # Red occupied text
    cv2.putText(
        frame,
        label,
        (
            label_x - tw // 2,
            label_y
        ),
        font,
        scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA
    )

    # Pointer
    cv2.line(
        frame,
        (label_x, by2),
        (cx, cy),
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (cx, cy),
        5,
        (0, 0, 255),
        -1
    )


# ============================================================
# DASHBOARD
# ============================================================

def draw_dashboard(frame, zone, floor, section, spaces):
    """Compatibility wrapper.

    The professional dashboard is now rendered in a separate window so the
    camera/video remains completely unobstructed.  The function is retained
    because process_frame already calls it.
    """
    return frame


# ============================================================
# PROFESSIONAL COMMAND-CENTER DASHBOARD
# ============================================================

def create_dashboard(current_zone):
    """Create a standalone, mall-ready parking command center dashboard."""

    W, H = 1440, 900
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Premium dark UI palette (OpenCV uses BGR)
    BG = (9, 15, 24)
    PANEL = (16, 25, 37)
    PANEL_2 = (20, 31, 45)
    BORDER = (42, 58, 75)
    WHITE = (242, 246, 250)
    MUTED = (145, 158, 173)
    CYAN = (255, 190, 20)
    GREEN = (65, 220, 80)
    RED = (70, 70, 240)
    ORANGE = (40, 165, 255)
    BLUE = (235, 135, 45)
    GOLD = (30, 190, 255)

    img[:] = BG

    def txt(text, x, y, scale=0.6, color=WHITE, thick=1):
        cv2.putText(img, str(text), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                    cv2.LINE_AA)

    def rounded_box(x1, y1, x2, y2, fill=PANEL, border=BORDER, radius=16, thickness=1):
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), fill, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), fill, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, fill, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, fill, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, fill, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, fill, -1)
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), border, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), border, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, border, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, border, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, border, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border, thickness)

    def floor_stats(z):
        if z not in zone_data:
            return 0, 0, 0, []
        spaces_z = zone_data[z]["spaces"]
        states_z = zone_data[z]["states"]
        occ = 0
        mov = 0
        available_ids = []
        for sp in spaces_z:
            st = states_z[sp["id"]]["status"]
            if st == "OCCUPIED":
                occ += 1
            elif st == "MOVING":
                mov += 1
            else:
                available_ids.append(sp["id"])
        return len(spaces_z), occ, mov, available_ids

    stats = {}
    for z in (1, 2, 3):
        total, occ, mov, ids = floor_stats(z)
        stats[z] = {
            "total": total,
            "occupied": occ,
            "moving": mov,
            "available": len(ids),
            "ids": ids
        }

    total_all = sum(v["total"] for v in stats.values())
    occupied_all = sum(v["occupied"] for v in stats.values())
    moving_all = sum(v["moving"] for v in stats.values())
    available_all = max(0, total_all - occupied_all)
    occupancy_pct = (occupied_all / total_all * 100) if total_all else 0

    current = stats[current_zone]
    current_cfg = ZONES[current_zone]

    # Select a recommendation across ALL floors, not only the selected floor.
    # Prefer the currently selected floor if it has an empty slot; otherwise
    # search Floors 1, 2, and 3 in order.
    recommendation_zone = None
    recommendation = None
    search_order = [current_zone] + [z for z in (1, 2, 3) if z != current_zone]
    for z in search_order:
        if stats[z]["ids"]:
            recommendation_zone = z
            recommendation = stats[z]["ids"][0]
            break
    recommendation_section = ZONES[recommendation_zone]["section"] if recommendation_zone else None

    # ========================================================
    # HEADER
    # ========================================================
    cv2.rectangle(img, (0, 0), (W, 92), (11, 20, 31), -1)
    cv2.line(img, (34, 91), (W - 34, 91), BORDER, 1)

    # Logo mark
    cv2.circle(img, (62, 44), 27, (35, 75, 110), -1)
    cv2.circle(img, (62, 44), 21, CYAN, 2)
    txt("P", 53, 53, 0.75, WHITE, 2)

    txt("SMART MALL", 103, 39, 0.92, WHITE, 2)
    txt("PARKING INTELLIGENCE PLATFORM", 105, 66, 0.43, CYAN, 1)

    # Live indicator
    cv2.circle(img, (1185, 37), 6, GREEN, -1)
    txt("SYSTEM LIVE", 1200, 43, 0.48, GREEN, 2)
    txt(time.strftime("%d %b %Y   %I:%M:%S %p"), 1045, 69, 0.38, MUTED, 1)

    # ========================================================
    # TOP KPI CARDS
    # ========================================================
    cards = [
        ("TOTAL CAPACITY", total_all, WHITE, "ALL FLOORS"),
        ("AVAILABLE NOW", available_all, GREEN, "READY TO PARK"),
        ("OCCUPIED", occupied_all, RED, f"{occupancy_pct:.0f}% UTILIZATION"),
        ("VEHICLES MOVING", moving_all, ORANGE, "IN TRANSIT")
    ]

    card_x = [40, 385, 730, 1075]
    for i, (label, value, color, sub) in enumerate(cards):
        x = card_x[i]
        rounded_box(x, 115, x + 320, 228, PANEL, BORDER, 14)
        txt(label, x + 22, 146, 0.42, MUTED, 1)
        txt(value, x + 22, 195, 1.35, color, 3)
        txt(sub, x + 22, 216, 0.35, MUTED, 1)
        cv2.rectangle(img, (x + 275, 135), (x + 298, 158), color, -1)

    # ========================================================
    # FLOOR OPERATIONS
    # ========================================================
    rounded_box(40, 250, 930, 640, PANEL, BORDER, 16)
    txt("FLOOR OPERATIONS", 66, 286, 0.62, WHITE, 2)
    txt("REAL-TIME AVAILABILITY BY PARKING LEVEL", 66, 312, 0.36, MUTED, 1)

    floor_colors = {1: CYAN, 2: BLUE, 3: GOLD}
    y = 345
    for z in (1, 2, 3):
        d = stats[z]
        selected = z == current_zone
        accent = floor_colors[z]
        bg = (25, 38, 52) if selected else PANEL_2

        rounded_box(62, y, 908, y + 82, bg, accent if selected else BORDER, 12, 2 if selected else 1)
        txt(f"FLOOR {z}", 84, y + 32, 0.58, WHITE, 2)
        txt(f"SECTION {ZONES[z]['section']}", 84, y + 57, 0.35, MUTED, 1)

        txt("AVAILABLE", 265, y + 27, 0.34, MUTED, 1)
        txt(d["available"], 265, y + 58, 0.80, GREEN, 2)

        txt("OCCUPIED", 405, y + 27, 0.34, MUTED, 1)
        txt(d["occupied"], 405, y + 58, 0.80, RED, 2)

        txt("MOVING", 535, y + 27, 0.34, MUTED, 1)
        txt(d["moving"], 535, y + 58, 0.80, ORANGE, 2)

        txt("CAPACITY", 650, y + 27, 0.34, MUTED, 1)
        txt(d["total"], 650, y + 58, 0.80, WHITE, 2)

        ratio = d["occupied"] / d["total"] if d["total"] else 0
        bar_x, bar_y, bar_w = 755, y + 31, 125
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (35, 46, 58), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y + 12), RED, -1)
        txt(f"{ratio * 100:.0f}%", 810, y + 63, 0.35, MUTED, 1)
        y += 96

    # ========================================================
    # CURRENT FLOOR / AVAILABILITY MAP
    # ========================================================
    rounded_box(995, 250, 1400, 640, PANEL, BORDER, 16)
    txt(f"FLOOR {current_zone} · SECTION {current_cfg['section']}", 1022, 286, 0.60, WHITE, 2)
    txt("CURRENT PARKING INVENTORY", 1022, 312, 0.35, MUTED, 1)

    if current["ids"]:
        txt("AVAILABLE SPACES", 1022, 350, 0.42, GREEN, 2)
        ids = current["ids"][:12]
        for i, sid in enumerate(ids):
            col = i % 3
            row = i // 3
            bx = 1022 + col * 118
            by = 370 + row * 48
            rounded_box(bx, by, bx + 105, by + 35, (12, 42, 29), (35, 120, 65), 8)
            txt(sid, bx + 10, by + 23, 0.38, GREEN, 2)
        if len(current["ids"]) > 12:
            txt(f"+ {len(current['ids']) - 12} more available", 1022, 585, 0.35, MUTED, 1)
    else:
        txt("PARKING FULL", 1022, 390, 0.75, RED, 2)
        txt("No currently available space", 1022, 420, 0.38, MUTED, 1)

    # ========================================================
    # RECOMMENDATION CARD
    # ========================================================
    rounded_box(40, 665, 1400, 855, (12, 35, 23), GREEN, 16, 2)
    txt("SMART PARKING RECOMMENDATION", 68, 704, 0.52, WHITE, 2)
    cv2.circle(img, (83, 758), 25, (30, 95, 45), -1)
    txt("P", 74, 768, 0.65, GREEN, 2)

    if recommendation:
        txt("BEST AVAILABLE SPACE", 130, 742, 0.35, MUTED, 1)
        txt(recommendation, 130, 785, 1.15, GREEN, 3)
        txt(f"FLOOR {recommendation_zone}  ·  SECTION {recommendation_section}", 130, 813, 0.42, WHITE, 1)
        txt("Proceed to the selected floor and follow mall directional signage.", 130, 837, 0.34, MUTED, 1)
        rounded_box(1050, 715, 1368, 815, (20, 78, 38), GREEN, 12, 1)
        txt("AVAILABLE", 1155, 757, 0.48, GREEN, 2)
        txt("NOW", 1188, 785, 0.36, WHITE, 1)
    else:
        txt("NO SPACE CURRENTLY AVAILABLE", 130, 770, 0.70, RED, 2)
        txt("Please check another floor.", 130, 808, 0.40, MUTED, 1)

    # ========================================================
    # FOOTER / CONTROLS
    # ========================================================
    txt("1", 45, 885, 0.42, CYAN, 2)
    txt("FLOOR 1", 64, 885, 0.34, MUTED, 1)
    txt("2", 165, 885, 0.42, BLUE, 2)
    txt("FLOOR 2", 184, 885, 0.34, MUTED, 1)
    txt("3", 285, 885, 0.42, GOLD, 2)
    txt("FLOOR 3", 304, 885, 0.34, MUTED, 1)
    txt("V", 430, 885, 0.42, CYAN, 2)
    txt("VIDEO", 449, 885, 0.34, MUTED, 1)
    txt("Q", 555, 885, 0.42, RED, 2)
    txt("EXIT", 575, 885, 0.34, MUTED, 1)
    txt("AI OCCUPANCY MONITORING  ·  YOLO + TRACKING", 1010, 885, 0.32, MUTED, 1)

    return img


# ============================================================
# OPEN ZONE
# ============================================================

def open_zone(zone_number):

    config = ZONES[zone_number]

    cap = cv2.VideoCapture(config["video"])

    if not cap.isOpened():
        raise RuntimeError(
            f"\nCannot open video: {config['video']}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    spaces = load_parking_spaces(
        config["json"],
        width,
        height
    )

    states = {}

    for space in spaces:

        states[space["id"]] = {
            "status": "EMPTY",
            "vehicle_id": None,
            "inside_since": None,
            "empty_since": None
        }

    zone_data[zone_number] = {
        "cap": cap,
        "spaces": spaces,
        "states": states,
        "fps": fps,
        "width": width,
        "height": height
    }

    print("\n----------------------------------------")
    print(
        f"ZONE {zone_number} / FLOOR {config['floor']}"
    )
    print(
        f"Video: {config['video']}"
    )
    print(
        f"Resolution: {width} x {height}"
    )
    print(
        f"Parking spaces: {len(spaces)}"
    )
    print("----------------------------------------")

    return cap


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(zone, frame, model):

    config = ZONES[zone]

    data = zone_data[zone]

    spaces = data["spaces"]

    # --------------------------------------------------------
    # YOLO + ByteTrack
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONFIDENCE,
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    detections = []

    if results and results[0].boxes is not None:

        boxes = results[0].boxes

        for i in range(len(boxes)):

            try:

                xyxy = boxes.xyxy[i].cpu().numpy()

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                if boxes.id is not None:
                    track_id = int(
                        boxes.id[i]
                    )
                else:
                    track_id = i

                cx = int(
                    (x1 + x2) / 2
                )

                cy = int(
                    (y1 + y2) / 2
                )

                movement = get_movement(
                    zone,
                    track_id,
                    cx,
                    cy
                )

                detections.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "movement": movement
                })

            except Exception:
                continue

    # --------------------------------------------------------
    # Update parking states
    # --------------------------------------------------------

    update_parking(
        zone,
        spaces,
        detections
    )

    # --------------------------------------------------------
    # Draw ONLY occupied pointer
    # --------------------------------------------------------

    for space in spaces:

        draw_pointer(
            frame,
            space,
            data["states"][space["id"]]["status"]
        )

    # --------------------------------------------------------
    # Zone key help
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "1:ZONE1  2:ZONE2  3:ZONE3  Q:EXIT",
        (15, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SMART MALL PARKING SYSTEM")
    print("=" * 60)
    print("\nLoading YOLO model...")

    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully.")

    # Load all three parking floors.
    for zone in [1, 2, 3]:
        open_zone(zone)

    video_window = "SMART MALL - LIVE PARKING VIDEO"
    dashboard_window = "SMART MALL - COMMAND CENTER"

    cv2.namedWindow(video_window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(video_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.namedWindow(dashboard_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(dashboard_window, 1440, 900)
    cv2.moveWindow(dashboard_window, 0, 0)

    current_zone = 1

    for zone in [1, 2, 3]:
        zone_data[zone]["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("\n============================================================")
    print("COMMAND CENTER READY")
    print("============================================================")
    print("1 = Floor 1   2 = Floor 2   3 = Floor 3")
    print("V = Focus video   D = Focus dashboard   Q = Exit")
    print("============================================================\n")

    # Cache the latest annotated frame for each floor. Every loop reads and
    # processes ALL THREE videos, so the dashboard reflects current occupancy
    # on Floors 1, 2, and 3 even when another floor is selected for viewing.
    latest_frames = {1: None, 2: None, 3: None}

    while True:

        for zone in (1, 2, 3):
            data = zone_data[zone]
            cap = data["cap"]
            ret, frame = cap.read()

            if not ret:
                # Loop recorded test videos. For physical CCTV, replace this
                # with the appropriate camera source and reconnect handling.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                vehicle_history[zone].clear()
                ret, frame = cap.read()

            if not ret:
                print(f"Cannot read video for Floor {zone}: {ZONES[zone]['video']}")
                continue

            latest_frames[zone] = process_frame(zone, frame, model)

        # Show the selected floor's latest processed frame.
        if latest_frames[current_zone] is not None:
            cv2.imshow(video_window, latest_frames[current_zone])

        # Dashboard aggregates the latest state of all three floors.
        dashboard = create_dashboard(current_zone)
        cv2.imshow(dashboard_window, dashboard)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            current_zone = 1
            print("Switched to Floor 1 / Section A")
        elif key == ord("2"):
            current_zone = 2
            print("Switched to Floor 2 / Section B")
        elif key == ord("3"):
            current_zone = 3
            print("Switched to Floor 3 / Section C")
        elif key == ord("v"):
            cv2.setWindowProperty(video_window, cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty(video_window, cv2.WND_PROP_TOPMOST, 0)
        elif key == ord("d"):
            cv2.setWindowProperty(dashboard_window, cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty(dashboard_window, cv2.WND_PROP_TOPMOST, 0)
        elif key == ord("q") or key == 27:
            break

    for zone in zone_data:
        try:
            zone_data[zone]["cap"].release()
        except Exception:
            pass

    cv2.destroyAllWindows()
    print("\nSMART MALL PARKING SYSTEM STOPPED.")


if __name__ == "__main__":
    main()
