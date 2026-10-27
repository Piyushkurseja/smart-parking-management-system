import cv2
import numpy as np
import json
import os
import math
import torch
from ultralytics import YOLO
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"

VIDEOS = [
    {
        "path": "1.mp4",
        "floor": "1",
        "section": "A",
        "prefix": "1A"
    },
    {
        "path": "2.mp4",
        "floor": "2",
        "section": "B",
        "prefix": "2B"
    },
    {
        "path": "3.mp4",
        "floor": "3",
        "section": "C",
        "prefix": "3C"
    }
]

OUTPUT_FILE = "parking_spaces.json"

CONFIDENCE = 0.25

# Number of frames sampled during automatic setup
SAMPLE_FRAMES = 80

# Vehicle classes in COCO
VEHICLE_CLASSES = [2, 3, 5, 7]

# A vehicle is considered stationary if
# its center moves less than this distance
STATIONARY_THRESHOLD = 20

# Minimum observations needed for a parking location
MIN_OBSERVATIONS = 3


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = 0
    print("GPU detected.")

else:

    DEVICE = "cpu"
    print("No GPU detected - using CPU.")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded.")
print()


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# CLUSTER POINTS
# ============================================================

def cluster_points(points, threshold):

    if len(points) == 0:
        return []

    clusters = []

    for point in points:

        assigned = False

        for cluster in clusters:

            center = np.mean(
                cluster,
                axis=0
            )

            if distance(
                point,
                center
            ) < threshold:

                cluster.append(point)

                assigned = True

                break

        if not assigned:

            clusters.append(
                [point]
            )

    # Repeat clustering until stable
    changed = True

    while changed:

        changed = False

        new_clusters = []

        while clusters:

            current = clusters.pop(0)

            current_center = np.mean(
                current,
                axis=0
            )

            merged = False

            for other in new_clusters:

                other_center = np.mean(
                    other,
                    axis=0
                )

                if distance(
                    current_center,
                    other_center
                ) < threshold:

                    other.extend(current)

                    merged = True

                    changed = True

                    break

            if not merged:

                new_clusters.append(
                    current
                )

        clusters = new_clusters

    return clusters


# ============================================================
# DETECT VEHICLES FROM FRAME
# ============================================================

def detect(frame):

    results = model.predict(

        frame,

        conf=CONFIDENCE,

        classes=VEHICLE_CLASSES,

        device=DEVICE,

        verbose=False

    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = xyxy

            width = x2 - x1
            height = y2 - y1

            if width < 20 or height < 20:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height
                }
            )

    return detections


# ============================================================
# AUTOMATIC SPACE DISCOVERY
# ============================================================

def discover_spaces(video_info):

    path = video_info["path"]

    print()
    print("================================================")
    print(f"AUTOMATIC SETUP: {path}")
    print(
        f"Floor {video_info['floor']} "
        f"| Section {video_info['section']}"
    )
    print("================================================")

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():

        print(
            f"ERROR: Cannot open {path}"
        )

        return []

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    # --------------------------------------------------------
    # Select frames throughout complete video
    # --------------------------------------------------------

    if total_frames <= SAMPLE_FRAMES:

        frame_numbers = list(
            range(total_frames)
        )

    else:

        frame_numbers = np.linspace(
            0,
            total_frames - 1,
            SAMPLE_FRAMES
        ).astype(int)

    # --------------------------------------------------------
    # Collect vehicle detections
    # --------------------------------------------------------

    all_detections = []

    for i, frame_number in enumerate(
        frame_numbers
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_number)
        )

        success, frame = cap.read()

        if not success:
            continue

        detections = detect(frame)

        for d in detections:

            d["frame"] = frame_number

            all_detections.append(d)

        print(
            f"\rAnalysing "
            f"{i + 1}/{len(frame_numbers)} frames...",
            end=""
        )

    cap.release()

    print()

    if len(all_detections) == 0:

        print(
            "No vehicles detected."
        )

        return []

    print(
        f"Vehicle detections collected: "
        f"{len(all_detections)}"
    )

    # --------------------------------------------------------
    # Remove detections near image borders
    # --------------------------------------------------------

    valid = []

    for d in all_detections:

        if (
            d["cx"] < 10 or
            d["cy"] < 10
        ):
            continue

        valid.append(d)

    all_detections = valid

    # --------------------------------------------------------
    # Create candidate stationary positions
    #
    # We compare detections close to each other spatially.
    # Repeated detections at almost the same position indicate
    # a parked vehicle.
    # --------------------------------------------------------

    points = []

    widths = []

    heights = []

    for d in all_detections:

        points.append(
            (
                d["cx"],
                d["cy"]
            )
        )

        widths.append(
            d["width"]
        )

        heights.append(
            d["height"]
        )

    # --------------------------------------------------------
    # Estimate average vehicle size
    # --------------------------------------------------------

    median_width = int(
        np.median(widths)
    )

    median_height = int(
        np.median(heights)
    )

    # Avoid excessively large slots
    median_width = max(
        35,
        min(median_width, 300)
    )

    median_height = max(
        35,
        min(median_height, 250)
    )

    print(
        f"Estimated vehicle size: "
        f"{median_width} x {median_height}"
    )

    # --------------------------------------------------------
    # Cluster vehicle positions
    # --------------------------------------------------------

    position_clusters = cluster_points(

        points,

        threshold=max(
            30,
            int(median_width * 0.55)
        )

    )

    # --------------------------------------------------------
    # Keep clusters containing enough detections
    # --------------------------------------------------------

    parking_centers = []

    for cluster in position_clusters:

        if len(cluster) < MIN_OBSERVATIONS:

            continue

        center = np.mean(
            cluster,
            axis=0
        )

        parking_centers.append(
            (
                int(center[0]),
                int(center[1])
            )
        )

    print(
        f"Candidate parking positions: "
        f"{len(parking_centers)}"
    )

    if len(parking_centers) == 0:

        print(
            "Could not discover parking spaces."
        )

        return []

    # --------------------------------------------------------
    # Automatically discover rows
    # --------------------------------------------------------

    row_threshold = max(
        40,
        int(median_height * 0.65)
    )

    rows = cluster_points(

        parking_centers,

        threshold=row_threshold

    )

    # --------------------------------------------------------
    # Create slots
    # --------------------------------------------------------

    spaces = []

    space_number = 1

    for row in rows:

        # Sort from left to right
        row.sort(
            key=lambda p: p[0]
        )

        for center in row:

            cx, cy = center

            half_w = int(
                median_width * 0.65
            )

            half_h = int(
                median_height * 0.70
            )

            x1 = max(
                0,
                cx - half_w
            )

            y1 = max(
                0,
                cy - half_h
            )

            x2 = cx + half_w

            y2 = cy + half_h

            slot_id = (

                f"{video_info['prefix']}-"
                f"{100 + space_number:03d}"

            )

            spaces.append(
                {
                    "id": slot_id,

                    "floor":
                        video_info["floor"],

                    "section":
                        video_info["section"],

                    "points":
                        [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]
                        ]
                }
            )

            space_number += 1

    # --------------------------------------------------------
    # Remove overlapping duplicate spaces
    # --------------------------------------------------------

    final_spaces = []

    for space in spaces:

        p = np.array(
            space["points"],
            dtype=np.float32
        )

        center_x = np.mean(
            p[:, 0]
        )

        center_y = np.mean(
            p[:, 1]
        )

        duplicate = False

        for existing in final_spaces:

            ep = np.array(
                existing["points"],
                dtype=np.float32
            )

            ex = np.mean(
                ep[:, 0]
            )

            ey = np.mean(
                ep[:, 1]
            )

            if distance(
                (center_x, center_y),
                (ex, ey)
            ) < median_width * 0.45:

                duplicate = True

                break

        if not duplicate:

            final_spaces.append(
                space
            )

    # --------------------------------------------------------
    # Renumber
    # --------------------------------------------------------

    for i, space in enumerate(
        final_spaces
    ):

        space["id"] = (

            f"{video_info['prefix']}-"
            f"{100 + i + 1:03d}"

        )

    print()
    print(
        f"Automatically discovered "
        f"{len(final_spaces)} parking spaces."
    )

    return final_spaces


# ============================================================
# SAVE VISUAL PREVIEW
# ============================================================

def create_preview(
    video_info,
    spaces
):

    cap = cv2.VideoCapture(
        video_info["path"]
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        return

    for space in spaces:

        points = np.array(
            space["points"],
            dtype=np.int32
        )

        cv2.polylines(
            frame,
            [points],
            True,
            (0, 255, 0),
            2
        )

        center_x = int(
            np.mean(points[:, 0])
        )

        center_y = int(
            np.mean(points[:, 1])
        )

        cv2.putText(
            frame,
            space["id"],
            (
                center_x - 50,
                center_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output_name = (

        f"outputs/"
        f"floor_{video_info['floor']}_"
        f"automatic_setup.jpg"

    )

    cv2.imwrite(
        output_name,
        frame
    )

    print(
        f"Preview saved: {output_name}"
    )


# ============================================================
# MAIN
# ============================================================

all_spaces = {}

print()
print("================================================")
print("AUTOMATIC MULTI-FLOOR PARKING SETUP")
print("================================================")
print()
print("NO MANUAL SPACE SELECTION")
print("NO FIXED NUMBER OF SPACES")
print()

for video_info in VIDEOS:

    path = video_info["path"]

    if not os.path.exists(path):

        print(
            f"WARNING: {path} not found."
        )

        continue

    spaces = discover_spaces(
        video_info
    )

    all_spaces[path] = spaces

    create_preview(
        video_info,
        spaces
    )


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_FILE,
    "w"
) as file:

    json.dump(
        all_spaces,
        file,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("================================================")
print("AUTOMATIC SETUP COMPLETED")
print("================================================")

total = 0

for video_info in VIDEOS:

    spaces = all_spaces.get(
        video_info["path"],
        []
    )

    print(
        f"Floor {video_info['floor']} "
        f"({video_info['path']}): "
        f"{len(spaces)} spaces"
    )

    total += len(spaces)

print()
print(
    f"TOTAL DISCOVERED SPACES: {total}"
)

print()
print(
    f"Configuration saved to: "
    f"{OUTPUT_FILE}"
)

print("================================================")