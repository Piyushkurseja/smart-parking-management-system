import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN


# ============================================================
# SMART MALL PARKING SYSTEM
# AUTOMATIC PARKING-SPACE DISCOVERY
# ============================================================
#
# IMPORTANT:
#
# ONLY CAR IS DETECTED
#
# YOLO COCO:
#     2 = car
#
# Ignored:
#     person
#     bicycle
#     motorcycle
#     bus
#     truck
#     road arrows
#     road markings
#     etc.
#
# NO FIXED NUMBER OF SPACES
#
# Videos:
#     1.mp4 -> parking_spaces_1.json
#     2.mp4 -> parking_spaces_2.json
#     3.mp4 -> parking_spaces_3.json
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolov8n.pt"


VIDEOS = {

    1: {
        "video": "1.mp4",
        "json": "parking_spaces_1.json",
        "floor": 1
    },

    2: {
        "video": "2.mp4",
        "json": "parking_spaces_2.json",
        "floor": 2
    },

    3: {
        "video": "3.mp4",
        "json": "parking_spaces_3.json",
        "floor": 3
    }
}


# ============================================================
# ONLY CAR
# ============================================================

CAR_CLASS = 2


# ============================================================
# YOLO
# ============================================================

CONFIDENCE = 0.30


# ============================================================
# VIDEO SAMPLING
# ============================================================

# Process approximately every Nth frame.

FRAME_STEP = 8


# Maximum number of frames used for setup.

MAX_SETUP_FRAMES = 800


# ============================================================
# PARKED-CAR FILTER
# ============================================================
#
# A parking space should normally contain a car repeatedly
# at approximately the same position.
#
# Cars moving through the driving lane will have different
# positions and therefore will not form strong clusters.
# ============================================================

MIN_OBSERVATIONS = 5


# ============================================================
# DBSCAN
# ============================================================
#
# Larger value joins nearby observations.
#
# Adjust only if necessary.
# ============================================================

CLUSTER_DISTANCE = 55


# ============================================================
# DUPLICATE SPACE DISTANCE
# ============================================================

DUPLICATE_DISTANCE = 45


# ============================================================
# PARKING BOX SCALE
# ============================================================
#
# The generated parking area is based on the median car size.
#
# Slightly enlarge it so a parked car fits inside.
# ============================================================

BOX_SCALE = 1.15


# ============================================================
# MINIMUM / MAXIMUM PARKING SIZE
# ============================================================

MIN_BOX_WIDTH = 30
MIN_BOX_HEIGHT = 20

MAX_BOX_WIDTH = 500
MAX_BOX_HEIGHT = 500


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "outputs"


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("SMART MALL PARKING SYSTEM")
print("=" * 60)
print()

print("Detection Mode : ONLY CAR")
print("YOLO Class     : 2")
print()

print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)

print("YOLO model loaded successfully.")


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(
        p1,
        p2
):

    return float(

        np.sqrt(

            (
                p1[0] - p2[0]
            ) ** 2

            +

            (
                p1[1] - p2[1]
            ) ** 2
        )
    )


# ============================================================
# DISCOVER PARKING SPACES
# ============================================================

def discover_spaces(
        video_info
):

    video_path = video_info[
        "video"
    ]

    floor = video_info[
        "floor"
    ]

    print()
    print("=" * 60)
    print(
        f"PROCESSING FLOOR {floor}"
    )
    print("=" * 60)
    print(
        f"Video: {video_path}"
    )


    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        print(
            f"ERROR: Cannot open {video_path}"
        )

        return []


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


    if fps <= 0:

        fps = 25


    total_frames = int(

        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"FPS: {fps:.1f}"
    )

    print(
        f"Total frames: {total_frames}"
    )


    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    observations = []

    vehicle_sizes = []


    frame_number = 0

    processed_frames = 0


    last_frame = None


    # ========================================================
    # READ VIDEO
    # ========================================================

    while True:

        ret, frame = cap.read()


        if not ret:

            break


        frame_number += 1


        # ----------------------------------------------------
        # FRAME SKIP
        # ----------------------------------------------------

        if (
            frame_number % FRAME_STEP
            != 0
        ):

            continue


        processed_frames += 1


        last_frame = frame.copy()


        # ----------------------------------------------------
        # MAX FRAMES
        # ----------------------------------------------------

        if (
            processed_frames
            >
            MAX_SETUP_FRAMES
        ):

            break


        # ====================================================
        # YOLO
        # ====================================================
        #
        # VERY IMPORTANT:
        #
        # classes=[2]
        #
        # ONLY CAR
        #
        # ====================================================

        results = model.predict(

            frame,

            conf=CONFIDENCE,

            classes=[CAR_CLASS],

            verbose=False
        )


        if (
            not results
            or results[0].boxes is None
        ):

            continue


        boxes = results[
            0
        ].boxes


        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        for i in range(
            len(boxes)
        ):

            try:

                # ------------------------------------------------
                # EXTRA CAR CHECK
                # ------------------------------------------------

                class_id = int(

                    boxes.cls[i]
                    .item()
                )


                if class_id != CAR_CLASS:

                    continue


                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                confidence = float(

                    boxes.conf[i]
                    .item()
                )


                if confidence < CONFIDENCE:

                    continue


                # ------------------------------------------------
                # BOX
                # ------------------------------------------------

                x1, y1, x2, y2 = map(

                    int,

                    boxes.xyxy[
                        i
                    ].cpu().numpy()
                )


                # ------------------------------------------------
                # VALID BOX
                # ------------------------------------------------

                box_width = x2 - x1

                box_height = y2 - y1


                if (
                    box_width <= 0
                    or box_height <= 0
                ):

                    continue


                # ------------------------------------------------
                # FILTER VERY SMALL / HUGE OBJECTS
                # ------------------------------------------------

                if box_width < 15:

                    continue

                if box_height < 10:

                    continue

                if box_width > width * 0.8:

                    continue

                if box_height > height * 0.8:

                    continue


                # ------------------------------------------------
                # CENTER
                # ------------------------------------------------

                center_x = int(

                    (x1 + x2)
                    /
                    2
                )


                center_y = int(

                    (y1 + y2)
                    /
                    2
                )


                # ------------------------------------------------
                # SAVE OBSERVATION
                # ------------------------------------------------

                observations.append(

                    [
                        center_x,
                        center_y
                    ]
                )


                vehicle_sizes.append(

                    [
                        box_width,
                        box_height
                    ]
                )


            except Exception:

                continue


    cap.release()


    # ========================================================
    # CHECK OBSERVATIONS
    # ========================================================

    print()

    print(
        "Car observations:",
        len(observations)
    )


    if len(observations) < MIN_OBSERVATIONS:

        print(
            "Not enough car observations."
        )

        return []


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    points = np.array(

        observations,

        dtype=np.float32
    )


    # ========================================================
    # DBSCAN
    # ========================================================

    print(
        "Finding repeated car positions..."
    )


    clustering = DBSCAN(

        eps=CLUSTER_DISTANCE,

        min_samples=MIN_OBSERVATIONS
    ).fit(
        points
    )


    labels = clustering.labels_


    # ========================================================
    # VALID CLUSTERS
    # ========================================================

    unique_labels = sorted(

        set(labels)
    )


    unique_labels = [

        label

        for label in unique_labels

        if label != -1
    ]


    print(
        "Repeated positions found:",
        len(unique_labels)
    )


    # ========================================================
    # MEDIAN CAR SIZE
    # ========================================================

    sizes = np.array(

        vehicle_sizes,

        dtype=np.float32
    )


    median_width = float(

        np.median(
            sizes[:, 0]
        )
    )


    median_height = float(

        np.median(
            sizes[:, 1]
        )
    )


    # ========================================================
    # LIMIT BOX SIZE
    # ========================================================

    box_width = int(

        median_width
        *
        BOX_SCALE
    )


    box_height = int(

        median_height
        *
        BOX_SCALE
    )


    box_width = max(

        MIN_BOX_WIDTH,

        min(
            box_width,
            MAX_BOX_WIDTH
        )
    )


    box_height = max(

        MIN_BOX_HEIGHT,

        min(
            box_height,
            MAX_BOX_HEIGHT
        )
    )


    print(
        f"Estimated parking size: "
        f"{box_width} x {box_height}"
    )


    # ========================================================
    # CREATE SPACES
    # ========================================================

    candidate_spaces = []


    for label in unique_labels:

        indices = np.where(

            labels == label
        )[0]


        if len(indices) < MIN_OBSERVATIONS:

            continue


        cluster_points = points[
            indices
        ]


        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        center_x = int(

            np.mean(
                cluster_points[:, 0]
            )
        )


        center_y = int(

            np.mean(
                cluster_points[:, 1]
            )
        )


        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        x = int(

            center_x
            -
            box_width / 2
        )


        y = int(

            center_y
            -
            box_height / 2
        )


        # ----------------------------------------------------
        # CLAMP TO VIDEO
        # ----------------------------------------------------

        x = max(
            0,
            min(
                x,
                width - box_width - 1
            )
        )


        y = max(
            0,
            min(
                y,
                height - box_height - 1
            )
        )


        # ----------------------------------------------------
        # POINTS
        # ----------------------------------------------------

        points_polygon = [

            [
                x,
                y
            ],

            [
                x + box_width,
                y
            ],

            [
                x + box_width,
                y + box_height
            ],

            [
                x,
                y + box_height
            ]

        ]


        candidate_spaces.append({

            "id": "",

            "points": points_polygon,

            "x": x,

            "y": y,

            "width": box_width,

            "height": box_height,

            "observations": len(
                indices
            )
        })


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    final_spaces = []


    for space in candidate_spaces:

        center_x = (

            space["x"]
            +
            space["width"] / 2
        )


        center_y = (

            space["y"]
            +
            space["height"] / 2
        )


        duplicate = False


        for existing in final_spaces:

            ex = (

                existing["x"]
                +
                existing["width"] / 2
            )


            ey = (

                existing["y"]
                +
                existing["height"] / 2
            )


            if distance(

                (
                    center_x,
                    center_y
                ),

                (
                    ex,
                    ey
                )

            ) < DUPLICATE_DISTANCE:

                duplicate = True

                break


        if not duplicate:

            final_spaces.append(
                space
            )


    # ========================================================
    # SORT
    # ========================================================

    final_spaces.sort(

        key=lambda s: (

            s["y"],

            s["x"]
        )
    )


    # ========================================================
    # ASSIGN IDS
    # ========================================================

    for index, space in enumerate(

        final_spaces,

        start=1
    ):

        space["id"] = (

            f"F{floor}-P{index:03d}"
        )


        # Remove internal setup information
        # from final JSON.

        space.pop(
            "observations",
            None
        )


    # ========================================================
    # SAVE JSON
    # ========================================================

    output_file = video_info[
        "json"
    ]


    with open(

        output_file,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            final_spaces,

            f,

            indent=4
        )


    # ========================================================
    # PREVIEW
    # ========================================================

    if last_frame is not None:

        preview = last_frame.copy()


        for space in final_spaces:

            pts = np.array(

                space["points"],

                dtype=np.int32
            )


            # ----------------------------------------------
            # Parking boundary
            # ----------------------------------------------

            cv2.polylines(

                preview,

                [pts],

                True,

                (0, 255, 0),

                2
            )


            # ----------------------------------------------
            # ID
            # ----------------------------------------------

            cx = int(

                np.mean(
                    pts[:, 0]
                )
            )


            cy = int(

                np.mean(
                    pts[:, 1]
                )
            )


            cv2.putText(

                preview,

                space["id"],

                (
                    cx - 30,
                    cy
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 255, 0),

                2,

                cv2.LINE_AA
            )


        os.makedirs(

            OUTPUT_DIR,

            exist_ok=True
        )


        preview_file = os.path.join(

            OUTPUT_DIR,

            f"floor_{floor}_parking_setup.jpg"
        )


        cv2.imwrite(

            preview_file,

            preview
        )


        print(
            f"Preview saved: {preview_file}"
        )


    # ========================================================
    # RESULT
    # ========================================================

    print()

    print(
        f"Floor {floor}: "
        f"{len(final_spaces)} parking spaces discovered."
    )


    print(
        f"Saved: {output_file}"
    )


    return final_spaces


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "Starting automatic "
        "three-floor parking setup..."
    )

    print()

    total_spaces = 0


    # ========================================================
    # PROCESS EACH VIDEO
    # ========================================================

    for floor, video_info in VIDEOS.items():

        video_file = video_info[
            "video"
        ]


        # ----------------------------------------------------
        # CHECK VIDEO
        # ----------------------------------------------------

        if not os.path.exists(
            video_file
        ):

            print()

            print(
                f"WARNING: {video_file} "
                f"not found."
            )

            print(
                "Skipping this floor."
            )

            continue


        # ----------------------------------------------------
        # DISCOVER
        # ----------------------------------------------------

        spaces = discover_spaces(

            video_info
        )


        total_spaces += len(
            spaces
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "AUTOMATIC PARKING SETUP COMPLETED"
    )

    print(
        "=" * 60
    )

    print()

    for floor, video_info in VIDEOS.items():

        json_file = video_info[
            "json"
        ]


        if os.path.exists(
            json_file
        ):

            try:

                with open(

                    json_file,

                    "r",

                    encoding="utf-8"

                ) as f:

                    spaces = json.load(
                        f
                    )


                print(

                    f"Floor {floor}: "
                    f"{len(spaces)} spaces"
                )


            except:

                print(

                    f"Floor {floor}: "
                    f"JSON error"
                )

        else:

            print(

                f"Floor {floor}: "
                f"not created"
            )


    print()

    print(
        f"TOTAL DISCOVERED SPACES: "
        f"{total_spaces}"
    )


    print()

    print(
        "Generated files:"
    )


    print(
        "  parking_spaces_1.json"
    )

    print(
        "  parking_spaces_2.json"
    )

    print(
        "  parking_spaces_3.json"
    )


    print()

    print(
        "You can now run:"
    )

    print(
        "python live_dector1.py"
    )

    print(
        "=" * 60
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()