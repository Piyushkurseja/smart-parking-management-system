import cv2
import json
import os

# =========================================================
# VIDEO CONFIGURATION
# =========================================================

VIDEOS = [
    {
        "path": "1.mp4",
        "floor": "1",
        "section": "A",
        "start_number": 101
    },
    {
        "path": "2.mp4",
        "floor": "2",
        "section": "B",
        "start_number": 201
    },
    {
        "path": "3.mp4",
        "floor": "3",
        "section": "C",
        "start_number": 301
    }
]

OUTPUT_FILE = "parking_spaces.json"

# =========================================================
# GLOBAL VARIABLES
# =========================================================

current_video_index = 0
frame = None

parking_spaces = {}

current_points = []
current_spaces = []


# =========================================================
# CREATE SPACE ID
# =========================================================

def get_space_id(config, index):

    number = config["start_number"] + index

    return f'{config["floor"]}{config["section"]}-{number}'


# =========================================================
# MOUSE CALLBACK
# =========================================================

def mouse_callback(event, x, y, flags, param):

    global current_points
    global current_spaces

    if event == cv2.EVENT_LBUTTONDOWN:

        current_points.append([x, y])

        print(f"Point selected: {x}, {y}")

        # Four points = one parking space
        if len(current_points) == 4:

            space_index = len(current_spaces)

            config = VIDEOS[current_video_index]

            space_id = get_space_id(
                config,
                space_index
            )

            space = {
                "id": space_id,
                "points": current_points.copy()
            }

            current_spaces.append(space)

            print(
                f"Parking space created: {space_id}"
            )

            current_points = []


# =========================================================
# DRAW SPACES
# =========================================================

def draw_spaces(image):

    # Draw completed spaces
    for space in current_spaces:

        points = space["points"]

        polygon = []

        for p in points:
            polygon.append((p[0], p[1]))

        polygon = cv2.convexHull(
            __import__("numpy").array(
                polygon,
                dtype=__import__("numpy").int32
            )
        )

        cv2.polylines(
            image,
            [polygon],
            True,
            (0, 255, 0),
            2
        )

        x = points[0][0]
        y = points[0][1]

        cv2.putText(
            image,
            space["id"],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Draw points currently being selected
    for p in current_points:

        cv2.circle(
            image,
            (p[0], p[1]),
            5,
            (0, 0, 255),
            -1
        )


# =========================================================
# SAVE CONFIGURATION
# =========================================================

def save_configuration():

    global parking_spaces

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        json.dump(
            parking_spaces,
            file,
            indent=4
        )

    print()
    print("====================================")
    print("PARKING SPACES SAVED SUCCESSFULLY")
    print(f"File: {OUTPUT_FILE}")
    print("====================================")


# =========================================================
# PROCESS VIDEO SETUP
# =========================================================

def setup_video(video_index):

    global frame
    global current_points
    global current_spaces

    current_points = []

    config = VIDEOS[video_index]

    video_path = config["path"]

    if not os.path.exists(video_path):

        print()
        print(f"ERROR: Video not found: {video_path}")

        return False

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            f"ERROR: Cannot open {video_path}"
        )

        return False

    success, frame = cap.read()

    cap.release()

    if not success:

        print(
            f"ERROR: Cannot read {video_path}"
        )

        return False

    # Load existing spaces if available
    current_spaces = parking_spaces.get(
        video_path,
        []
    )

    window_name = (
        f"SETUP - FLOOR {config['floor']} "
        f"SECTION {config['section']}"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        1400,
        800
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    print()
    print("====================================")
    print(
        f"VIDEO: {video_path}"
    )
    print(
        f"FLOOR: {config['floor']}"
    )
    print(
        f"SECTION: {config['section']}"
    )
    print("====================================")
    print()
    print(
        "INSTRUCTIONS:"
    )
    print(
        "Click 4 corners of each parking space"
    )
    print(
        "R = Remove last space"
    )
    print(
        "N = Next video"
    )
    print(
        "S = Save"
    )
    print(
        "ESC = Exit"
    )

    while True:

        display = frame.copy()

        draw_spaces(display)

        cv2.putText(
            display,
            f"Floor {config['floor']} | "
            f"Section {config['section']} | "
            f"Spaces: {len(current_spaces)}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        cv2.imshow(
            window_name,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        # Save
        if key == ord("s"):

            parking_spaces[video_path] = (
                current_spaces
            )

            save_configuration()

        # Remove last
        elif key == ord("r"):

            if len(current_points) > 0:

                current_points.pop()

            elif len(current_spaces) > 0:

                removed = current_spaces.pop()

                print(
                    f"Removed: {removed['id']}"
                )

        # Next video
        elif key == ord("n"):

            parking_spaces[video_path] = (
                current_spaces
            )

            cv2.destroyWindow(
                window_name
            )

            return True

        # Exit
        elif key == 27:

            parking_spaces[video_path] = (
                current_spaces
            )

            save_configuration()

            cv2.destroyAllWindows()

            return False


# =========================================================
# MAIN
# =========================================================

print()
print("==============================================")
print("SMART MULTI-FLOOR PARKING SPACE SETUP")
print("==============================================")

# Load existing file if available
if os.path.exists(OUTPUT_FILE):

    try:

        with open(
            OUTPUT_FILE,
            "r"
        ) as file:

            parking_spaces = json.load(file)

    except:

        parking_spaces = {}


# Setup each video
for index in range(len(VIDEOS)):

    continue_program = setup_video(index)

    if not continue_program:

        break


save_configuration()

print()
print("SETUP COMPLETED")
print()

cv2.destroyAllWindows()