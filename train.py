from ultralytics import YOLO
import os

# Load trained model
model = YOLO(
    "runs/detect/smart_parking_test/weights/best.pt"
)

# Validation images
image_folder = "dataset/valid/images"

images = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not images:
    print("No images found!")
    exit()

# Select first image
image_path = os.path.join(image_folder, images[0])

print("\nTesting image:", image_path)

# Run prediction
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

# Count classes
empty = 0
occupied = 0

for result in results:

    for box in result.boxes:

        class_id = int(box.cls[0])

        if class_id == 0:
            empty += 1

        elif class_id == 1:
            occupied += 1

total = empty + occupied

if total > 0:
    occupancy = (occupied / total) * 100
    availability = (empty / total) * 100
else:
    occupancy = 0
    availability = 0

print("\n================================")
print("       SMART PARKING STATUS")
print("================================")

print(f"Total Spaces : {total}")
print(f"Occupied     : {occupied}")
print(f"Available    : {empty}")
print(f"Occupancy    : {occupancy:.2f}%")
print(f"Availability : {availability:.2f}%")

print("================================")
print("Detection completed!")