import json
import os
import shutil

# -----------------------------
# SETTINGS
# -----------------------------

DATASET_DIR = "archive"
OUTPUT_DIR = "dataset"

# -----------------------------
# CONVERSION FUNCTION
# -----------------------------

def convert_split(split):

    print(f"\nProcessing {split} dataset...")

    image_dir = os.path.join(DATASET_DIR, split)
    json_file = os.path.join(image_dir, "_annotations.coco.json")

    output_images = os.path.join(
        OUTPUT_DIR, split, "images"
    )

    output_labels = os.path.join(
        OUTPUT_DIR, split, "labels"
    )

    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    # Read COCO JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    # Map image ID -> image information
    images = {
        image["id"]: image
        for image in data["images"]
    }

    # Create annotations grouped by image
    annotations = {}

    for ann in data["annotations"]:
        image_id = ann["image_id"]

        if image_id not in annotations:
            annotations[image_id] = []

        annotations[image_id].append(ann)

    # Process images
    for index, image_id in enumerate(images):

        image = images[image_id]

        file_name = image["file_name"]

        width = image["width"]
        height = image["height"]

        source_image = os.path.join(
            image_dir,
            file_name
        )

        destination_image = os.path.join(
            output_images,
            file_name
        )

        # Copy image
        if os.path.exists(source_image):

            shutil.copy2(
                source_image,
                destination_image
            )

        else:
            print(
                f"WARNING: Image not found: {file_name}"
            )

            continue

        # Label file
        label_name = os.path.splitext(
            file_name
        )[0] + ".txt"

        label_path = os.path.join(
            output_labels,
            label_name
        )

        with open(label_path, "w") as label_file:

            for ann in annotations.get(
                image_id, []
            ):

                category_id = ann["category_id"]

                # YOLO uses class IDs starting from 0
                # We will use:
                # 0 = space-empty
                # 1 = space-occupied

                if category_id == 0:
                    continue

                if category_id == 1:
                    yolo_class = 0

                elif category_id == 2:
                    yolo_class = 1

                else:
                    continue

                x, y, w, h = ann["bbox"]

                # Convert COCO -> YOLO
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height

                w_normalized = w / width
                h_normalized = h / height

                label_file.write(
                    f"{yolo_class} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{w_normalized:.6f} "
                    f"{h_normalized:.6f}\n"
                )

        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1}/{len(images)} images"
            )

    print(f"{split} conversion completed!")


# -----------------------------
# RUN CONVERSION
# -----------------------------

convert_split("train")
convert_split("valid")

print("\n================================")
print("CONVERSION COMPLETED")
print("================================")