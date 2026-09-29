from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt


IMAGE_DIR = Path("data/train/images")
LABEL_DIR = Path("data/train/labels")

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


images = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))

print(f"Images found: {len(images)}")

if not images:
    raise RuntimeError("No images found in data/train/images")


samples = random.sample(
    images,
    min(9, len(images))
)


for image_path in samples:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width = image.shape[:2]

    if label_path.exists():

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            values = line.strip().split()

            if len(values) != 5:
                continue

            cls, x, y, w, h = map(float, values)

            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)

            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)

            class_name = CLASS_NAMES[int(cls)]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )

            cv2.putText(
                image,
                class_name,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

    plt.figure(figsize=(12, 8))

    plt.imshow(image)

    plt.title(image_path.name)

    plt.axis("off")

    plt.show()