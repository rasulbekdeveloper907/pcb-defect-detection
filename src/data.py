from pathlib import Path
from collections import Counter

from PIL import Image

from config import (
    RAW_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    CLASS_NAMES,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def find_images(directory: Path):
    images = []

    if not directory.exists():
        return images

    for extension in IMAGE_EXTENSIONS:
        images.extend(directory.rglob(f"*{extension}"))

    return images


def check_images(directory: Path):
    images = find_images(directory)

    valid_images = []
    corrupted_images = []

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()

            valid_images.append(image_path)

        except Exception:
            corrupted_images.append(image_path)

    return valid_images, corrupted_images


def print_dataset_summary():
    print("=" * 60)
    print("PCB DATASET SUMMARY")
    print("=" * 60)

    directories = {
        "RAW": RAW_DIR,
        "TRAIN": TRAIN_DIR,
        "VALIDATION": VAL_DIR,
        "TEST": TEST_DIR,
    }

    for name, directory in directories.items():

        images = find_images(directory)

        print(f"\n{name}")
        print("-" * 40)
        print(f"Directory : {directory}")
        print(f"Images    : {len(images)}")


def check_dataset():
    print_dataset_summary()

    print("\nChecking image integrity...")

    for name, directory in {
        "RAW": RAW_DIR,
        "TRAIN": TRAIN_DIR,
        "VALIDATION": VAL_DIR,
        "TEST": TEST_DIR,
    }.items():

        valid, corrupted = check_images(directory)

        print(
            f"{name}: "
            f"valid={len(valid)}, "
            f"corrupted={len(corrupted)}"
        )

        if corrupted:
            print("Corrupted files:")

            for path in corrupted[:10]:
                print(path)


if __name__ == "__main__":
    check_dataset()