from pathlib import Path
from collections import defaultdict
import random
import shutil
import xml.etree.ElementTree as ET

from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT_DIR / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
ANNOTATIONS_DIR = RAW_DIR / "Annotations"

TRAIN_DIR = ROOT_DIR / "data" / "train"
VAL_DIR = ROOT_DIR / "data" / "val"
TEST_DIR = ROOT_DIR / "data" / "test"


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# DIRECTORY FUNCTIONS
# ============================================================

def create_directories():
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (split_dir / "images").mkdir(
            parents=True,
            exist_ok=True
        )

        (split_dir / "labels").mkdir(
            parents=True,
            exist_ok=True
        )


def clear_directory(directory):
    if directory.exists():
        for item in directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def clear_output_directories():
    print("Cleaning previous dataset splits...")

    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        clear_directory(split_dir)

    create_directories()


# ============================================================
# IMAGE SEARCH
# ============================================================

def build_image_index():
    """
    Creates:

        filename -> full image path

    This allows XML annotations to find their corresponding
    image even if images are inside class subdirectories.
    """

    print("\nBuilding image index...")

    image_index = {}

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".JPG",
        ".JPEG",
        ".PNG",
    }

    for image_path in IMAGES_DIR.rglob("*"):
        if image_path.is_file() and image_path.suffix in extensions:
            image_index[image_path.name] = image_path

    print(f"Images found: {len(image_index)}")

    return image_index


# ============================================================
# XML PARSING
# ============================================================

def parse_xml(xml_path):
    """
    Reads Pascal VOC XML and returns:

        filename
        image width
        image height
        objects
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find("filename")
    size_node = root.find("size")

    if filename_node is None:
        raise ValueError(
            f"Missing <filename>: {xml_path}"
        )

    if size_node is None:
        raise ValueError(
            f"Missing <size>: {xml_path}"
        )

    filename = filename_node.text.strip()

    width = int(
        size_node.find("width").text
    )

    height = int(
        size_node.find("height").text
    )

    objects = []

    for object_node in root.findall("object"):

        name_node = object_node.find("name")
        bbox_node = object_node.find("bndbox")

        if name_node is None or bbox_node is None:
            continue

        class_name = name_node.text.strip().lower()

        if class_name not in CLASS_TO_ID:
            print(
                f"WARNING: Unknown class "
                f"'{class_name}' in {xml_path}"
            )
            continue

        xmin = float(
            bbox_node.find("xmin").text
        )

        ymin = float(
            bbox_node.find("ymin").text
        )

        xmax = float(
            bbox_node.find("xmax").text
        )

        ymax = float(
            bbox_node.find("ymax").text
        )

        # ----------------------------------------------------
        # Clamp coordinates
        # ----------------------------------------------------

        xmin = max(0, min(xmin, width))
        ymin = max(0, min(ymin, height))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))

        # ----------------------------------------------------
        # Validate bounding box
        # ----------------------------------------------------

        if xmax <= xmin:
            continue

        if ymax <= ymin:
            continue

        objects.append(
            {
                "class_id": CLASS_TO_ID[class_name],
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


# ============================================================
# VOC -> YOLO CONVERSION
# ============================================================

def convert_bbox_to_yolo(
    xmin,
    ymin,
    xmax,
    ymax,
    image_width,
    image_height,
):
    """
    Pascal VOC:

        xmin ymin xmax ymax

    to YOLO:

        class_id x_center y_center width height

    All coordinates are normalized to [0, 1].
    """

    box_width = xmax - xmin
    box_height = ymax - ymin

    x_center = xmin + box_width / 2
    y_center = ymin + box_height / 2

    x_center /= image_width
    y_center /= image_height

    box_width /= image_width
    box_height /= image_height

    return (
        x_center,
        y_center,
        box_width,
        box_height,
    )


def create_yolo_label(annotation):
    lines = []

    for obj in annotation["objects"]:

        (
            x_center,
            y_center,
            box_width,
            box_height,
        ) = convert_bbox_to_yolo(
            obj["xmin"],
            obj["ymin"],
            obj["xmax"],
            obj["ymax"],
            annotation["width"],
            annotation["height"],
        )

        line = (
            f"{obj['class_id']} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        lines.append(line)

    return "\n".join(lines)


# ============================================================
# BOARD ID
# ============================================================

def get_board_id(filename):
    """
    Example:

        01_missing_hole_01.jpg
        -> 01

        12_spur_10.jpg
        -> 12
    """

    return filename.split("_")[0]


# ============================================================
# COLLECT ANNOTATIONS
# ============================================================

def collect_annotations(image_index):

    print("\nReading XML annotations...")

    annotations = []

    missing_images = []
    invalid_annotations = []

    xml_files = list(
        ANNOTATIONS_DIR.rglob("*.xml")
    )

    print(
        f"XML files found: {len(xml_files)}"
    )

    for xml_path in xml_files:

        try:
            annotation = parse_xml(xml_path)

        except Exception as error:
            invalid_annotations.append(
                (xml_path, str(error))
            )
            continue

        filename = annotation["filename"]

        if filename not in image_index:
            missing_images.append(
                (xml_path, filename)
            )
            continue

        if len(annotation["objects"]) == 0:
            continue

        annotation["image_path"] = image_index[
            filename
        ]

        annotation["xml_path"] = xml_path

        annotation["board_id"] = get_board_id(
            filename
        )

        annotations.append(annotation)

    print(
        f"Valid annotations: {len(annotations)}"
    )

    print(
        f"Missing images: {len(missing_images)}"
    )

    print(
        f"Invalid XML files: {len(invalid_annotations)}"
    )

    return annotations


# ============================================================
# GROUP BY BOARD
# ============================================================

def split_by_board(annotations):

    groups = defaultdict(list)

    for annotation in annotations:
        groups[
            annotation["board_id"]
        ].append(annotation)

    board_ids = list(groups.keys())

    random.seed(RANDOM_SEED)

    random.shuffle(board_ids)

    total = len(board_ids)

    train_count = round(
        total * TRAIN_RATIO
    )

    val_count = round(
        total * VAL_RATIO
    )

    # Make sure at least one board remains for test
    if train_count + val_count >= total:
        val_count = max(
            1,
            total - train_count - 1
        )

    train_boards = board_ids[
        :train_count
    ]

    val_boards = board_ids[
        train_count:
        train_count + val_count
    ]

    test_boards = board_ids[
        train_count + val_count:
    ]

    train_annotations = []
    val_annotations = []
    test_annotations = []

    for board_id in train_boards:
        train_annotations.extend(
            groups[board_id]
        )

    for board_id in val_boards:
        val_annotations.extend(
            groups[board_id]
        )

    for board_id in test_boards:
        test_annotations.extend(
            groups[board_id]
        )

    print("\nBoard split:")
    print("-" * 50)

    print(
        f"Train boards: {train_boards}"
    )

    print(
        f"Validation boards: {val_boards}"
    )

    print(
        f"Test boards: {test_boards}"
    )

    print("\nImage split:")
    print("-" * 50)

    print(
        f"Train: {len(train_annotations)}"
    )

    print(
        f"Validation: {len(val_annotations)}"
    )

    print(
        f"Test: {len(test_annotations)}"
    )

    return (
        train_annotations,
        val_annotations,
        test_annotations,
    )


# ============================================================
# COPY DATASET
# ============================================================

def copy_split(
    annotations,
    split_dir,
    split_name,
):

    print(
        f"\nPreparing {split_name} dataset..."
    )

    image_output_dir = (
        split_dir / "images"
    )

    label_output_dir = (
        split_dir / "labels"
    )

    class_counter = defaultdict(int)

    for index, annotation in enumerate(
        annotations,
        start=1,
    ):

        source_image = annotation[
            "image_path"
        ]

        image_name = source_image.name

        destination_image = (
            image_output_dir / image_name
        )

        destination_label = (
            label_output_dir
            / f"{source_image.stem}.txt"
        )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(
            source_image,
            destination_image,
        )

        # ----------------------------------------------------
        # Create YOLO label
        # ----------------------------------------------------

        yolo_label = create_yolo_label(
            annotation
        )

        destination_label.write_text(
            yolo_label + "\n",
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Count classes
        # ----------------------------------------------------

        for obj in annotation["objects"]:
            class_counter[
                obj["class_name"]
            ] += 1

        if index % 100 == 0:
            print(
                f"Processed: {index}/"
                f"{len(annotations)}"
            )

    print(
        f"{split_name} completed."
    )

    return class_counter


# ============================================================
# DATASET YAML
# ============================================================

def create_dataset_yaml():

    yaml_content = """path: data

train: train/images
val: val/images
test: test/images

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
"""

    yaml_path = (
        ROOT_DIR / "dataset.yaml"
    )

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
    )

    print(
        f"\ndataset.yaml created: {yaml_path}"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(
    train_count,
    val_count,
    test_count,
):

    total = (
        train_count
        + val_count
        + test_count
    )

    print("\n")
    print("=" * 60)
    print("PCB DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print(
        f"Total images : {total}"
    )

    print(
        f"Train        : {train_count}"
    )

    print(
        f"Validation   : {val_count}"
    )

    print(
        f"Test         : {test_count}"
    )

    print("\nYOLO classes:")

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"{class_id}: {class_name}"
        )

    print("\nOutput structure:")
    print(
        "data/train/images"
    )
    print(
        "data/train/labels"
    )
    print(
        "data/val/images"
    )
    print(
        "data/val/labels"
    )
    print(
        "data/test/images"
    )
    print(
        "data/test/labels"
    )

    print("\nDataset is ready for YOLO training.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PCB DEFECT DATASET PREPARATION")
    print("=" * 60)

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images directory not found:\n"
            f"{IMAGES_DIR}"
        )

    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(
            f"Annotations directory not found:\n"
            f"{ANNOTATIONS_DIR}"
        )

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    clear_output_directories()

    # --------------------------------------------------------
    # Index images
    # --------------------------------------------------------

    image_index = build_image_index()

    # --------------------------------------------------------
    # Read XML annotations
    # --------------------------------------------------------

    annotations = collect_annotations(
        image_index
    )

    if len(annotations) == 0:
        raise RuntimeError(
            "No valid annotations found."
        )

    # --------------------------------------------------------
    # Split by PCB board
    # --------------------------------------------------------

    (
        train_annotations,
        val_annotations,
        test_annotations,
    ) = split_by_board(
        annotations
    )

    # --------------------------------------------------------
    # Copy train
    # --------------------------------------------------------

    copy_split(
        train_annotations,
        TRAIN_DIR,
        "TRAIN",
    )

    # --------------------------------------------------------
    # Copy validation
    # --------------------------------------------------------

    copy_split(
        val_annotations,
        VAL_DIR,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # Copy test
    # --------------------------------------------------------

    copy_split(
        test_annotations,
        TEST_DIR,
        "TEST",
    )

    # --------------------------------------------------------
    # Create dataset.yaml
    # --------------------------------------------------------

    create_dataset_yaml()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        len(train_annotations),
        len(val_annotations),
        len(test_annotations),
    )


if __name__ == "__main__":
    main()