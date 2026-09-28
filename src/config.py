from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUTS_DIR = ROOT_DIR / "outputs"

PLOTS_DIR = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"


# ============================================================
# MODEL
# ============================================================

MODEL_DIR = ROOT_DIR / "models"

MODEL_NAME = "yolo11n.pt"

BEST_MODEL = MODEL_DIR / "best.pt"


# ============================================================
# DATASET
# ============================================================

DATASET_YAML = ROOT_DIR / "dataset.yaml"


# ============================================================
# PCB CLASSES
# ============================================================

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# TRAINING CONFIG
# ============================================================

IMAGE_SIZE = 640

EPOCHS = 50

BATCH_SIZE = 16

WORKERS = 4

PATIENCE = 10

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DIRECTORIES = [
    RAW_DIR,
    PROCESSED_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    METRICS_DIR,
]


def create_directories():
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print("Project directories created successfully.")