from pathlib import Path

from ultralytics import YOLO

from config import (
    MODEL_NAME,
    DATASET_YAML,
    IMAGE_SIZE,
    EPOCHS,
    BATCH_SIZE,
    WORKERS,
    PATIENCE,
    MODEL_DIR,
    RANDOM_SEED,
)


def train_model():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PCB DEFECT DETECTION TRAINING")
    print("=" * 60)

    print(f"Model       : {MODEL_NAME}")
    print(f"Dataset     : {DATASET_YAML}")
    print(f"Image size  : {IMAGE_SIZE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch size  : {BATCH_SIZE}")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATASET_YAML),

        imgsz=IMAGE_SIZE,

        epochs=EPOCHS,

        batch=BATCH_SIZE,

        workers=WORKERS,

        patience=PATIENCE,

        seed=RANDOM_SEED,

        project=str(MODEL_DIR),

        name="pcb_defect_training",

        pretrained=True,

        optimizer="auto",

        verbose=True,

        plots=True,

        save=True,

        exist_ok=True,
    )

    print("\nTraining completed.")

    print(f"Results saved to: {MODEL_DIR}")

    return results


if __name__ == "__main__":
    train_model()