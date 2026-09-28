import json

from ultralytics import YOLO

from config import (
    BEST_MODEL,
    DATASET_YAML,
    IMAGE_SIZE,
    BATCH_SIZE,
    METRICS_DIR,
)


def evaluate_model():

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not BEST_MODEL.exists():

        raise FileNotFoundError(
            f"Model not found: {BEST_MODEL}\n"
            "Train the model first."
        )

    print("=" * 60)
    print("PCB MODEL EVALUATION")
    print("=" * 60)

    model = YOLO(str(BEST_MODEL))

    metrics = model.val(
        data=str(DATASET_YAML),

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        split="test",

        plots=True,

        project=str(METRICS_DIR),

        name="evaluation",

        exist_ok=True,
    )

    results = {
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
    }

    output_file = METRICS_DIR / "metrics.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nEvaluation results:")
    print("-" * 40)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nMetrics saved to: {output_file}")

    return metrics


if __name__ == "__main__":
    evaluate_model()