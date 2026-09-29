from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from src.config import (
    BEST_MODEL,
    CONFIDENCE_THRESHOLD,
)


class PCBDetector:

    def __init__(
        self,
        model_path=BEST_MODEL,
        confidence=CONFIDENCE_THRESHOLD,
    ):

        model_path = Path(model_path)

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.model = YOLO(str(model_path))

        self.confidence = confidence


    def predict(self, image):

        results = self.model.predict(
            source=image,

            conf=self.confidence,

            verbose=False,
        )

        return results[0]


    def predict_image(self, image):

        result = self.predict(image)

        annotated_image = result.plot()

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                class_name = (
                    result.names[class_id]
                )

                detections.append(
                    {
                        "class": class_name,

                        "confidence": round(
                            confidence,
                            4
                        ),

                        "box": [
                            round(float(value), 2)
                            for value in coordinates
                        ],
                    }
                )

        return annotated_image, detections


if __name__ == "__main__":

    detector = PCBDetector()

    print("PCB detector loaded successfully.")