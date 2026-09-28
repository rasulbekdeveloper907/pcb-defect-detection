import gradio as gr
import pandas as pd

from pathlib import Path
import sys


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))


from src.inference import PCBDetector


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = ROOT_DIR / "models" / "best.pt"


detector = PCBDetector(
    model_path=MODEL_PATH
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def detect_defects(image, confidence):

    if image is None:

        return (
            None,
            pd.DataFrame(
                columns=[
                    "Defect",
                    "Confidence",
                    "X1",
                    "Y1",
                    "X2",
                    "Y2",
                ]
            ),
            "Please upload a PCB image."
        )

    detector.confidence = confidence

    annotated_image, detections = (
        detector.predict_image(image)
    )

    rows = []

    for detection in detections:

        x1, y1, x2, y2 = (
            detection["box"]
        )

        rows.append(
            {
                "Defect": detection["class"],

                "Confidence": (
                    f"{detection['confidence']:.2%}"
                ),

                "X1": round(x1, 2),
                "Y1": round(y1, 2),
                "X2": round(x2, 2),
                "Y2": round(y2, 2),
            }
        )

    dataframe = pd.DataFrame(rows)

    if len(detections) == 0:

        status = "✅ No PCB defects detected."

    else:

        status = (
            f"⚠️ {len(detections)} "
            f"defect(s) detected."
        )

    return (
        annotated_image,
        dataframe,
        status
    )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="PCB Defect Detection"
) as demo:

    gr.Markdown(
        """
        # 🔧 PCB Defect Detection System

        Upload a PCB image and the YOLO model will
        detect manufacturing defects automatically.
        """
    )

    with gr.Row:

        with gr.Column:

            input_image = gr.Image(
                type="numpy",
                label="Upload PCB Image"
            )

            confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )

            detect_button = gr.Button(
                "🔍 Detect Defects",
                variant="primary"
            )

        with gr.Column:

            output_image = gr.Image(
                label="Detection Result"
            )

            status = gr.Markdown(
                "Upload an image to start."
            )

    results_table = gr.Dataframe(
        headers=[
            "Defect",
            "Confidence",
            "X1",
            "Y1",
            "X2",
            "Y2",
        ],
        label="Detected Defects",
        interactive=False,
    )


    detect_button.click(
        fn=detect_defects,

        inputs=[
            input_image,
            confidence,
        ],

        outputs=[
            output_image,
            results_table,
            status,
        ],
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch()