import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.inference import PCBDetector


MODEL_PATH = ROOT_DIR / "models" / "best.pt"

app = FastAPI(
    title="PCB Defect Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = PCBDetector(
    model_path=MODEL_PATH
)


@app.get("/")
def root():
    return {
        "message": "PCB Defect Detection API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    try:
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        detector.confidence = 0.25

        _, detections = detector.predict_image(
            image
        )

        return {
            "success": True,
            "count": len(detections),
            "detections": detections
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )