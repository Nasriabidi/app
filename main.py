"""Floor Inspection API - YOLOv8 Segmentation Inference."""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

model = None


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model
    model = load_model()
    yield


app = FastAPI(title="Crack Inspection API", lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict", response_class=Response)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    try:
        contents = await file.read()
        temp_path.write_bytes(contents)
        image = cv2.imread(str(temp_path))
        if image is None:
            raise HTTPException(400, "Invalid image")

        results = model.predict(
            str(temp_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        if not results:
            raise HTTPException(500, "Model returned no results")

        annotated = results[0].plot()
        _, buffer = cv2.imencode(".jpg", annotated)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
