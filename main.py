from pathlib import Path
import uuid

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import Response

# ---------------------------------------------------------------------
# configuration constants (same as before)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best.torchscript"
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASS_NAMES = ["crack"]
PALETTE = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

# global model variable
model = None

# ---------------------------------------------------------------------
# FastAPI app and startup handler
# ---------------------------------------------------------------------
app = FastAPI(title="Floor Inspection API")


@app.on_event("startup")
def startup_event():
    global model
    model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
    model.eval()


# ---------------------------------------------------------------------
# auxiliary functions (preprocessing, nms, etc.) – identical to before
# ---------------------------------------------------------------------
def preprocess(image):
    # …same implementation you already have…
    …
def xywh_to_xyxy(boxes):
    …
def nms(boxes, scores, iou_thr):
    …
def draw_detections(img, dets, ratio, pad):
    …

# ---------------------------------------------------------------------
# health‑check route – mirrors the diabetes example’s “/” endpoint
# ---------------------------------------------------------------------
class HealthResponse(BaseModel):
    message: str


@app.get("/", response_model=HealthResponse)
def read_root():
    return {"message": "Floor Inspection API is live"}


# ---------------------------------------------------------------------
# predict route – file upload instead of Pydantic body model
# ---------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if npimg is None:
        raise HTTPException(status_code=400, detail="invalid image")

    tensor, orig_hw, ratio, pad = preprocess(npimg)
    with torch.no_grad():
        out = model(tensor)[0].numpy()

    # …same post‑processing and drawing code you had previously…
    annotated = draw_detections(npimg, out, ratio, pad)
    ret, buf = cv2.imencode(".jpg", annotated)
    return Response(content=buf.tobytes(), media_type="image/jpeg")