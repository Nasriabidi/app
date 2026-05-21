"""Floor Inspection API - YOLOv8 Segmentation Inference."""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
import cv2
import whylogs as why
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import (
    CollectorRegistry, Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
)
from ultralytics import YOLO
from whylogs.core import DatasetProfileView

# ── Paths & config ────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "model-manifest.json"
MODEL_PATH    = BASE_DIR / "model" / "best.pt"
BASELINE_PATH = BASE_DIR / "baseline_profile.bin"
TEMP_DIR      = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

CONF_THRESHOLD = 0.30
IOU_THRESHOLD  = 0.45

# How many inferences to accumulate before computing drift
DRIFT_WINDOW = 10

# ── Global state ──────────────────────────────────────────────────────────────
model            = None
baseline_view    = None   # whylogs DatasetProfileView loaded from S3
inference_buffer = []     # accumulates recent inference records for drift

# ── Prometheus metrics ────────────────────────────────────────────────────────
METRICS_REGISTRY = CollectorRegistry()

inference_counter = Counter(
    "crack_inspection_inferences_total",
    "Total number of inference requests processed",
    registry=METRICS_REGISTRY,
)
inference_errors = Counter(
    "crack_inspection_errors_total",
    "Total number of failed inference requests",
    registry=METRICS_REGISTRY,
)
detections_gauge = Gauge(
    "crack_inspection_detections_last",
    "Number of cracks detected in the last inference",
    registry=METRICS_REGISTRY,
)
confidence_mean_gauge = Gauge(
    "crack_inspection_confidence_mean_last",
    "Mean confidence score of the last inference",
    registry=METRICS_REGISTRY,
)
confidence_max_gauge = Gauge(
    "crack_inspection_confidence_max_last",
    "Max confidence score of the last inference",
    registry=METRICS_REGISTRY,
)
latency_histogram = Histogram(
    "crack_inspection_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=METRICS_REGISTRY,
)
drift_score_gauge = Gauge(
    "crack_inspection_drift_score",
    "Drift score vs baseline (0=no drift, 1=max drift). Updated every DRIFT_WINDOW inferences.",
    registry=METRICS_REGISTRY,
)


# ── S3 helpers ────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    """Read model-manifest.json from the container filesystem."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"model-manifest.json not found at {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text())


def download_model_from_s3(manifest: dict) -> None:
    """Download model from S3 using the key in the manifest."""
    bucket  = manifest["models_bucket"]
    s3_key  = manifest["model_s3_key_latest"]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[startup] Downloading model from s3://{bucket}/{s3_key} ...")
    boto3.client("s3").download_file(bucket, s3_key, str(MODEL_PATH))
    print(f"[startup] Model downloaded to {MODEL_PATH}")


def download_baseline_from_s3(manifest: dict) -> DatasetProfileView | None:
    """Download whylogs baseline profile from S3 and return as DatasetProfileView."""
    baseline_key = manifest.get("baseline_profile_s3_key", "")
    if not baseline_key:
        print("[startup] No baseline_profile_s3_key in manifest — drift detection disabled.")
        return None
    bucket = manifest["models_bucket"]
    print(f"[startup] Downloading baseline from s3://{bucket}/{baseline_key} ...")
    boto3.client("s3").download_file(bucket, baseline_key, str(BASELINE_PATH))
    view = DatasetProfileView.read(str(BASELINE_PATH))
    print(f"[startup] Baseline profile loaded.")
    return view


# ── Drift computation ─────────────────────────────────────────────────────────

def compute_drift_score(current_records: list, baseline: DatasetProfileView) -> float:
    """
    Compare current inference window against baseline using whylogs.
    Returns a drift score between 0.0 (no drift) and 1.0 (max drift).

    Strategy: for each metric (num_detections, mean_confidence, max_confidence),
    compare the current window mean vs baseline mean, normalize by baseline stddev.
    Average the per-metric scores, capped at 1.0.
    """
    if not current_records or baseline is None:
        return 0.0

    try:
        metrics = ["num_detections", "mean_confidence", "max_confidence"]
        scores  = []

        for metric in metrics:
            # Get baseline stats
            col = baseline.get_column(metric)
            if col is None:
                continue
            dist = col.get_metric("distribution")
            if dist is None:
                continue

            baseline_mean = dist.mean.value
            baseline_std  = dist.stddev.value if dist.stddev.value > 0 else 1.0

            # Current window mean
            current_mean = sum(r[metric] for r in current_records) / len(current_records)

            # Normalized distance (z-score style), capped at 1.0
            score = min(abs(current_mean - baseline_mean) / baseline_std, 1.0)
            scores.append(score)

        if not scores:
            return 0.0

        return round(sum(scores) / len(scores), 4)

    except Exception as e:
        print(f"[drift] Error computing drift score: {e}")
        return 0.0


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, baseline_view

    manifest      = load_manifest()
    download_model_from_s3(manifest)
    model         = YOLO(str(MODEL_PATH))
    baseline_view = download_baseline_from_s3(manifest)

    print(f"[startup] Model loaded: {manifest.get('model_name', 'unknown')}")
    print(f"[startup] MLflow run  : {manifest.get('mlflow_run_id', 'unknown')}")
    print(f"[startup] Trained at  : {manifest.get('trained_at', 'unknown')}")
    print(f"[startup] mAP50 (box) : {manifest.get('mlflow_metrics', {}).get('mAP50_box', 'N/A')}")
    print(f"[startup] Drift window: every {DRIFT_WINDOW} inferences")

    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Crack Inspection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(METRICS_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_class=Response)
async def predict(file: UploadFile = File(...)):
    global inference_buffer

    if model is None:
        raise HTTPException(503, "Model not loaded")

    temp_path  = TEMP_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    start_time = time.time()

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

        result = results[0]
        boxes  = result.boxes

        num_detections = len(boxes) if boxes is not None else 0
        confidences    = boxes.conf.tolist() if boxes is not None and len(boxes) > 0 else []
        mean_conf      = sum(confidences) / len(confidences) if confidences else 0.0
        max_conf       = max(confidences) if confidences else 0.0
        latency        = time.time() - start_time

        # ── Prometheus metrics ────────────────────────────────────────────────
        inference_counter.inc()
        detections_gauge.set(num_detections)
        confidence_mean_gauge.set(mean_conf)
        confidence_max_gauge.set(max_conf)
        latency_histogram.observe(latency)

        # ── whylogs — log this inference ──────────────────────────────────────
        why.log({
            "num_detections":  num_detections,
            "mean_confidence": mean_conf,
            "max_confidence":  max_conf,
            "latency_seconds": latency,
        })

        # ── Drift detection — accumulate and compute every DRIFT_WINDOW ───────
        inference_buffer.append({
            "num_detections":  num_detections,
            "mean_confidence": mean_conf,
            "max_confidence":  max_conf,
        })

        if len(inference_buffer) >= DRIFT_WINDOW:
            score = compute_drift_score(inference_buffer, baseline_view)
            drift_score_gauge.set(score)
            print(f"[drift] Score updated: {score} (window={DRIFT_WINDOW})")
            inference_buffer = []   # reset window

        # ── Annotated image response ──────────────────────────────────────────
        annotated        = result.plot()
        _, buffer        = cv2.imencode(".jpg", annotated)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        inference_errors.inc()
        raise HTTPException(500, f"Inference error: {str(e)}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
