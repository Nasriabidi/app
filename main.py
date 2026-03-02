"""
Floor Inspection API - YOLOv8 Segmentation Inference
=====================================================
This FastAPI application provides an inference endpoint for a YOLOv8 
segmentation model trained for floor defect detection (cracks, spalls, etc.).

How it works:
1. On startup, the TorchScript model is loaded into memory (CPU).
2. POST /predict accepts an image file, runs inference, and returns 
   the annotated image with detected defects drawn on it.
3. GET /health returns API status for health checks.
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory where this script is located
BASE_DIR = Path(__file__).resolve().parent

# Path to the exported YOLOv8 TorchScript model
MODEL_PATH = BASE_DIR / "model" / "best.torchscript"

# Temporary directory for saving uploaded images before processing
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# YOLOv8 expects images resized to this dimension (640x640 is the default)
INPUT_SIZE = 640

# Minimum confidence score to consider a detection valid (0.0 to 1.0)
CONF_THRESHOLD = 0.25

# IoU threshold for Non-Maximum Suppression (removes overlapping boxes)
IOU_THRESHOLD = 0.45

# Class names that match the order used during model training
# IMPORTANT: Update this list to match YOUR trained model's classes
CLASS_NAMES = ["crack"]

# BGR colors for drawing each class (one color per class)
PALETTE = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255)]

# Global variable to hold the loaded model (populated at startup)
model = None


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    """
    Load the TorchScript model from disk.
    - Uses CPU for inference (suitable for deployment without GPU).
    - Sets model to evaluation mode (disables dropout, batch norm updates).
    """
    ts_model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
    ts_model.eval()
    return ts_model


# =============================================================================
# FASTAPI LIFESPAN (STARTUP/SHUTDOWN)
# =============================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    FastAPI lifespan context manager.
    - On startup: Load the model once into memory.
    - On shutdown: Clean up (optional).
    This ensures the model is loaded only once when the server starts,
    not on every request.
    """
    global model
    model = load_model()
    yield  # Server runs here
    # Shutdown: model can be cleaned up here if needed


# Create the FastAPI application with the lifespan handler
app = FastAPI(title="Floor Inspection API", lifespan=lifespan)


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess(image):
    """
    Prepare an image for YOLOv8 inference.
    
    Steps:
    1. Resize image while keeping aspect ratio (letterboxing).
    2. Add gray padding to make it exactly 640x640.
    3. Convert BGR to RGB (YOLOv8 expects RGB).
    4. Normalize pixel values from [0, 255] to [0.0, 1.0].
    5. Convert to PyTorch tensor with shape (1, 3, 640, 640).
    
    Args:
        image: Input BGR image (numpy array from cv2.imread).
    
    Returns:
        tensor: Preprocessed image tensor for model input.
        orig_hw: Original image height and width tuple.
        ratio: Scale factor used for resizing.
        pad_xy: (left, top) padding offsets.
    """
    # Get original dimensions
    orig_h, orig_w = image.shape[:2]
    
    # Calculate scale to fit image into INPUT_SIZE while keeping aspect ratio
    ratio = min(INPUT_SIZE / orig_h, INPUT_SIZE / orig_w)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding needed to reach INPUT_SIZE x INPUT_SIZE
    pad_w, pad_h = INPUT_SIZE - new_w, INPUT_SIZE - new_h
    top, left = pad_h // 2, pad_w // 2
    
    # Add gray padding (114 is YOLO's standard padding color)
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Convert BGR -> RGB, then normalize to [0, 1]
    blob = padded[:, :, ::-1].astype(np.float32) / 255.0
    
    # Convert from HWC (height, width, channels) to CHW format
    # Then add batch dimension: (3, 640, 640) -> (1, 3, 640, 640)
    tensor = torch.from_numpy(np.transpose(blob, (2, 0, 1))).unsqueeze(0)
    
    return tensor, (orig_h, orig_w), ratio, (left, top)


# =============================================================================
# BOUNDING BOX CONVERSION
# =============================================================================

def xywh_to_xyxy(boxes):
    """
    Convert bounding boxes from center format to corner format.
    
    Input format:  [center_x, center_y, width, height]
    Output format: [x1, y1, x2, y2] (top-left and bottom-right corners)
    
    Args:
        boxes: Numpy array of shape (N, 4) in xywh format.
    
    Returns:
        Numpy array of shape (N, 4) in xyxy format.
    """
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
    return out


# =============================================================================
# NON-MAXIMUM SUPPRESSION (NMS)
# =============================================================================

def nms(boxes, scores, iou_thr):
    """
    Non-Maximum Suppression: removes overlapping bounding boxes.
    
    When multiple boxes detect the same object, NMS keeps only the one
    with the highest confidence score and removes boxes that overlap
    significantly (IoU > threshold).
    
    Args:
        boxes: Numpy array (N, 4) of bounding boxes in xyxy format.
        scores: Numpy array (N,) of confidence scores.
        iou_thr: IoU threshold (0.45 typical) - boxes with higher overlap are removed.
    
    Returns:
        List of indices of boxes to keep.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)  # Calculate area of each box
    
    # Sort by confidence (highest first)
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        # Pick the box with highest remaining confidence
        i = order[0]
        keep.append(int(i))
        
        if order.size == 1:
            break
        
        # Calculate IoU of this box with all remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Intersection area
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        
        # IoU = intersection / union
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        # Keep only boxes with IoU below threshold (not overlapping too much)
        order = order[np.where(iou <= iou_thr)[0] + 1]
    
    return keep


# =============================================================================
# POSTPROCESSING (PARSE MODEL OUTPUT)
# =============================================================================

def postprocess(output, orig_hw, ratio, pad_xy, proto=None):
    """
    Parse the raw YOLOv8 model output into usable detections.
    
    YOLOv8 output format: tensor of shape (1, num_features, num_predictions)
    - First 4 values: bounding box (cx, cy, w, h)
    - Next N values: class probabilities (one per class)
    - Remaining values: mask coefficients (for segmentation models)
    
    Args:
        output: Raw model output tensor.
        orig_hw: Original image (height, width).
        ratio: Scale ratio used during preprocessing.
        pad_xy: (left, top) padding offsets from preprocessing.
        proto: Prototype masks from segmentation head (optional).
    
    Returns:
        List of detection dictionaries with keys:
        - class_id, class_name, confidence, bbox, mask (optional)
    """
    # Convert to numpy and transpose: (1, features, N) -> (N, features)
    pred = output[0].cpu().numpy().T
    num_classes = len(CLASS_NAMES)
    
    # Split the prediction columns
    boxes = pred[:, :4]                      # First 4 cols: bounding boxes
    class_scores = pred[:, 4:4 + num_classes]  # Next N cols: class scores
    
    # Remaining cols: mask coefficients (only for segmentation models)
    mask_coeffs = pred[:, 4 + num_classes:] if pred.shape[1] > 4 + num_classes else None

    # For each prediction, get the class with highest probability
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    # Filter out predictions below confidence threshold
    valid = confidences >= CONF_THRESHOLD
    boxes, confidences, class_ids = boxes[valid], confidences[valid], class_ids[valid]
    if mask_coeffs is not None:
        mask_coeffs = mask_coeffs[valid]

    # No detections above threshold
    if len(boxes) == 0:
        return []

    # Convert boxes from center format to corner format
    xyxy = xywh_to_xyxy(boxes)
    
    # Apply NMS separately for each class
    keep_indices = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_indices = np.where(cls_mask)[0]
        cls_keep = nms(xyxy[cls_mask], confidences[cls_mask], IOU_THRESHOLD)
        keep_indices.extend(cls_indices[cls_keep].tolist())

    # Keep only the boxes that survived NMS
    xyxy = xyxy[keep_indices]
    confidences = confidences[keep_indices]
    class_ids = class_ids[keep_indices]
    if mask_coeffs is not None:
        mask_coeffs = mask_coeffs[keep_indices]

    # Rescale boxes from padded 640x640 back to original image coordinates
    pad_x, pad_y = pad_xy
    xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - pad_x) / ratio, 0, orig_hw[1])
    xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - pad_y) / ratio, 0, orig_hw[0])

    # Process segmentation masks if available
    seg_masks = None
    if proto is not None and mask_coeffs is not None:
        seg_masks = process_masks(proto[0].cpu().numpy(), mask_coeffs, orig_hw, ratio, pad_xy)

    # Build list of detection dictionaries
    detections = []
    for idx in range(len(xyxy)):
        cls_id = int(class_ids[idx])
        det = {
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
            "confidence": float(confidences[idx]),
            "bbox": xyxy[idx].tolist(),  # [x1, y1, x2, y2]
        }
        if seg_masks is not None:
            det["mask"] = seg_masks[idx]
        detections.append(det)
    
    return detections


# =============================================================================
# SEGMENTATION MASK PROCESSING
# =============================================================================

def process_masks(proto, coeffs, orig_hw, ratio, pad_xy):
    """
    Generate binary segmentation masks from YOLOv8-seg output.
    
    YOLOv8-seg uses prototype masks + coefficients approach:
    - proto: Low-resolution prototype masks from the model (e.g., 160x160)
    - coeffs: Coefficients to combine prototypes into instance masks
    
    Args:
        proto: Prototype mask tensor of shape (num_protos, H, W).
        coeffs: Coefficients array of shape (N, num_protos).
        orig_hw: Original image (height, width).
        ratio: Scale ratio from preprocessing.
        pad_xy: Padding offsets from preprocessing.
    
    Returns:
        List of binary masks (numpy arrays) for each detection.
    """
    # Combine prototypes using coefficients, then apply sigmoid
    masks = 1.0 / (1.0 + np.exp(-(coeffs @ proto.reshape(proto.shape[0], -1))))
    masks = masks.reshape(len(coeffs), proto.shape[1], proto.shape[2])
    
    pad_x, pad_y = pad_xy
    unpadded_w, unpadded_h = int(orig_hw[1] * ratio), int(orig_hw[0] * ratio)
    
    result = []
    for m in masks:
        # Resize mask to INPUT_SIZE
        m_resized = cv2.resize(m, (INPUT_SIZE, INPUT_SIZE))
        
        # Remove the padding that was added during preprocessing
        m_cropped = m_resized[pad_y:pad_y + unpadded_h, pad_x:pad_x + unpadded_w]
        
        # Resize to original image dimensions
        m_orig = cv2.resize(m_cropped, (orig_hw[1], orig_hw[0]))
        
        # Binarize: pixels > 0.5 are part of the mask
        result.append((m_orig > 0.5).astype(np.uint8))
    
    return result


# =============================================================================
# DRAW DETECTIONS ON IMAGE
# =============================================================================

def draw_detections(image, detections):
    """
    Draw bounding boxes, labels, and segmentation masks on the image.
    
    Args:
        image: Original BGR image (numpy array).
        detections: List of detection dictionaries from postprocess().
    
    Returns:
        Annotated image with detections visualized.
    """
    overlay = image.copy()
    
    for det in detections:
        # Get color for this class
        colour = PALETTE[det["class_id"] % len(PALETTE)]
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # Draw segmentation mask (semi-transparent overlay)
        if "mask" in det:
            mask = det["mask"]
            coloured = np.zeros_like(overlay)
            coloured[:] = colour
            # Blend: 50% original + 50% mask color where mask == 1
            overlay[mask == 1] = cv2.addWeighted(
                overlay[mask == 1], 0.5, 
                coloured[mask == 1], 0.5, 0
            )

        # Draw bounding box rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
        
        # Draw label background and text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return overlay


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring/load balancers.
    
    Returns:
        {"status": "ok"} if the API is running.
    """
    return {"status": "ok"}


@app.post("/predict", response_class=Response)
async def predict(file: UploadFile = File(...)):
    """
    Run YOLOv8 segmentation inference on an uploaded image.
    
    Request:
        - POST with multipart/form-data
        - Field 'file': image file (JPEG, PNG, etc.)
    
    Response:
        - JPEG image with detections drawn on it.
        - Content-Type: image/jpeg
    
    Usage with curl:
        curl -X POST "http://localhost:8000/predict" -F "file=@photo.jpg" --output result.jpg
    """
    # Check model is loaded
    if model is None:
        raise HTTPException(503, "Model not loaded")

    # Save uploaded file temporarily
    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    
    try:
        # Read and save the uploaded file
        contents = await file.read()
        temp_path.write_bytes(contents)
        
        # Load image with OpenCV
        image = cv2.imread(str(temp_path))
        if image is None:
            raise HTTPException(400, "Invalid image")

        # Preprocess: resize, pad, normalize, convert to tensor
        tensor, orig_hw, ratio, pad_xy = preprocess(image)

        # Run inference (no gradient calculation needed)
        with torch.no_grad():
            raw_output = model(tensor)

        # YOLOv8-seg returns tuple: (detections, prototype_masks)
        # YOLOv8 (no seg) returns just detections
        if isinstance(raw_output, (tuple, list)) and len(raw_output) >= 2:
            det_output, proto = raw_output[0], raw_output[1]
        else:
            det_output = raw_output if isinstance(raw_output, torch.Tensor) else raw_output[0]
            proto = None

        # Parse model output into detection list
        detections = postprocess(det_output, orig_hw, ratio, pad_xy, proto)
        
        # Draw boxes and masks on the image
        annotated = draw_detections(image, detections)
        
        # Encode as JPEG and return
        _, buffer = cv2.imencode(".jpg", annotated)
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    finally:
        # Always clean up the temp file
        if temp_path.exists():
            temp_path.unlink()


# =============================================================================
# MAIN ENTRY POINT (for local development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    # Run server on all interfaces (0.0.0.0) port 8000
    # In production, use: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
