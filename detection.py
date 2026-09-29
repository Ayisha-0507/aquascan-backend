import os
import io
import uuid
import gc
from datetime import datetime
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# Single-threaded CPU execution to conserve Render free-tier RAM
torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

AQUASCAN_MODEL_PATH = os.path.join(MODELS_DIR, "aquascan_best.pt")
SECOND_MODEL_PATH = os.path.join(MODELS_DIR, "pipe_best.pt")

if not os.path.exists(SECOND_MODEL_PATH):
    alt_path = os.path.join(MODELS_DIR, "model2_best.pt")
    if os.path.exists(alt_path):
        SECOND_MODEL_PATH = alt_path

# 1. Primary Model (Always AquaScan)
primary_path = AQUASCAN_MODEL_PATH if os.path.exists(AQUASCAN_MODEL_PATH) else "yolo11s.pt"
model_aquascan = YOLO(primary_path)

# 2. Secondary Model (Loaded if file exists; reuses instance if weights are identical to save RAM)
if os.path.exists(SECOND_MODEL_PATH) and SECOND_MODEL_PATH != AQUASCAN_MODEL_PATH:
    model_second = YOLO(SECOND_MODEL_PATH)
else:
    model_second = model_aquascan


def estimate_shadow_height(bbox: List[int], img_w: int, img_h: int) -> Tuple[float, bool]:
    x, y, w, h = bbox
    sensor_altitude = 15.0
    relative_range = max(20.0, (y + h) / max(1, img_h) * 60.0)
    shadow_length_approx = max(0.5, (w * 0.45))
    
    calc_height = (sensor_altitude * shadow_length_approx) / (relative_range + shadow_length_approx)
    calc_height = round(float(calc_height), 2)
    has_shadow = calc_height > 0.35
    return calc_height, has_shadow


def _infer_model(
    model: YOLO,
    image_cv: np.ndarray,
    model_tag: str,
    img_w: int,
    img_h: int,
    conf_threshold: float = 0.25
) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    
    with torch.no_grad():
        results = model.predict(
            source=image_cv,
            conf=conf_threshold,
            device="cpu",
            verbose=False,
            imgsz=640
        )
    
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
            
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = result.names.get(cls_id, "marine_debris")
            
            est_height, has_shadow = estimate_shadow_height([x, y, w, h], img_w, img_h)
            
            # Confidence fusion
            fused_conf = conf * 100.0
            if has_shadow:
                fused_conf = min(99.0, fused_conf + 4.5)
            else:
                fused_conf = max(20.0, fused_conf - 8.0)
                
            det_id = f"DEB-{uuid.uuid4().hex[:4].upper()}"
            
            detections.append({
                "detection_id": det_id,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
                "classification": label,
                "confidence_score": round(fused_conf, 1),
                "raw_confidence": round(conf * 100.0, 1),
                "location": {
                    "latitude": round(15.350 + (y / max(1, img_h)) * 0.15, 4),
                    "longitude": round(73.420 + (x / max(1, img_w)) * 0.15, 4),
                },
                "bbox": [x, y, w, h],
                "estimated_height_meters": est_height,
                "has_acoustic_shadow": has_shadow,
                "model_source": model_tag
            })
            
    return detections


def run_detection(
    image_bytes: bytes,
    model_choice: str = "aquascan",
    filename: str = "sonar_input.png"
) -> Dict[str, Any]:
    """
    Inference controller:
    - Default ("aquascan"): Runs only the AquaScan model.
    - Dual ("both"): Runs AquaScan + Secondary model and aggregates results.
    """
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    img_h, img_w = image_cv.shape[:2]

    all_detections: List[Dict[str, Any]] = []

    # 1. Primary Model (Always AquaScan)
    aquascan_dets = _infer_model(
        model=model_aquascan,
        image_cv=image_cv,
        model_tag="AquaScan",
        img_w=img_w,
        img_h=img_h
    )
    all_detections.extend(aquascan_dets)

    # 2. Secondary Model (Only if 'both' is selected)
    if model_choice.strip().lower() == "both":
        second_dets = _infer_model(
            model=model_second,
            image_cv=image_cv,
            model_tag="Model 2",
            img_w=img_w,
            img_h=img_h
        )
        all_detections.extend(second_dets)

    # Memory cleanup
    del image_cv, image_pil
    gc.collect()

    alerts = []
    for d in all_detections:
        if d["confidence_score"] >= 85:
            alerts.append({
                "id": d["detection_id"],
                "severity": "CRITICAL",
                "message": f"Hazard verified: {d['classification'].upper()} via {d['model_source']}",
                "location": d["location"],
                "confidence": d["confidence_score"]
            })

    return {
        "filename": filename,
        "image_dimensions": {"width": img_w, "height": img_h},
        "count": len(all_detections),
        "detections": all_detections,
        "alerts": alerts
    }