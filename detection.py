import cv2
import numpy as np
from ultralytics import YOLO
import uuid
from datetime import datetime
import pandas as pd
import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
model_paths = {
    "aquascan": os.path.join(MODEL_DIR, "aquascan_best.pt"),
    "pipe": os.path.join(MODEL_DIR, "pipe_best.pt"),
}
models = {}


def get_model(model_name):
    if model_name not in model_paths:
        raise ValueError("model_choice must be 'aquascan', 'pipe', or 'both'")
    if model_name not in models:
        models[model_name] = YOLO(model_paths[model_name])
    return models[model_name]

def extract_shadow(img, bbox, direction='below'):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    box_h = max(y2 - y1, 1)

    if direction == 'below':
        region = img[y2:min(y2 + box_h * 2, h), x1:x2]
    elif direction == 'above':
        region = img[max(0, y1 - box_h * 2):y1, x1:x2]
    else:
        region = img[y1:y2, x2:min(x2 + box_h * 2, w)]

    if region.size == 0:
        return 0, False

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    shadow_pixels = cv2.countNonZero(dark_mask)
    has_shadow = shadow_pixels > (region.shape[0] * region.shape[1] * 0.15)
    shadow_length_px = shadow_pixels / box_h
    return shadow_length_px, has_shadow

def estimate_height(shadow_length_px, altitude_m=3.1, range_m=14.8, pixel_to_meter_ratio=0.05):
    shadow_length_m = shadow_length_px * pixel_to_meter_ratio
    height = altitude_m * shadow_length_m / (shadow_length_m + range_m)
    return round(float(height), 2)

def fuse_confidence(yolo_conf, has_shadow):
    score = yolo_conf * 100
    score += 5 if has_shadow else -10
    return round(max(0, min(100, score)), 1)

def geotag(pixel_x, pixel_y, img_width, img_height, base_lat=9.842000, base_lon=78.113000, area_span_deg=0.01):
    lat = base_lat + (pixel_y / img_height) * area_span_deg
    lon = base_lon + (pixel_x / img_width) * area_span_deg
    return round(lat, 6), round(lon, 6)

def calculate_iou(first_bbox, second_bbox):
    first_x1, first_y1, first_w, first_h = first_bbox
    second_x1, second_y1, second_w, second_h = second_bbox
    first_x2, first_y2 = first_x1 + first_w, first_y1 + first_h
    second_x2, second_y2 = second_x1 + second_w, second_y1 + second_h
    intersection_w = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_h = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_w * intersection_h
    union = (first_w * first_h) + (second_w * second_h) - intersection
    return intersection / union if union else 0


def run_detection(image_bytes: bytes, model_choice="both"):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return [], 0, 0

    h, w = img.shape[:2]
    selected_models = list(models.keys()) if model_choice == "both" else [model_choice]
    if any(selected_model not in models for selected_model in selected_models):
        raise ValueError("model_choice must be 'aquascan', 'pipe', or 'both'")

    detections = []
    for model_name in selected_models:
        results = get_model(model_name).predict(source=img, conf=0.15, imgsz=640, max_det=100, verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                shadow_len, has_shadow = extract_shadow(img, [x1, y1, x2, y2])
                height_m = estimate_height(shadow_len)
                final_conf = fuse_confidence(conf, has_shadow)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                lat, lon = geotag(cx, cy, w, h)
                bbox = [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]

                duplicate = next((item for item in detections if calculate_iou(item["bbox"], bbox) >= 0.5), None)
                if duplicate:
                    if final_conf > duplicate["confidence_score"]:
                        duplicate.update({
                            "classification": cls_name,
                            "confidence_score": final_conf,
                            "raw_confidence": round(conf * 100, 1),
                            "bbox": bbox,
                            "estimated_height_meters": height_m,
                            "has_acoustic_shadow": has_shadow,
                            "model_source": model_name,
                        })
                    continue

                detections.append({
                    "detection_id": str(uuid.uuid4())[:8],
                    "timestamp": datetime.now().isoformat(),
                    "classification": cls_name,
                    "confidence_score": final_conf,
                    "raw_confidence": round(conf * 100, 1),
                    "location": {"latitude": lat, "longitude": lon},
                    "bbox": bbox,
                    "estimated_height_meters": height_m,
                    "has_acoustic_shadow": has_shadow,
                    "model_source": model_name,
                })

    return detections, w, h

def check_alerts(detections, confidence_threshold=65.0):
    return [
        {
            "id": d["detection_id"],
            "severity": "CRITICAL" if d["classification"] == "mine_cylinder" else "WARNING",
            "message": f"{d['classification'].replace('_', ' ').title()} confirmed with acoustic shadow",
            "location": d["location"],
            "confidence": d["confidence_score"]
        }
        for d in detections
        if d["confidence_score"] >= confidence_threshold
    ]