from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from detection import run_detection, check_alerts

app = FastAPI(title="AquaScan Sonar AI Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "model": "YOLOv16 drishti-sss detector"}

@app.post("/api/detect")
async def detect_debris(file: UploadFile = File(...), model: str = Form("both")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    image_bytes = await file.read()
    if model not in {"aquascan", "pipe", "both"}:
        raise HTTPException(status_code=400, detail="model must be aquascan, pipe, or both")

    detections, img_w, img_h = run_detection(image_bytes, model)
    alerts = check_alerts(detections)

    return JSONResponse(content={
        "filename": file.filename,
        "model": model,
        "image_dimensions": {"width": img_w, "height": img_h},
        "count": len(detections),
        "detections": detections,
        "alerts": alerts
    })