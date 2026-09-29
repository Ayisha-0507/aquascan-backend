from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from detection import run_detection, check_alerts

app = FastAPI(title="AquaScan Sonar AI Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "online"}


@app.post("/api/detect")
async def detect_debris(file: UploadFile = File(...), model: str = Form("pipe")):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")

    image_bytes = await file.read()
    model = model.strip().lower()
    model = {
        "aquascan model": "aquascan",
        "pipe model": "pipe",
        "both models": "both",
    }.get(model, model)
    if model not in {"aquascan", "pipe", "both"}:
        raise HTTPException(status_code=400, detail="model must be aquascan, pipe, or both")

    try:
        detections, img_w, img_h = await run_in_threadpool(run_detection, image_bytes, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={
        "filename": file.filename,
        "model": model,
        "image_dimensions": {"width": img_w, "height": img_h},
        "count": len(detections),
        "detections": detections,
        "alerts": check_alerts(detections),
    })