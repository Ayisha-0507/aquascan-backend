from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from detection import run_detection

app = FastAPI(title="AquaScan AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    model: str = Form("aquascan")
):
    contents = await file.read()
    
    # run_detection now returns the full dictionary including 'detections' and 'alerts'
    results = run_detection(
        image_bytes=contents,
        model_choice=model,
        filename=file.filename or "sonar_scan.png"
    )
    
    return results