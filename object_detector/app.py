from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from main import detect_objects

app = FastAPI()

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # Read image bytes and decode with OpenCV
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run YOLOv11 detection
    result = detect_objects(image)

    return JSONResponse(content=result)
