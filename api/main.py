from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cv2
import numpy as np
import uuid
import os

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
OUTPUT_DIR = os.path.join(STATIC_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "..", "templates"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process_image(
    file: UploadFile = File(...),
    operation: str = Form(...)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if operation == "sketch":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        output = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif operation == "cartoon":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        output = cartoon

    elif operation == "color2bw":
        output = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    elif operation == "bw2color":
        # Fake colorization with heatmap for demo
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        colorized = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        output = colorized

    else:
        return JSONResponse({"error": "Invalid operation"}, status_code=400)

    filename = f"{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(out_path, output)

    return {"filename": filename}
