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
    num_down = 2  # downsample steps
    num_bilateral = 7  # number of bilateral filtering steps

    img_color = img
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)

    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)

    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    output = cv2.bitwise_and(img_color, img_edge)
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
