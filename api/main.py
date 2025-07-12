from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cv2
import numpy as np
import uuid
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process/")
async def process_image(
    request: Request,
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
        sketch = cv2.divide(gray, 255 - blur, scale=256.0)
        result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif operation == "cartoon":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(img, 9, 250, 250)
        result = cv2.bitwise_and(color, color, mask=edges)

    else:
        result = img

    filename = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, result)

    output_url = f"/static/outputs/{filename}"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "output_url": output_url}
    )
