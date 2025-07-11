from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import uuid
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process_image(request: Request, file: UploadFile, operation: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if operation == "sketch":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        inv_blur = cv2.bitwise_not(blur)
        output = cv2.divide(gray, inv_blur, scale=256.0)
    elif operation == "cartoon":
        num_down = 2
        num_bilateral = 7

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
    elif operation == "bw2color":
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        return {"error": "Invalid operation"}

    filename = f"static/output_{uuid.uuid4().hex}.png"
    cv2.imwrite(filename, output)
    return FileResponse(filename, media_type="image/png", filename="result.png")
