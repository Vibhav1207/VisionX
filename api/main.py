from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

OUTPUT_PATH = os.path.join(BASE_DIR, "output.jpg")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process/")
async def process_image(
    request: Request,
    file: UploadFile,
    operation: str = Form(...)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if operation == "sketch":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        inv_blur = 255 - blur
        sketch = cv2.divide(gray, inv_blur, scale=256.0)
        output = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif operation == "cartoon":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 9
        )
        color = cv2.bilateralFilter(img, 9, 300, 300)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        output = cv2.bitwise_and(color, edges)

    elif operation == "color2bw":
        output = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    elif operation == "bw2color":
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    else:
        return {"error": "Invalid operation"}
    
    output_path = "output/result.jpg"
    cv2.imwrite(output_path, output)
    return FileResponse(output_path, media_type="image/jpeg")

    cv2.imwrite(OUTPUT_PATH, output)
    return FileResponse(OUTPUT_PATH, media_type="image/jpeg", filename="output.jpg")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
