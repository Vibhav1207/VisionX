from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
from .utils import to_bw, day_to_night, cartoonify, pencil_sketch, fake_colorize

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("../static/index.html") as f:
        return f.read()

@app.post("/process")
async def process_image(effect: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if effect == "bw":
        result = to_bw(img)
        output_path = "output/bw_image.png"
    elif effect == "night":
        result = day_to_night(img)
        output_path = "output/night_image.png"
    elif effect == "cartoon":
        result = cartoonify(img)
        output_path = "output/cartoon_image.png"
    elif effect == "sketch":
        result = pencil_sketch(img)
        output_path = "output/sketch_image.png"
    elif effect == "colorize":
        result = fake_colorize(img)
        output_path = "output/colorized_image.png"
    else:
        return {"error": "Invalid effect"}

    cv2.imwrite(output_path, result)
    return FileResponse(output_path)
