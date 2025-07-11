import cv2
import numpy as np

def to_bw(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def day_to_night(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * 0.3, 0, 255).astype(np.uint8)
    hsv_night = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_night, cv2.COLOR_HSV2BGR)

def cartoonify(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch

def fake_colorize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colorized = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colorized
