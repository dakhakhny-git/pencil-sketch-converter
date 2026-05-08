import cv2
import numpy as np


def pencil_sketch(image, blur_ksize=21, strength=256, rgb=False):

    if rgb:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted = 255 - gray

    if blur_ksize < 3:
        blur_ksize = 3

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    blurred = cv2.GaussianBlur(inverted, (blur_ksize, blur_ksize), 0)

    inverted_blur = 255 - blurred

    sketch = cv2.divide(gray, inverted_blur, scale=strength)

    return sketch


def edge_sketch(image, rgb=False):

    if rgb:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    edges = cv2.Canny(smooth, 50, 150)

    return edges