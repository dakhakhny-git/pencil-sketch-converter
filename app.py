import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ----------- Sketch Functions -----------

def pencil_sketch(image, blur_ksize=21, strength=256):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    inverted = 255 - gray

    if blur_ksize < 3:
        blur_ksize = 3

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    blurred = cv2.GaussianBlur(inverted, (blur_ksize, blur_ksize), 0)

    inverted_blur = 255 - blurred

    sketch = cv2.divide(gray, inverted_blur, scale=strength)

    return sketch


def edge_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    edges = cv2.Canny(smooth, 50, 150)

    edges = 255 - edges

    return edges


def convert_to_downloadable_image(image_array):
    image_pil = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


# ----------- Streamlit App -----------

st.set_page_config(
    page_title="Pencil Sketch Converter",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Image to Pencil Sketch Converter")

st.write(
    "Upload a color image and convert it into a pencil-sketch-style image "
    "using classical image processing techniques."
)

st.sidebar.header("Settings")

mode = st.sidebar.selectbox(
    "Choose sketch mode",
    ["Classic Pencil Sketch", "Edge-Based Sketch"]
)

blur = st.sidebar.slider(
    "Blur Size",
    min_value=3,
    max_value=99,
    value=21,
    step=2
)

strength = st.sidebar.slider(
    "Sketch Strength",
    min_value=100,
    max_value=400,
    value=256,
    step=10
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    if mode == "Classic Pencil Sketch":
        result = pencil_sketch(image_np, blur, strength)

    else:
        sketch = pencil_sketch(image_np, blur, strength)
        edges = edge_sketch(image_np)
        result = cv2.bitwise_and(sketch, edges)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Sketch Result")
        st.image(result, clamp=True, use_container_width=True)

    download_image = convert_to_downloadable_image(result)

    st.download_button(
        label="Download Sketch",
        data=download_image,
        file_name="pencil_sketch.png",
        mime="image/png"
    )

else:
    st.info("Upload an image to start.")