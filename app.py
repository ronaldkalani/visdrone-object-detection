import streamlit as st
import torch
import cv2
import os
import numpy as np
from PIL import Image

# ---- SETTINGS ----
# ✅ UPDATE this path to match your local folder structure
IMAGE_DIR = r"G:\My Drive\ComputerVision Consultant\VisDrone2019-DET-train\VisDrone2019-DET-train\images"

# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# ---- APP HEADER ----
st.set_page_config(page_title="VisDrone Object Detection", layout="wide")
st.title("🛰️ VisDrone Object Detection with YOLOv5")
st.markdown("This demo runs YOLOv5 on real drone surveillance images.")

# ---- LOAD IMAGES ----
if not os.path.exists(IMAGE_DIR):
    st.error(f"Image directory not found: `{IMAGE_DIR}`")
    st.stop()

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])

if not image_files:
    st.warning("No `.jpg` images found in the folder.")
    st.stop()

# ---- IMAGE SELECTION ----
selected_file = st.selectbox("Choose an image:", image_files)
image_path = os.path.join(IMAGE_DIR, selected_file)

# ---- LOAD & DISPLAY IMAGE ----
image = Image.open(image_path)
st.image(image, caption="Original Image", use_column_width=True)

# ---- OBJECT DETECTION ----
img_np = np.array(image)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

with st.spinner("Detecting objects..."):
    results = model(img_bgr)

# ---- DISPLAY RESULTS ----
results.render()
st.image(results.ims[0], caption="Detected Objects", use_column_width=True)

# ---- DISPLAY TABLE ----
st.subheader("Detection Results")
st.dataframe(results.pandas().xyxy[0])
