import streamlit as st
import os
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import time

# ---- SETTINGS ----
IMAGE_DIR = r"G:\My Drive\ComputerVision Consultant\VisDrone2019-DET-train\VisDrone2019-DET-train\images"
SUPPORTED_IMAGE_EXTS = ['.jpg', '.jpeg', '.png']
FRAME_DELAY = 0.1  # seconds between frames

# ---- Load YOLOv5 Model ----
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()
model.conf = 0.4  # confidence threshold

# ---- Load DeepSORT Tracker ----
tracker = DeepSort(max_age=30)

# ---- Helper Functions ----
def is_image_file(f):
    return any(f.lower().endswith(ext) for ext in SUPPORTED_IMAGE_EXTS)

def load_images(folder):
    return sorted([f for f in os.listdir(folder) if is_image_file(f)])

# ---- Streamlit UI ----
st.title("🎥 VisDrone Object Tracking (YOLOv5 + DeepSORT) - Auto Playback")

st.write("📁 Loading image directory:", IMAGE_DIR)
if not os.path.exists(IMAGE_DIR):
    st.error(f"❌ Image directory not found: {IMAGE_DIR}")
    st.stop()

image_files = load_images(IMAGE_DIR)
if not image_files:
    st.warning("⚠️ No image files found in the directory.")
    st.stop()

# ---- Controls ----
start_idx = st.slider("Select starting frame", 0, len(image_files) - 1, 0)
play_button = st.button("▶️ Play All Frames")

frame_placeholder = st.empty()

# ---- Playback ----
if play_button:
    for idx in range(start_idx, len(image_files)):
        img_path = os.path.join(IMAGE_DIR, image_files[idx])
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # ---- YOLOv5 Detection ----
        results = model(image_rgb)
        detections = results.xyxy[0].cpu().numpy()

        # ---- Correct DeepSORT Input Format ----
        tracker_inputs = []
        for det in detections:
            if len(det) < 6:
                continue

            x1, y1, x2, y2, conf, cls = det[:6]
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            confidence = float(conf)
            class_name = str(int(cls))

            tracker_inputs.append([bbox, confidence, class_name])

        # ---- DeepSORT Tracking ----
        tracks = tracker.update_tracks(tracker_inputs, frame=image_bgr)

        # ---- Annotate Image ----
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = track.track_id
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_rgb, f'ID:{track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---- Display Frame ----
        frame_placeholder.image(image_rgb, caption=f"Frame {idx + 1}/{len(image_files)}", use_container_width=True)
        time.sleep(FRAME_DELAY)



