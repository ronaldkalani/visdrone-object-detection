
import streamlit as st
import torch
import cv2
import os
import numpy as np
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---- SETTINGS ----
VIDEO_DIR = r"G:\My Drive\ComputerVision Consultant\VisDrone2019-DET-train\VisDrone2019-DET-train\videos"

# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# ---- INIT TRACKER ----
tracker = DeepSort(max_age=30)

# ---- APP HEADER ----
st.set_page_config(page_title="VisDrone Object Tracking", layout="wide")
st.title("🎥 VisDrone Object Tracking with YOLOv5 + DeepSORT")
st.markdown("Upload a video or select a file to see object detection with tracking in action.")

# ---- LOAD VIDEO FILES ----
if not os.path.exists(VIDEO_DIR):
    st.error(f"Video directory not found: `{VIDEO_DIR}`")
    st.stop()

video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi'))])

if not video_files:
    st.warning("No video files found in the folder.")
    st.stop()

selected_file = st.selectbox("Choose a video:", video_files)
video_path = os.path.join(VIDEO_DIR, selected_file)

# ---- PLAY VIDEO FRAME BY FRAME ----
cap = cv2.VideoCapture(video_path)
stframe = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    # DeepSORT Tracking
    tracked = tracker.update_tracks(detections, frame=frame)

    # Draw boxes
    for obj in tracked:
        if not obj.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, obj.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj.track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    stframe.image(frame, channels="BGR", use_column_width=True)

cap.release()
