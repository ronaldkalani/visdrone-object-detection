import streamlit as st
import torch
import cv2
import numpy as np
from deepsort_tracker import track_objects

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

st.title("🎥 Object Tracking with YOLOv5 + DeepSORT")

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if video_file:
    tfile = f"temp_video.mp4"
    with open(tfile, 'wb') as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(tfile)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        tracked = track_objects(detections, frame)

        for obj in tracked:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj['track_id']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
