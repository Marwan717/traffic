# app.py — Local Test Version (Stable)

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from scipy.spatial import distance

st.set_page_config(layout="wide")
st.title("🚦 AI Intelligent Transportation System (Local Test)")

# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.header("Controls")

video_file = st.sidebar.file_uploader("Upload Video", type=["mp4","mov","avi"])
pixel_to_meter = st.sidebar.slider("Pixel → Meter Calibration", 0.01, 0.2, 0.05)
distance_threshold = st.sidebar.slider("Near Distance (px)", 30, 200, 80)
ttc_threshold = st.sidebar.slider("TTC Threshold (sec)", 0.5, 5.0, 2.0)
frame_skip = st.sidebar.slider("Frame Skip (Speed Boost)", 1, 5, 2)

run = st.sidebar.button("Start Analysis")

# -----------------------
# Simple Centroid Tracker
# -----------------------
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def update(self, detections):
        updated = {}

        for cx, cy in detections:
            matched = False
            for obj_id, prev in self.objects.items():
                if distance.euclidean(prev, (cx, cy)) < self.max_distance:
                    updated[obj_id] = (cx, cy)
                    matched = True
                    break

            if not matched:
                updated[self.next_id] = (cx, cy)
                self.next_id += 1

        self.objects = updated
        return self.objects

# -----------------------
# Speed + TTC
# -----------------------
def compute_speed(prev, curr, fps):
    d = distance.euclidean(prev, curr)
    meters = d * pixel_to_meter
    speed_mps = meters * fps
    return speed_mps * 2.237  # mph

def compute_ttc(p1, v1, p2, v2):
    relative_speed = abs(v1 - v2)
    dist_m = distance.euclidean(p1, p2) * pixel_to_meter
    if relative_speed == 0:
        return 999
    return dist_m / (relative_speed / 2.237)

# -----------------------
# Main Processing
# -----------------------
if video_file and run:

    model = YOLO("yolov8n.pt")  # lightweight model
    tracker = CentroidTracker()

    temp_video = "temp.mp4"
    with open(temp_video, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    stframe = st.empty()

    history = {}
    speeds = {}
    events = []
    heatmap = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (1280, 720))

        if heatmap is None:
            heatmap = np.zeros((720, 1280)).astype(np.float32)

        results = model(frame, classes=[2,3,5,7], verbose=False)

        detections = []
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2 = box
            cx = int((x1 + x2)/2)
            cy = int((y1 + y2)/2)
            detections.append((cx,cy))

        objects = tracker.update(detections)

        # Update speed history
        for obj_id, pos in objects.items():

            if obj_id not in history:
                history[obj_id] = []

            history[obj_id].append(pos)
            if len(history[obj_id]) > 5:
                history[obj_id].pop(0)

            if len(history[obj_id]) >= 2:
                speeds[obj_id] = compute_speed(
                    history[obj_id][-2],
                    history[obj_id][-1],
                    fps/frame_skip
                )
            else:
                speeds[obj_id] = 0

        ids = list(objects.keys())

        # -----------------------
        # Near Miss Logic
        # -----------------------
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):

                id1, id2 = ids[i], ids[j]
                p1, p2 = objects[id1], objects[id2]

                dist = distance.euclidean(p1, p2)

                if dist < distance_threshold:

                    ttc = compute_ttc(p1, speeds[id1], p2, speeds[id2])

                    if ttc < ttc_threshold:

                        heatmap[p1[1], p1[0]] += 5
                        heatmap[p2[1], p2[0]] += 5

                        events.append({
                            "frame": frame_count,
                            "vehicle_A": id1,
                            "vehicle_B": id2,
                            "speed_A_mph": round(speeds[id1],1),
                            "speed_B_mph": round(speeds[id2],1),
                            "TTC": round(ttc,2)
                        })

                        cv2.putText(frame, "NEAR MISS",
                                    (50,50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,(0,0,255),3)

        # Draw vehicles
        for obj_id, pos in objects.items():
            cv2.circle(frame, pos, 5, (0,255,0), -1)
            cv2.putText(frame,
                        f"ID:{obj_id} {round(speeds[obj_id],1)} mph",
                        (pos[0], pos[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(255,255,255),2)

        # Heatmap overlay
        hm_blur = cv2.GaussianBlur(heatmap, (31,31), 0)
        hm_norm = cv2.normalize(hm_blur, None, 0, 255, cv2.NORM_MINMAX)
        hm_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.85, hm_color, 0.15, 0)

        stframe.image(overlay, channels="BGR")

    cap.release()

    if events:
        df = pd.DataFrame(events)
        st.subheader("⚠ Near Miss Events")
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "events.csv")
    else:
        st.success("No conflicts detected.")
