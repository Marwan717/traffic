import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from collections import defaultdict, deque

st.set_page_config(layout="wide")
st.title("Turning Vehicle Trajectory Analyzer")

# ---------------- PARAMETERS ----------------

TURN_THRESHOLD = 35        # degrees of heading change
HISTORY_LENGTH = 50        # trajectory length

# ---------------- HELPER ----------------

def compute_heading(dx, dy):
    return np.degrees(np.arctan2(dy, dx))

# ---------------- VIDEO UPLOAD ----------------

uploaded_video = st.file_uploader(
    "Upload Intersection Video",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_video:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1 / fps

    model = YOLO("yolov8n.pt")

    prev_positions = {}
    speeds = {}

    trajectory_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
    heading_history = defaultdict(lambda: deque(maxlen=20))

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        annotated = frame.copy()

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, track_id, cls in zip(boxes, ids, classes):

                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # ---------------- SPEED ----------------
                if track_id in prev_positions:
                    dx = cx - prev_positions[track_id][0]
                    dy = cy - prev_positions[track_id][1]

                    speed_pixels = np.sqrt(dx**2 + dy**2)
                    speed_mps = (speed_pixels / 50) / dt  # approximate scaling
                    speed_kmh = speed_mps * 3.6
                    speeds[track_id] = round(speed_kmh, 1)

                    heading = compute_heading(dx, dy)
                    heading_history[track_id].append(heading)
                else:
                    speeds[track_id] = 0

                prev_positions[track_id] = (cx, cy)

                # ---------------- TRAJECTORY ----------------
                trajectory_history[track_id].append((cx, cy))

                # ---------------- TURN DETECTION ----------------
                is_turning = False

                if len(heading_history[track_id]) > 5:
                    angle_change = abs(
                        heading_history[track_id][-1] -
                        heading_history[track_id][0]
                    )

                    if angle_change > TURN_THRESHOLD:
                        is_turning = True

                # ---------------- DRAW TRAJECTORY ----------------
                for i in range(1, len(trajectory_history[track_id])):

                    color = (255, 0, 0)     # blue for straight
                    thickness = 2

                    if is_turning:
                        color = (0, 140, 255)   # orange for turning
                        thickness = 4

                    cv2.line(
                        annotated,
                        trajectory_history[track_id][i-1],
                        trajectory_history[track_id][i],
                        color,
                        thickness
                    )

                # ---------------- DRAW BOX + LABEL ----------------
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                label = model.names[int(cls)]
                speed_text = f"{label} {speeds[track_id]} km/h"

                if is_turning:
                    speed_text += "  TURN"

                cv2.putText(
                    annotated,
                    speed_text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        stframe.image(annotated, channels="BGR", use_column_width=True)

    cap.release()
