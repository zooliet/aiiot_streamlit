import streamlit as st

import os
import time

from ultralytics import YOLO
import torch
import supervision as sv
import cv2

# st.set_page_config(layout="wide")
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

LINE_START = sv.Point(720, 330)
LINE_END = sv.Point(0, 330)

st.title("트래픽 카운트")
st.markdown("---")
# st.sidebar.title("Settings")

cfg_model_path = 'models/yolov8n.pt'
# check if model file is available
if not os.path.isfile(cfg_model_path):
    st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")

# load model
model = YOLO(cfg_model_path)

line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

# confidence slider
# confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

# custom classes
# if st.sidebar.checkbox("Custom Classes"):
#     model_names = list(model.model.names.values())
#     assigned_class = st.sidebar.multiselect("Select Classes", model_names, default=[model_names[0]])
#     classes = [model_names.index(name) for name in assigned_class]
#     model.classes = classes
# else:
#     model.classes = list(model.model.names.keys())

# st.sidebar.markdown("---")

vid_file = "videos/sample.mp4"
cap = cv2.VideoCapture(vid_file)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# custom_size = st.sidebar.checkbox("Custom frame size")
# if custom_size:
#     width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
#     height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)
# 
fps = 0
output = st.empty()

st1, st2, st3 = st.columns(3)
with st1:
    st.markdown("## 높이")
    st1_text = st.markdown(f"{height}")
with st2:
    st.markdown("## 폭")
    st2_text = st.markdown(f"{width}")
with st3:
    st.markdown("## FPS")
    st3_text = st.markdown(f"{fps}")


prev_time = 0
curr_time = 0
curr_frame = 0
show_interval = 10

for result in model.track(source=vid_file, show=False, stream=True, agnostic_nms=False, verbose=False):
    frame = result.orig_img
    # frame = cv2.resize(frame, (width, height))

    detections = sv.Detections.from_yolov8(result)
    # detections = detecions[detections.class_id != 0] # filter out person class

    if result.boxes.id is not None:
        detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

    labels = [
        f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id 
        in detections
    ]

    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections,
        labels=labels
    )

    line_counter.trigger(detections)
    line_annotator.annotate(frame=frame, line_counter=line_counter)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    st1_text.markdown(f"**{height}**")
    st2_text.markdown(f"**{width}**")
    st3_text.markdown(f"**{fps:.2f}**")

    if (curr_frame % show_interval == 0):
        output.image(frame)
    curr_frame += 1
    # cv2.waitKey(30)
    # if (cv2.waitKey(30) == 27):
    #     break

cap.release()
st.button("Re-run")


